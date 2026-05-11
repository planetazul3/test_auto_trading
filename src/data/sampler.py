"""Sampler temporal DDP-aware con purged train/val/test split.

Características:

* Preserva el orden cronológico al particionar entre ranks (cada rank
  recibe un shard contiguo no solapado), evitando la fuga de
  información entre ventanas vecinas que ocurre con
  ``torch.utils.data.DistributedSampler`` standard.
* **Purged & embargoed split** (de Lopez de Prado, 2018): se descarta
  un margen de ``max(horizon)`` muestras entre splits y se opcionalmente
  añade un *embargo* extra para enmascarar la auto-correlación residual.
* Compatible con DDP, single-GPU y CPU: cuando ``num_replicas=1``
  degenera a un sampler secuencial puro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class PurgedSplit:
    """Índices de un split temporal purgado.

    ``train_indices``, ``val_indices``, ``test_indices`` son posiciones
    enteras del dataset (no epochs). El margen ``purge`` ya está
    descontado.
    """

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def purged_split(
    n: int,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    purge: int = 0,
    embargo: int = 0,
) -> PurgedSplit:
    """Particiona ``[0, n)`` en train/val/test con purga + embargo.

    El orden temporal se preserva: ``train < val < test``.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val + test must be < 1")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")

    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test_start = n - n_test
    val_start = test_start - n_val

    margin = purge + embargo

    train_end = max(0, val_start - margin)
    val_clean_start = val_start
    val_clean_end = max(val_clean_start, test_start - margin)

    train = np.arange(0, train_end, dtype=np.int64)
    val = np.arange(val_clean_start, val_clean_end, dtype=np.int64)
    test = np.arange(test_start, n, dtype=np.int64)
    return PurgedSplit(train, val, test)


class DistributedTimeSeriesSampler(Sampler[int]):
    """Sampler temporal con sharding contiguo entre ranks DDP.

    Parameters
    ----------
    indices:
        Lista global de índices del dataset que pertenecen a este sampler
        (típicamente la salida de ``purged_split`` para train/val/test).
    num_replicas:
        Tamaño del world (DDP). ``1`` para CPU/single-GPU.
    rank:
        ID del proceso actual. ``0`` para CPU/single-GPU.
    shuffle:
        Si ``True`` baraja el orden de las **ventanas dentro del shard
        local** (no entre shards) — útil para SGD pero mantiene el
        sharding temporal estricto entre ranks.
    drop_last:
        Si ``True`` descarta las muestras finales para que cada rank
        reciba exactamente el mismo número (requerido por
        ``DistributedDataParallel`` para no hangear en all-reduce).
    seed:
        Semilla del shuffler intra-shard. Se incrementa con ``set_epoch``.
    """

    def __init__(
        self,
        indices: Sequence[int],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = False,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        idx = np.asarray(indices, dtype=np.int64)
        total = idx.shape[0]
        if drop_last:
            per_rank = total // num_replicas
            usable = per_rank * num_replicas
            idx = idx[:usable]
        else:
            # Padded mode (no se recomienda para series temporales).
            per_rank = (total + num_replicas - 1) // num_replicas
            pad_n = per_rank * num_replicas - total
            if pad_n > 0:
                idx = np.concatenate([idx, idx[:pad_n]])

        # Shard contiguo: rank 0 obtiene el bloque más antiguo, rank N-1 el más nuevo.
        chunk = idx.shape[0] // num_replicas
        start = rank * chunk
        end = start + chunk
        self._shard = idx[start:end].copy()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            order = rng.permutation(self._shard.shape[0])
            yield from (int(self._shard[i]) for i in order)
        else:
            yield from (int(i) for i in self._shard)

    def __len__(self) -> int:
        return int(self._shard.shape[0])

    @property
    def shard_indices(self) -> np.ndarray:
        """Snapshot read-only del shard asignado (para debugging/tests)."""
        return self._shard.copy()


__all__ = [
    "DistributedTimeSeriesSampler",
    "PurgedSplit",
    "purged_split",
]
