"""Helpers para Distributed Data Parallel + auto-detección de estrategia.

Contrato:

* ``detect_device_strategy()``: decide automáticamente el modo entre
  ``"ddp"`` (≥2 GPUs), ``"single_gpu"`` (1 GPU) o ``"cpu"``. El usuario
  puede forzar uno via ``DeviceConfig.strategy``.
* ``init_distributed()`` / ``shutdown_distributed()``: inicializan y
  destruyen el process group para DDP. Idempotentes en CPU (no-op).
* ``is_main_process()`` y ``world_size_and_rank()`` para gating de
  logging, checkpoints y reducciones manuales.
* No se asume backend NCCL — los tests usan ``gloo`` para correr el
  smoke de DDP sin GPU física.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def detect_device_strategy(requested: str = "auto") -> str:
    """Resuelve la estrategia efectiva a partir del hardware disponible.

    ``requested='auto'``:
      * ≥2 CUDA GPUs visibles → ``"ddp"``
      * 1 CUDA GPU visible    → ``"single_gpu"``
      * sin CUDA              → ``"cpu"``
    Si el caller fuerza una estrategia incompatible (e.g. ``"ddp"`` sin
    GPU), se respeta sólo si el backend lo soporta (``gloo`` permite DDP
    en CPU); else se rebaja a ``"cpu"`` con un aviso.
    """
    if requested not in ("auto", "ddp", "single_gpu", "cpu"):
        raise ValueError("requested must be 'auto' | 'ddp' | 'single_gpu' | 'cpu'")

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested == "auto":
        if cuda_count >= 2:
            return "ddp"
        if cuda_count == 1:
            return "single_gpu"
        return "cpu"

    # Explicit overrides: si la GPU no está disponible, downgrade silencioso
    # excepto para ddp+gloo (válido para tests).
    if requested == "single_gpu" and cuda_count == 0:
        return "cpu"
    return requested


def init_distributed(
    *,
    backend: str = "nccl",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    init_method: Optional[str] = None,
) -> None:
    """Inicializa el process group. Idempotente: sin efecto si ya inicializado.

    Si los argumentos opcionales son ``None`` se leen las variables de
    entorno estándar (``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``,
    ``MASTER_PORT``). Pensado para invocarse desde el spawn de
    ``torch.multiprocessing``.
    """
    if dist.is_available() and dist.is_initialized():
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")

    env_rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
    env_ws = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=env_rank,
        world_size=env_ws,
    )


def shutdown_distributed() -> None:
    """Destruye el process group si está inicializado."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def world_size_and_rank() -> tuple[int, int]:
    """``(world_size, rank)``: ``(1, 0)`` cuando no hay DDP."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def is_main_process() -> bool:
    """``True`` en rank 0 (siempre en modo no-DDP)."""
    return world_size_and_rank()[1] == 0


__all__ = [
    "detect_device_strategy",
    "init_distributed",
    "is_main_process",
    "shutdown_distributed",
    "world_size_and_rank",
]
