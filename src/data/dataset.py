"""``WindowDataset``: ventanas deslizantes (X, y) cross-asset / cross-timeframe.

Cubre el camino completo:

  DuckDB → (epoch, columnas según kind) → features causales →
  ventanas ``(W, F)`` + labels multi-contract/multi-horizon →
  Tensores PyTorch.

Sin hardcodes: cualquier ``(symbol, kind, granularity)`` que exista en
el store puede materializarse como dataset. Para un mismo ``epoch`` de
ancla pueden coexistir múltiples contratos (CALL/PUT, HIGHER/LOWER,
TOUCH/NOTOUCH…) con su lista de horizontes.

* ``WindowSample`` (NamedTuple): paquete tensorial atómico devuelto por
  ``__getitem__``. Contiene la ventana de features, los labels por
  (contract, horizon) con su máscara de validez, el ID de símbolo y
  granularidad (para el ``AssetTimeframeEmbedding``) y el epoch de
  ancla — útil para trazabilidad / backtests.
* La pre-computación happens eagerly (todo en RAM tras ``__init__``):
  para volúmenes muy grandes el caller debería partir el rango en
  shards. Cero magia: el coste de memoria es ``O(N * F * 4 bytes +
  N * C * H * 1 byte)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.connectors.deriv.storage import DuckDBStore
from src.data.features import (
    BaseFeatureBuilder,
    FeatureBuilderConfig,
    build_feature_builder,
)
from src.data.labels import DERIV_LABELERS, IGNORE_LABEL
from src.data.store_adapter import StoreView, load_view
from src.models.conditioning import AssetTimeframeEmbedding


class WindowSample(NamedTuple):
    """Paquete tensorial atómico."""

    features: torch.Tensor          # (W, F) float32
    labels: torch.Tensor            # (C, H) int8, IGNORE_LABEL si inválido
    label_mask: torch.Tensor        # (C, H) bool: True si la etiqueta es válida
    symbol_id: torch.Tensor         # () long
    granularity_id: torch.Tensor    # () long
    anchor_epoch: torch.Tensor      # () int64


@dataclass
class LabelSpec:
    """Especificación de un cabezal de etiquetado.

    Combina:
      - ``contract``: clave del cabezal (alineada con
        ``MultiContractMultiHorizonHead.contracts``).
      - ``labeler``: callable que materializa labels (default: lookup
        en ``DERIV_LABELERS``).
      - ``kwargs``: parámetros del labeler (e.g. ``barrier_pct``).
    """

    contract: str
    labeler: Optional[Any] = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Any:
        if self.labeler is not None:
            return self.labeler
        if self.contract in DERIV_LABELERS:
            return DERIV_LABELERS[self.contract]
        raise ValueError(
            f"no labeler registered for contract {self.contract!r}; "
            "pass one explicitly via LabelSpec(labeler=...)"
        )


@dataclass
class WindowDatasetConfig:
    """Hiperparámetros del dataset."""

    window_size: int = 60
    stride: int = 1
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    label_specs: tuple[LabelSpec, ...] = field(
        default_factory=lambda: (LabelSpec("CALLPUT"),)
    )
    feature_config: FeatureBuilderConfig = field(default_factory=FeatureBuilderConfig)
    drop_warmup: bool = True

    def __post_init__(self) -> None:
        if self.window_size <= 1:
            raise ValueError("window_size must be > 1")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be > 0")
        if not self.label_specs:
            raise ValueError("label_specs must be non-empty")


class WindowDataset(Dataset[WindowSample]):
    """Dataset de ventanas (X, y) sobre un slice ``(symbol, kind, granularity)``.

    Parameters
    ----------
    store / view:
        Store y vista canónica de donde leer la serie.
    config:
        Hiperparámetros (ver ``WindowDatasetConfig``).
    embedding:
        ``AssetTimeframeEmbedding`` para mapear ``(symbol, granularity)``
        a IDs reproducibles. Se reutiliza entre datasets del mismo job
        de entrenamiento; el dataset auto-registra el símbolo y la
        granularidad si no están aún catalogados.
    price_column:
        Columna de precio para los labelers. Default: ``"close"`` para
        candles, ``"quote"`` para ticks.
    """

    def __init__(
        self,
        store: DuckDBStore,
        view: StoreView,
        config: WindowDatasetConfig,
        embedding: AssetTimeframeEmbedding,
        *,
        price_column: Optional[str] = None,
    ) -> None:
        self.store = store
        self.view = view
        self.config = config
        self.embedding = embedding
        self.price_column = price_column or (
            "close" if view.kind == "candles" else "quote"
        )

        df = load_view(store, view)
        if len(df) < config.window_size + max(config.horizons) + 1:
            raise ValueError(
                f"not enough rows ({len(df)}) for window={config.window_size} "
                f"and max horizon={max(config.horizons)}"
            )

        self.builder: BaseFeatureBuilder = build_feature_builder(
            view.kind, config=config.feature_config
        )
        feats = self.builder.fit_transform(df)
        self._features = feats  # (N, F) float32

        prices = df[self.price_column].astype(np.float64).to_numpy()
        self._epochs = df["epoch"].astype(np.int64).to_numpy()

        # Construir labels por (contract, horizon). Stack final: (N, C, H) int8.
        contract_keys: list[str] = []
        per_contract: list[np.ndarray] = []
        for spec in config.label_specs:
            labeler = spec.resolve()
            mapping = labeler(prices, list(config.horizons), **spec.kwargs)
            arr = np.stack([mapping[h] for h in config.horizons], axis=1)  # (N, H)
            per_contract.append(arr)
            contract_keys.append(spec.contract)
        self._labels = np.stack(per_contract, axis=1).astype(np.int8)  # (N, C, H)
        self.contract_keys = tuple(contract_keys)

        # Posiciones de ancla válidas: hay ventana completa + algún label válido.
        warmup = config.window_size - 1 if config.drop_warmup else 0
        # No filtramos por label_mask; los samples con todas las labels
        # inválidas pueden ser usados para warm-up del calibrador.
        indices = np.arange(warmup, feats.shape[0], dtype=np.int64)
        self._anchor_indices = indices[:: config.stride]

        self._symbol_id = embedding.register_symbol(view.symbol)
        self._granularity_id = embedding.register_granularity(view.granularity)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._anchor_indices.shape[0])

    def __getitem__(self, idx: int) -> WindowSample:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        t = int(self._anchor_indices[idx])
        w = self.config.window_size
        feats = self._features[t - (w - 1) : t + 1]  # (W, F)
        labels = self._labels[t]                      # (C, H)
        mask = labels != IGNORE_LABEL
        # Asegurar dtype al pasar a tensor.
        return WindowSample(
            features=torch.from_numpy(feats.copy()),
            labels=torch.from_numpy(labels.copy()),
            label_mask=torch.from_numpy(mask.copy()),
            symbol_id=torch.tensor(self._symbol_id, dtype=torch.long),
            granularity_id=torch.tensor(self._granularity_id, dtype=torch.long),
            anchor_epoch=torch.tensor(int(self._epochs[t]), dtype=torch.int64),
        )

    # ------------------------------------------------------------------
    # Metadatos
    # ------------------------------------------------------------------

    @property
    def num_features(self) -> int:
        return self.builder.num_features

    @property
    def feature_names(self) -> list[str]:
        return self.builder.feature_names

    @property
    def anchor_epochs(self) -> np.ndarray:
        return self._epochs[self._anchor_indices]


class MultiSymbolWindowDataset(ConcatDataset):
    """Concatena varios ``WindowDataset`` preservando los IDs por sample.

    Cada sub-dataset registra su propio ``(symbol_id, granularity_id)`` en el
    ``AssetTimeframeEmbedding`` compartido. Como cada ``WindowSample`` ya
    lleva esos IDs adentro, ``ConcatDataset`` los mantiene intactos al
    indexar — no hay riesgo de mezclar símbolos entre ranks.

    Conveniencia: ``num_features`` y ``feature_names`` se exigen idénticos
    en todos los sub-datasets (mismo schema → mismo input layer del modelo).
    Si no coinciden, falla rápido al construir.
    """

    def __init__(self, datasets: Iterable[WindowDataset]) -> None:
        ds_list = list(datasets)
        if not ds_list:
            raise ValueError("datasets must be non-empty")
        n_feats = ds_list[0].num_features
        names = ds_list[0].feature_names
        for d in ds_list[1:]:
            if d.num_features != n_feats:
                raise ValueError(
                    f"all datasets must share num_features; got {n_feats} and "
                    f"{d.num_features}"
                )
            if d.feature_names != names:
                raise ValueError(
                    "all datasets must share feature_names (same schema)"
                )
        super().__init__(ds_list)
        self._num_features = n_feats
        self._feature_names = list(names)

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def sub_datasets(self) -> list[WindowDataset]:
        return list(self.datasets)  # type: ignore[return-value]


def collate_window_samples(batch: Sequence[WindowSample]) -> dict[str, torch.Tensor]:
    """Collate function compatible con ``DataLoader``."""
    if not batch:
        raise ValueError("empty batch")
    return {
        "features": torch.stack([b.features for b in batch], dim=0),
        "labels": torch.stack([b.labels for b in batch], dim=0),
        "label_mask": torch.stack([b.label_mask for b in batch], dim=0),
        "symbol_id": torch.stack([b.symbol_id for b in batch], dim=0),
        "granularity_id": torch.stack([b.granularity_id for b in batch], dim=0),
        "anchor_epoch": torch.stack([b.anchor_epoch for b in batch], dim=0),
    }


__all__ = [
    "LabelSpec",
    "MultiSymbolWindowDataset",
    "WindowDataset",
    "WindowDatasetConfig",
    "WindowSample",
    "collate_window_samples",
]
