"""Capa de datos: DuckDB + Deriv → tensores PyTorch listos para entrenar.

Pieza de unión entre ``src.connectors.deriv.storage.DuckDBStore`` (ticks
y candles persistidos) y los modelos. Soporta:

* Cualquier símbolo de Deriv (cross-asset).
* Cualquier granularidad ofrecida por la API v2 (ticks + candles).
* Multi-contract / multi-horizon labels.
* DDP-aware sampler con purged time-series split.
"""

from .dataset import (
    LabelSpec,
    MultiSymbolWindowDataset,
    WindowDataset,
    WindowDatasetConfig,
    WindowSample,
    collate_window_samples,
)
from .features import (
    BaseFeatureBuilder,
    CandleFeatureBuilder,
    TickFeatureBuilder,
    build_feature_builder,
)
from .labels import (
    ContractLabeler,
    DERIV_LABELERS,
    callput_labeler,
    digit_even_odd_labeler,
    higherlower_labeler,
    touch_notouch_labeler,
)
from .sampler import DistributedTimeSeriesSampler
from .store_adapter import StoreView, load_candles, load_ticks

__all__ = [
    "BaseFeatureBuilder",
    "CandleFeatureBuilder",
    "ContractLabeler",
    "DERIV_LABELERS",
    "DistributedTimeSeriesSampler",
    "LabelSpec",
    "MultiSymbolWindowDataset",
    "StoreView",
    "TickFeatureBuilder",
    "WindowDataset",
    "WindowDatasetConfig",
    "WindowSample",
    "build_feature_builder",
    "callput_labeler",
    "collate_window_samples",
    "digit_even_odd_labeler",
    "higherlower_labeler",
    "load_candles",
    "load_ticks",
    "touch_notouch_labeler",
]
