"""Pipeline dinámico de features para ticks y candles Deriv.

Diseño:

* ``BaseFeatureBuilder`` define el contrato común: ``fit_transform`` y
  ``transform`` operan sobre DataFrames canónicos (epoch + columnas
  específicas del tipo) y devuelven una matriz ``np.ndarray`` y la
  lista de nombres de features. Causales por construcción.
* ``CandleFeatureBuilder`` opera sobre OHLC: returns logarítmicos en
  múltiples ventanas, volatilidad realizada, rango high-low, momentum,
  z-scores rodantes — todo computado con `pandas` sin look-ahead.
* ``TickFeatureBuilder`` opera sobre ticks: spread, mid, micro-returns,
  e intensidad inter-tick (segundos entre eventos), todo causal.
* Switch dinámico vía ``build_feature_builder(kind, granularity)``: el
  caller declara qué quiere y el adaptador elige la implementación.

Cero hardcodes: las ventanas, columnas y horizontes vienen del config.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBuilderConfig:
    """Hyperparámetros del pipeline de features.

    Todos los parámetros son opcionales con defaults razonables para
    granularidades intermedias; en producción se sobreescriben desde
    ``DataConfig``.
    """

    return_windows: tuple[int, ...] = (1, 3, 5, 10)
    volatility_windows: tuple[int, ...] = (5, 10, 20)
    zscore_window: int = 20
    zscore_min_periods: int = 5
    momentum_windows: tuple[int, ...] = (3, 5, 10)
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if not self.return_windows:
            raise ValueError("return_windows must be non-empty")
        if any(w <= 0 for w in self.return_windows):
            raise ValueError("return_windows must be > 0")
        if self.zscore_window <= self.zscore_min_periods:
            raise ValueError("zscore_window must be > zscore_min_periods")


class BaseFeatureBuilder(abc.ABC):
    """Contrato común para builders de features causales."""

    def __init__(self, config: FeatureBuilderConfig = FeatureBuilderConfig()) -> None:
        self.config = config
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        if not self._feature_names:
            raise RuntimeError("call .fit_transform first")
        return list(self._feature_names)

    @property
    def num_features(self) -> int:
        return len(self._feature_names)

    @abc.abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Construye features y guarda la lista de nombres."""

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Default: re-ejecuta ``fit_transform`` (causal, sin estado)."""
        return self.fit_transform(df)


# ---------------------------------------------------------------------------
# Candle features
# ---------------------------------------------------------------------------


class CandleFeatureBuilder(BaseFeatureBuilder):
    """Features para OHLC sin look-ahead.

    Requiere columnas ``epoch, open, high, low, close``.
    """

    REQUIRED_COLUMNS = ("epoch", "open", "high", "low", "close")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"missing required column {col!r}")
        cfg = self.config
        n = len(df)
        out: list[np.ndarray] = []
        names: list[str] = []

        close = df["close"].astype(np.float64)
        high = df["high"].astype(np.float64)
        low = df["low"].astype(np.float64)
        open_ = df["open"].astype(np.float64)

        log_close = np.log(np.clip(close.to_numpy(), cfg.eps, None))

        # Returns en múltiples ventanas (log-returns).
        for w in cfg.return_windows:
            r = pd.Series(log_close).diff(w).to_numpy()
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(r)
            names.append(f"logret_{w}")

        # Volatilidad realizada (std del retorno a 1) en varias ventanas.
        ret1 = pd.Series(log_close).diff().fillna(0.0)
        for w in cfg.volatility_windows:
            v = ret1.rolling(w, min_periods=max(2, w // 2)).std().fillna(0.0).to_numpy()
            out.append(v)
            names.append(f"realized_vol_{w}")

        # Rango high-low normalizado por close (proxy de volatilidad intra-bar).
        hl_range = ((high - low) / (close + cfg.eps)).fillna(0.0).to_numpy()
        out.append(hl_range)
        names.append("hl_range_norm")

        # Body relativo (close - open) / (high - low + eps).
        body = ((close - open_) / (high - low + cfg.eps)).fillna(0.0).to_numpy()
        out.append(body)
        names.append("body_norm")

        # Momentum normalizado por volatilidad rodante.
        for w in cfg.momentum_windows:
            mom = (close - close.shift(w)).fillna(0.0)
            vol = ret1.rolling(w, min_periods=max(2, w // 2)).std().fillna(cfg.eps)
            mom_norm = (mom / (close.shift(w).abs() + cfg.eps) / (vol + cfg.eps)).fillna(0.0).to_numpy()
            out.append(np.clip(mom_norm, -10.0, 10.0))
            names.append(f"momentum_{w}")

        # Z-score causal del close.
        w = cfg.zscore_window
        roll_mean = close.rolling(w, min_periods=cfg.zscore_min_periods).mean()
        roll_std = close.rolling(w, min_periods=cfg.zscore_min_periods).std()
        z = ((close - roll_mean) / (roll_std + cfg.eps)).fillna(0.0).to_numpy()
        out.append(np.clip(z, -10.0, 10.0))
        names.append("close_zscore")

        feats = np.stack(out, axis=1).astype(np.float32)
        self._feature_names = names
        return feats


# ---------------------------------------------------------------------------
# Tick features
# ---------------------------------------------------------------------------


class TickFeatureBuilder(BaseFeatureBuilder):
    """Features para ticks (Deriv emite ``quote`` y opcionalmente ``bid``/``ask``).

    Requiere columnas ``epoch, quote`` (``bid``/``ask`` opcionales).
    """

    REQUIRED_COLUMNS = ("epoch", "quote")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"missing required column {col!r}")
        cfg = self.config
        n = len(df)
        out: list[np.ndarray] = []
        names: list[str] = []

        quote = df["quote"].astype(np.float64)
        log_quote = np.log(np.clip(quote.to_numpy(), cfg.eps, None))

        for w in cfg.return_windows:
            r = pd.Series(log_quote).diff(w).to_numpy()
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            out.append(r)
            names.append(f"tick_logret_{w}")

        ret1 = pd.Series(log_quote).diff().fillna(0.0)
        for w in cfg.volatility_windows:
            v = ret1.rolling(w, min_periods=max(2, w // 2)).std().fillna(0.0).to_numpy()
            out.append(v)
            names.append(f"tick_realized_vol_{w}")

        # Inter-tick interval (causal).
        dt = df["epoch"].astype(np.int64).diff().fillna(0.0).clip(lower=0).to_numpy().astype(np.float32)
        out.append(dt)
        names.append("inter_tick_dt")

        # Spread y mid si están disponibles.
        if "bid" in df.columns and "ask" in df.columns and df["bid"].notna().any():
            bid = df["bid"].astype(np.float64).ffill().fillna(quote)
            ask = df["ask"].astype(np.float64).ffill().fillna(quote)
            spread = (ask - bid).fillna(0.0).to_numpy()
            mid = ((ask + bid) / 2.0).fillna(quote).to_numpy()
            spread_norm = spread / (np.abs(mid) + cfg.eps)
            out.append(np.clip(spread_norm.astype(np.float32), 0.0, 10.0))
            names.append("spread_norm")
            mid_dev = (quote.to_numpy() - mid) / (np.abs(mid) + cfg.eps)
            out.append(np.clip(mid_dev.astype(np.float32), -1.0, 1.0))
            names.append("mid_deviation")

        # Z-score causal del quote.
        w = cfg.zscore_window
        roll_mean = quote.rolling(w, min_periods=cfg.zscore_min_periods).mean()
        roll_std = quote.rolling(w, min_periods=cfg.zscore_min_periods).std()
        z = ((quote - roll_mean) / (roll_std + cfg.eps)).fillna(0.0).to_numpy()
        out.append(np.clip(z, -10.0, 10.0))
        names.append("quote_zscore")

        feats = np.stack(out, axis=1).astype(np.float32)
        self._feature_names = names
        return feats


def build_feature_builder(
    kind: str,
    *,
    config: Optional[FeatureBuilderConfig] = None,
) -> BaseFeatureBuilder:
    """Selecciona el builder adecuado según el tipo de dato."""
    if kind == "candles":
        return CandleFeatureBuilder(config or FeatureBuilderConfig())
    if kind == "ticks":
        return TickFeatureBuilder(config or FeatureBuilderConfig())
    raise ValueError(f"unsupported kind: {kind!r} (expected 'ticks' or 'candles')")


__all__ = [
    "BaseFeatureBuilder",
    "CandleFeatureBuilder",
    "FeatureBuilderConfig",
    "TickFeatureBuilder",
    "build_feature_builder",
]
