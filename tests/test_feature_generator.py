import numpy as np
import pandas as pd

from src.features.generator import FeatureGenerator


def test_generator_produces_no_nans(synthetic_ohlcv):
    fg = FeatureGenerator(use_causal_zscore=True, window=20, mad_fallback=True)
    df = fg.generate_features(synthetic_ohlcv)
    assert df.isna().sum().sum() == 0
    assert len(df) > 0


def test_safe_causal_zscore_uses_only_past(synthetic_ohlcv):
    fg = FeatureGenerator(use_causal_zscore=True, window=20, mad_fallback=True)
    series = synthetic_ohlcv["close"]
    z = fg.safe_causal_zscore(series, window=20)
    # Mutar índice posterior no puede afectar el z-score en el pasado.
    mutated = series.copy()
    pivot = len(series) // 2
    mutated.iloc[pivot:] *= 100.0
    z_mut = fg.safe_causal_zscore(mutated, window=20)
    pd.testing.assert_series_equal(
        z.iloc[:pivot].dropna(), z_mut.iloc[:pivot].dropna()
    )


def test_hurst_in_unit_interval(synthetic_ohlcv):
    fg = FeatureGenerator()
    h = fg._calculate_hurst(synthetic_ohlcv["close"], window=20).dropna()
    assert (h >= 0.0).all()
    assert (h <= 1.0).all()


def test_regime_states_are_stable_categories(synthetic_ohlcv):
    fg = FeatureGenerator(use_causal_zscore=True, window=20, mad_fallback=True)
    df = fg.generate_features(synthetic_ohlcv)
    states = set(np.unique(df["hmm_hidden_state"].dropna().astype(int)))
    assert states.issubset({0, 1, 2})


def test_regime_features_present(synthetic_ohlcv):
    fg = FeatureGenerator(use_causal_zscore=True)
    df = fg.generate_features(synthetic_ohlcv)
    assert "atr_volatility_ratio" in df.columns
    assert "dist_from_sma200" in df.columns
    assert "hurst_exponent" in df.columns
