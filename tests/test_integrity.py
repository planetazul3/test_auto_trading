import numpy as np
import pandas as pd
import pytest

from src.features.generator import FeatureGenerator
from src.utils.integrity import assert_no_target_leakage, temporal_integrity_check


def test_temporal_integrity_passes_for_causal_pipeline(synthetic_ohlcv):
    fg = FeatureGenerator(use_causal_zscore=True, window=20, mad_fallback=True)
    assert temporal_integrity_check(
        feature_pipeline=fg.generate_features,
        raw_df=synthetic_ohlcv,
        test_iterations=2,
        seed=0,
    )


def _leaky_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline intencionalmente fugada: usa la media GLOBAL de close."""
    out = df.copy()
    out["close_demeaned"] = out["close"] - out["close"].mean()
    return out


def test_temporal_integrity_detects_global_mean_leak(synthetic_ohlcv):
    with pytest.raises(AssertionError, match="LATENT LEAKAGE"):
        temporal_integrity_check(
            feature_pipeline=_leaky_pipeline,
            raw_df=synthetic_ohlcv,
            test_iterations=2,
            seed=0,
        )


def test_assert_no_target_leakage_catches_duplicated_target():
    n = 100
    df = pd.DataFrame(
        {
            "feat_a": np.arange(n, dtype=float),
            "target": np.arange(n, dtype=float),
        }
    )
    df["leaky_copy"] = df["target"].shift(-1)
    with pytest.raises(ValueError, match="fuga directa"):
        assert_no_target_leakage(df, target_col="target", horizon=1)
