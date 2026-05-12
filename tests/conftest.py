import importlib.util

import numpy as np
import pandas as pd
import pytest


# Tests que importan ``src.features.generator`` (pandas-ta) deben skipearse
# silenciosamente cuando la dependencia no está disponible — p.ej. en
# Python 3.11 donde las wheels de pandas-ta no se publican.
collect_ignore_glob: list[str] = []
if importlib.util.find_spec("pandas_ta") is None:
    collect_ignore_glob.extend(
        ["test_feature_generator.py", "test_integrity.py"]
    )


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    np.random.seed(7)
    n = 1500
    dates = pd.date_range("2025-01-01", periods=n, freq="5min")
    close = 100000 + np.random.randn(n).cumsum() * 10
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 2,
            "high": close + np.abs(np.random.randn(n) * 5),
            "low": close - np.abs(np.random.randn(n) * 5),
            "close": close,
            "volume": np.abs(np.random.randn(n) * 1000),
            "bid": close - 0.5,
            "ask": close + 0.5,
            "bid_vol": np.abs(np.random.randn(n) * 500),
            "ask_vol": np.abs(np.random.randn(n) * 500),
        },
        index=dates,
    )
