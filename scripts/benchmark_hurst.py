
import pandas as pd
import numpy as np
import time
from numba import njit

def original_calculate_hurst(series: pd.Series, window: int = 20) -> pd.Series:
    """Original implementation for benchmarking."""
    def hurst(x):
        if len(x) < 10:
            return 0.5
        lags = range(2, 10)
        tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    return series.rolling(window=window).apply(hurst, raw=True)

@njit
def fast_hurst(x):
    # Fixed lags from 2 to 9
    lags = np.arange(2, 10)
    log_lags = np.log(lags)
    log_tau = np.empty(len(lags))
    
    for i in range(len(lags)):
        lag = lags[i]
        # x is 1D array
        # subtract(x[lag:], x[:-lag])
        diff = x[lag:] - x[:-lag]
        # np.std in numba
        std_val = np.std(diff)
        log_tau[i] = np.log(np.sqrt(std_val))
    
    # Manual simple linear regression for polyfit(log_lags, log_tau, 1)
    # y = mx + c
    # m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
    n = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xx = np.sum(log_lags**2)
    sum_xy = np.sum(log_lags * log_tau)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
    return slope * 2.0

@njit
def rolling_hurst_numba(values, window):
    n = len(values)
    res = np.empty(n)
    res[:] = np.nan
    for i in range(window - 1, n):
        window_data = values[i - window + 1:i + 1]
        res[i] = fast_hurst(window_data)
    return res

if __name__ == "__main__":
    np.random.seed(42)
    size = 5000
    data = 100 + np.random.randn(size).cumsum()
    series = pd.Series(data)
    window = 20

    print(f"Benchmarking with {size} points, window={window}...")

    # Warmup Numba
    _ = rolling_hurst_numba(series.values, window)

    start = time.time()
    orig_res = original_calculate_hurst(series, window)
    orig_time = time.time() - start
    print(f"Original time: {orig_time:.4f}s")

    start = time.time()
    fast_res_values = rolling_hurst_numba(series.values, window)
    fast_res = pd.Series(fast_res_values, index=series.index)
    fast_time = time.time() - start
    print(f"Numba time: {fast_time:.4f}s")
    print(f"Speedup: {orig_time / fast_time:.2f}x")

    # Check correctness
    # Filter out NaNs for comparison
    mask = ~np.isnan(orig_res)
    diff = np.abs(orig_res[mask] - fast_res[mask]).max()
    print(f"Max difference: {diff:.2e}")
    
    if diff < 1e-10:
        print("Outputs match!")
    else:
        print("Outputs DO NOT match!")
        # Print a few samples
        idx = np.where(mask)[0][0]
        print(f"Sample at index {idx}:")
        print(f"Original: {orig_res.iloc[idx]}")
        print(f"Fast: {fast_res.iloc[idx]}")
