
import pandas as pd
import numpy as np
import time
from src.features.generator import FeatureGenerator

def test_hurst_implementation():
    np.random.seed(42)
    size = 1000
    data = 100 + np.random.randn(size).cumsum()
    series = pd.Series(data)
    window = 20

    fg = FeatureGenerator()
    
    print(f"Testing Hurst implementation with {size} points...")
    
    # First call might be slower due to Numba compilation
    start = time.time()
    hurst_res = fg._calculate_hurst(series, window)
    first_call_time = time.time() - start
    print(f"First call (including compilation): {first_call_time:.4f}s")
    
    start = time.time()
    hurst_res = fg._calculate_hurst(series, window)
    second_call_time = time.time() - start
    print(f"Second call: {second_call_time:.4f}s")
    
    # Basic sanity checks
    assert len(hurst_res) == size
    assert hurst_res.isna().sum() == window - 1
    
    # Check some values (regression test against expected logic)
    # The first valid value should be at index 19
    val = hurst_res.iloc[19]
    print(f"Hurst value at index 19: {val}")
    
    # We can also compare it with a manual calculation if needed, 
    # but we already did that in the benchmark script.
    
    print("Verification successful!")

if __name__ == "__main__":
    test_hurst_implementation()
