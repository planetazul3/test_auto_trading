import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.models.calibration import (
    LowLatencyRollingIsotonicCalibrator,
    fast_isotonic_inference,
)


def test_fast_isotonic_matches_sklearn_within_tolerance():
    rng = np.random.default_rng(0)
    margins = np.sort(rng.standard_normal(500)).astype(np.float32)
    labels = (margins > 0.0).astype(np.int8) ^ (rng.random(500) < 0.1).astype(np.int8)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        margins, labels.astype(float)
    )
    x_th = ir.X_thresholds_.astype(np.float32)
    y_th = ir.y_thresholds_.astype(np.float32)

    test_points = np.linspace(margins.min() - 1, margins.max() + 1, 200, dtype=np.float32)
    expected = ir.predict(test_points)
    got = np.array([fast_isotonic_inference(float(t), x_th, y_th) for t in test_points])
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_calibrator_monotonic_after_fit():
    cal = LowLatencyRollingIsotonicCalibrator(window_size=2000)
    rng = np.random.default_rng(1)
    for _ in range(1500):
        m = float(rng.standard_normal())
        lbl = int(m + 0.3 * rng.standard_normal() > 0)
        cal.add_observation(m, lbl)
    cal.update_calibration_curve()
    grid = np.linspace(-3, 3, 100)
    probs = np.array([cal.calibrate_signal(float(x)) for x in grid])
    diffs = np.diff(probs)
    # Isotónica no decreciente — toleramos jitter numérico.
    assert (diffs > -1e-6).all()
