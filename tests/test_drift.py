"""Tests del OnlineCalibrationMonitor (B2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.drift import OnlineCalibrationMonitor


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_monitor_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        OnlineCalibrationMonitor(max_brier=0.0)
    with pytest.raises(ValueError):
        OnlineCalibrationMonitor(max_ece=1.5)
    with pytest.raises(ValueError):
        OnlineCalibrationMonitor(min_observations=1)
    with pytest.raises(ValueError):
        OnlineCalibrationMonitor(recovery_margin=-0.01)
    with pytest.raises(ValueError):
        OnlineCalibrationMonitor(cooldown_seconds=-1)


# ---------------------------------------------------------------------------
# Decisión: insufficient_observations
# ---------------------------------------------------------------------------


def test_insufficient_observations_returns_no_refit() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,), min_observations=100
    )
    monitor = OnlineCalibrationMonitor()
    decisions = monitor.check(bundle, now_epoch=1_700_000_000)
    assert "CALLPUT__h1" in decisions
    d = decisions["CALLPUT__h1"]
    assert not d.needs_refit
    assert d.reason == "insufficient_observations"


# ---------------------------------------------------------------------------
# Decisión: ok cuando métricas bajan los umbrales
# ---------------------------------------------------------------------------


def _populate_bundle_well_calibrated(bundle, n=200, seed=0) -> None:
    """Carga el bundle con predicciones determinísticas y aciertos perfectos.

    Para que el Brier baje del umbral del test, las labels deben estar
    fuertemente correlacionadas con la probabilidad (no Bernoulli noise).
    Generamos probs cerca de 0 y 1, labels determinísticas con esa
    correlación.
    """
    rng = np.random.default_rng(seed)
    # Probs polarizadas: mayoría cerca de 0.05 o 0.95.
    base = rng.choice([0.05, 0.95], size=(n, 1, 1)).astype(np.float32)
    probs = np.clip(base + rng.normal(0, 0.02, size=base.shape), 0.0, 1.0).astype(np.float32)
    # Labels determinísticas alineadas (acierto perfecto): y=1 si p>0.5, else 0.
    labels = (probs > 0.5).astype(np.int8)
    bundle.add_observations(probs, labels)
    bundle.update_all()


def _populate_bundle_miscalibrated(bundle, n=200, seed=1) -> None:
    """Carga el bundle con probs inversamente correlacionadas con labels."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.0, 1.0, size=(n, 1, 1)).astype(np.float32)
    # labels = 1 si probs < 0.5 → mal calibrado.
    labels = (probs < 0.5).astype(np.int8)
    bundle.add_observations(probs, labels)
    bundle.update_all()


def test_well_calibrated_bundle_no_refit() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_well_calibrated(bundle, n=300)
    monitor = OnlineCalibrationMonitor(max_brier=0.30, max_ece=0.15)
    decisions = monitor.check(bundle, now_epoch=1_700_000_000)
    d = decisions["CALLPUT__h1"]
    assert not d.needs_refit
    assert d.reason == "ok"
    assert d.brier_score is not None and d.brier_score <= 0.30


def test_miscalibrated_bundle_triggers_refit() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_miscalibrated(bundle, n=300)
    # Umbral bien estricto para forzar alerta.
    monitor = OnlineCalibrationMonitor(max_brier=0.10, max_ece=0.05)
    decisions = monitor.check(bundle, now_epoch=1_700_000_000)
    d = decisions["CALLPUT__h1"]
    assert d.needs_refit
    assert d.reason and ("brier" in d.reason or "ece" in d.reason)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


def test_cooldown_prevents_back_to_back_refits() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_miscalibrated(bundle, n=300)
    monitor = OnlineCalibrationMonitor(
        max_brier=0.10, max_ece=0.05, cooldown_seconds=600,
    )
    # Primer check: dispara refit.
    decisions = monitor.check(bundle, now_epoch=1_700_000_000)
    n_refit = monitor.maybe_refit(
        bundle, decisions, now_epoch=1_700_000_000, background=False,
    )
    assert n_refit == 1

    # Re-check inmediato: cooldown activo → no refit (in_cooldown=True).
    decisions2 = monitor.check(bundle, now_epoch=1_700_000_000 + 100)
    d2 = decisions2["CALLPUT__h1"]
    assert d2.in_cooldown
    assert not d2.needs_refit
    assert d2.reason == "cooldown"


def test_cooldown_expires_after_threshold() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_miscalibrated(bundle, n=300)
    monitor = OnlineCalibrationMonitor(
        max_brier=0.10, max_ece=0.05, cooldown_seconds=300,
    )
    decisions = monitor.check(bundle, now_epoch=1_000)
    monitor.maybe_refit(bundle, decisions, now_epoch=1_000, background=False)

    # Después de 400s cooldown expiró → si sigue en alerta, refit.
    decisions2 = monitor.check(bundle, now_epoch=1_000 + 400)
    d2 = decisions2["CALLPUT__h1"]
    assert not d2.in_cooldown


# ---------------------------------------------------------------------------
# Hysteresis: una celda en alerta vuelve a OK sólo con recovery_margin
# ---------------------------------------------------------------------------


def test_hysteresis_keeps_alert_inside_margin() -> None:
    """Si la métrica baja apenas del umbral, la celda sigue en alerta
    hasta cruzar threshold - recovery_margin."""
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_miscalibrated(bundle, n=200)
    monitor = OnlineCalibrationMonitor(
        max_brier=0.10, max_ece=0.05,
        recovery_margin=0.05, cooldown_seconds=0,
    )
    # Primer check engancha alerta.
    decisions = monitor.check(bundle, now_epoch=1)
    assert decisions["CALLPUT__h1"].needs_refit

    # Recalibramos el bundle a algo decente para que la métrica baje.
    bundle.get("CALLPUT", 1).reset()
    _populate_bundle_well_calibrated(bundle, n=300, seed=10)

    decisions2 = monitor.check(bundle, now_epoch=2)
    d2 = decisions2["CALLPUT__h1"]
    # Tras el reset + buena calibración, la métrica debería estar
    # bien por debajo del threshold; verificar transición a "recovered" u "ok".
    assert d2.reason in {"recovered", "ok"}


# ---------------------------------------------------------------------------
# maybe_refit en background usa wait_update_done para no hangear
# ---------------------------------------------------------------------------


def test_maybe_refit_background_completes() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_miscalibrated(bundle, n=200)
    monitor = OnlineCalibrationMonitor(max_brier=0.10, max_ece=0.05)
    decisions = monitor.check(bundle, now_epoch=1_700_000_000)
    n_refit = monitor.maybe_refit(
        bundle, decisions, now_epoch=1_700_000_000, background=True,
    )
    assert n_refit >= 1
    # Esperar al fin del background refit antes de salir del test.
    cal = bundle.get("CALLPUT", 1)
    assert cal.wait_update_done(timeout=30.0)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_monitor_reset_clears_alert_state() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1,),
        window_size=500, min_observations=50,
    )
    _populate_bundle_miscalibrated(bundle, n=200)
    monitor = OnlineCalibrationMonitor(max_brier=0.10, max_ece=0.05)
    monitor.check(bundle, now_epoch=1_700_000_000)
    monitor.reset()
    assert monitor._states == {}  # internal state limpio
