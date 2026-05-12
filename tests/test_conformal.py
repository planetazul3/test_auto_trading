"""Tests para Inductive Conformal Prediction (B3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.connectors.deriv.storage import CandleRow, DuckDBStore
from src.data.dataset import (
    LabelSpec,
    WindowDataset,
    WindowDatasetConfig,
)
from src.data.store_adapter import StoreView
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.conformal import (
    ConformalBundle,
    ConformalPrediction,
    InductiveConformalPredictor,
)
from src.models.ensemble import SignalPolicy


# ---------------------------------------------------------------------------
# Properties básicas de ConformalPrediction
# ---------------------------------------------------------------------------


def test_conformal_prediction_singleton_is_confident() -> None:
    p = ConformalPrediction(include_zero=False, include_one=True)
    assert p.is_confident
    assert p.predicted_class == 1


def test_conformal_prediction_ambivalent_is_not_confident() -> None:
    p = ConformalPrediction(include_zero=True, include_one=True)
    assert not p.is_confident
    assert p.predicted_class == -1


def test_conformal_prediction_empty_is_not_confident() -> None:
    p = ConformalPrediction(include_zero=False, include_one=False)
    assert not p.is_confident
    assert p.is_empty
    assert p.predicted_class == -1


# ---------------------------------------------------------------------------
# InductiveConformalPredictor
# ---------------------------------------------------------------------------


def test_icp_validates_alpha_and_labels() -> None:
    with pytest.raises(ValueError):
        InductiveConformalPredictor(alpha=0.0)
    with pytest.raises(ValueError):
        InductiveConformalPredictor(alpha=1.0)
    cp = InductiveConformalPredictor()
    with pytest.raises(ValueError):
        cp.add_observation(0.5, 2)


def test_icp_returns_ambivalent_without_enough_data() -> None:
    cp = InductiveConformalPredictor(min_observations=50)
    for _ in range(10):
        cp.add_observation(0.7, 1)
    pred = cp.predict(0.8)
    # Insuficiente data → fallback ambivalente.
    assert pred.include_zero and pred.include_one


def test_icp_coverage_empirical_at_least_target() -> None:
    """Verifica garantía de coverage marginal ≥ 1 - α en datos sintéticos
    independientes (idénticamente distribuidos)."""
    rng = np.random.default_rng(0)
    # Generar dataset binario con clasificador "honesto" cuyas
    # probabilidades reflejan la posterior real.
    n_cal = 1000
    n_test = 2000
    # p_true ∈ [0, 1]; label ~ Bernoulli(p_true). Probabilidad
    # predicha = p_true + ruido pequeño (clasificador bien calibrado).
    p_cal = rng.uniform(0.0, 1.0, size=n_cal)
    y_cal = (rng.uniform(0, 1, size=n_cal) < p_cal).astype(int)
    p_cal_pred = np.clip(p_cal + rng.normal(0, 0.02, n_cal), 0.0, 1.0)

    p_test = rng.uniform(0.0, 1.0, size=n_test)
    y_test = (rng.uniform(0, 1, size=n_test) < p_test).astype(int)
    p_test_pred = np.clip(p_test + rng.normal(0, 0.02, n_test), 0.0, 1.0)

    cp = InductiveConformalPredictor(alpha=0.1, min_observations=50)
    for p, y in zip(p_cal_pred, y_cal):
        cp.add_observation(float(p), int(y))

    covered = 0
    for p, y in zip(p_test_pred, y_test):
        pred = cp.predict(float(p))
        if (int(y) == 1 and pred.include_one) or (int(y) == 0 and pred.include_zero):
            covered += 1
    empirical_coverage = covered / n_test
    # Garantía marginal: ≥ 1 - α (con margen de tolerancia por finite-sample).
    assert empirical_coverage >= 0.85, f"coverage too low: {empirical_coverage}"


def test_icp_smaller_alpha_implies_larger_sets() -> None:
    """Con α menor, los sets deberían ser más grandes (más conservador)."""
    rng = np.random.default_rng(42)
    n = 500
    p_cal = rng.uniform(0, 1, size=n)
    y_cal = (rng.uniform(0, 1, size=n) < p_cal).astype(int)

    cp_strict = InductiveConformalPredictor(alpha=0.01, min_observations=50)
    cp_loose = InductiveConformalPredictor(alpha=0.3, min_observations=50)
    for p, y in zip(p_cal, y_cal):
        cp_strict.add_observation(float(p), int(y))
        cp_loose.add_observation(float(p), int(y))

    # En un grid de p ∈ [0, 1], contar samples con set tamaño 2 (ambivalente).
    test_ps = np.linspace(0.01, 0.99, 100)
    sets_strict = [cp_strict.predict(p) for p in test_ps]
    sets_loose = [cp_loose.predict(p) for p in test_ps]
    ambivalent_strict = sum(1 for s in sets_strict if s.include_zero and s.include_one)
    ambivalent_loose = sum(1 for s in sets_loose if s.include_zero and s.include_one)
    assert ambivalent_strict >= ambivalent_loose


def test_icp_ring_buffer_wraps_around() -> None:
    """El ring buffer descarta observaciones viejas correctamente."""
    cp = InductiveConformalPredictor(
        alpha=0.1, window_size=20, min_observations=5
    )
    # Carga 50 observaciones — sólo las últimas 20 deben quedar.
    for i in range(50):
        cp.add_observation(0.5 + i * 0.01 % 0.5, i % 2)
    assert cp.n_observations == 20


def test_icp_reset_clears_buffer() -> None:
    cp = InductiveConformalPredictor(alpha=0.1, min_observations=2)
    cp.add_observation(0.5, 1)
    cp.add_observation(0.6, 0)
    cp.reset()
    assert cp.n_observations == 0
    pred = cp.predict(0.7)
    assert pred.include_zero and pred.include_one  # fallback


# ---------------------------------------------------------------------------
# ConformalBundle
# ---------------------------------------------------------------------------


def test_bundle_add_observations_and_predict_sets() -> None:
    bundle = ConformalBundle(
        contracts=("CALLPUT", "HIGHERLOWER"),
        horizons=(1, 3),
        alpha=0.1, min_observations=20,
    )
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=(200, 2, 2)).astype(np.float32)
    labels = (probs > 0.5).astype(np.int8)  # alineadas → fácil calibración
    mask = np.ones_like(labels, dtype=bool)
    bundle.add_observations(probs, labels, mask)

    new_probs = rng.uniform(0.0, 1.0, size=(10, 2, 2)).astype(np.float32)
    sets = bundle.predict_sets(new_probs)
    assert sets.shape == (10, 2, 2, 2)
    assert sets.dtype == bool

    confident = bundle.is_confident(new_probs)
    assert confident.shape == (10, 2, 2)
    classes = bundle.predicted_classes(new_probs)
    assert classes.shape == (10, 2, 2)
    assert classes.dtype == np.int8
    assert set(np.unique(classes).tolist()).issubset({-1, 0, 1})


def test_bundle_without_calibration_returns_ambivalent_sets() -> None:
    bundle = ConformalBundle(contracts=("CALLPUT",), horizons=(1,), min_observations=50)
    probs = np.array([[[0.3]], [[0.9]]], dtype=np.float32)
    sets = bundle.predict_sets(probs)
    # Sin suficiente data → ambos incluidos por defecto.
    assert sets[..., 0].all() and sets[..., 1].all()


def test_bundle_coverage_report_lists_calibrated_cells() -> None:
    bundle = ConformalBundle(contracts=("CALLPUT",), horizons=(1,), min_observations=20)
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.0, 1.0, size=(80, 1, 1)).astype(np.float32)
    labels = (probs > 0.5).astype(np.int8)
    bundle.add_observations(probs, labels)
    report = bundle.coverage_report()
    assert "CALLPUT__h1" in report
    assert report["CALLPUT__h1"]["target_coverage"] == pytest.approx(0.9)
    assert 0.0 <= report["CALLPUT__h1"]["q_alpha"] <= 1.0


# ---------------------------------------------------------------------------
# Integración con BacktestEngine (conformal_gate)
# ---------------------------------------------------------------------------


class _DeterministicModel(nn.Module):
    def __init__(self, logits_template: torch.Tensor):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._template = logits_template

    def forward(self, features, symbol_id=None, granularity_id=None):
        b = features.shape[0]
        return self._template.to(features.device).unsqueeze(0).expand(b, -1, -1) + self.dummy * 0


@pytest.fixture
def tiny_ds(tmp_path):
    db = tmp_path / "conf.db"
    store = DuckDBStore(db)
    rng = np.random.default_rng(0)
    n = 200
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    rows = [
        CandleRow(
            symbol="R_100", granularity=60,
            epoch=int(epochs[i]),
            open=float(base[i]),
            high=float(base[i] + 0.1),
            low=float(base[i] - 0.1),
            close=float(base[i]),
        )
        for i in range(n)
    ]
    store.upsert_candles(rows)
    store.close()
    store_ro = DuckDBStore(db, read_only=True)
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg = WindowDatasetConfig(
        window_size=15, horizons=(1,),
        label_specs=(LabelSpec("CALLPUT"),),
    )
    ds = WindowDataset(store_ro, StoreView("R_100", "candles", 60), cfg, emb)
    yield ds
    store_ro.close()


def test_engine_with_uncalibrated_conformal_gate_forces_no_trade(tiny_ds) -> None:
    """Sin calibración el gate fuerza NO_TRADE (set ambivalente)."""
    model = _DeterministicModel(torch.tensor([[10.0]])).eval()  # p≈1 → would CALL
    calibrator = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    gate = ConformalBundle(
        contracts=("CALLPUT",), horizons=(1,), min_observations=50
    )  # sin observaciones → fallback ambivalente
    engine = BacktestEngine(
        model=model, calibrator=calibrator,
        contracts=("CALLPUT",), horizons=(1,),
        policy=SignalPolicy(), config=BacktestConfig(),
        conformal_gate=gate,
    )
    result = engine.run(tiny_ds)
    # Todos los signals deben ser NO_TRADE (gate ambivalente).
    assert result.events
    assert all(e.signal == "NO_TRADE" for e in result.events)


def test_engine_with_confident_conformal_gate_allows_signals(tiny_ds) -> None:
    """Con calibración que predice fuertemente clase 1 para p≈1, el gate
    permite que la señal CALL pase."""
    model = _DeterministicModel(torch.tensor([[10.0]])).eval()
    calibrator = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    gate = ConformalBundle(
        contracts=("CALLPUT",), horizons=(1,),
        alpha=0.3, min_observations=20,
    )
    # Calibrar el gate con muchas observaciones donde p alto correlaciona con y=1.
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.0, 1.0, size=(n, 1, 1)).astype(np.float32)
    y = (p > 0.5).astype(np.int8)
    gate.add_observations(p, y)
    engine = BacktestEngine(
        model=model, calibrator=calibrator,
        contracts=("CALLPUT",), horizons=(1,),
        policy=SignalPolicy(), config=BacktestConfig(),
        conformal_gate=gate,
    )
    result = engine.run(tiny_ds)
    # Con calibración consistente y p≈1, al menos algunas señales deben pasar.
    signals = [e.signal for e in result.events if not e.masked]
    assert "CALL" in signals
