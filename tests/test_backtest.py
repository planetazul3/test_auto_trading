"""Tests del backtester walk-forward + métricas + engine + CLI.

Diseñado para ser barato:
* Datasets sintéticos pequeños sobre DuckDB en memoria.
* Modelos chicos (embedding_dim=16, lstm_hidden=16) con 1-2 epochs por fold.
* Métricas validadas por igualdad numérica sobre series construidas a mano.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
)
from src.backtest.metrics import (
    compute_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardOrchestrator,
)
from src.connectors.deriv.storage import CandleRow, DuckDBStore
from src.data.dataset import (
    LabelSpec,
    WindowDataset,
    WindowDatasetConfig,
)
from src.data.labels import IGNORE_LABEL
from src.data.store_adapter import StoreView
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import BackboneWithHeads
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.ensemble import SignalPolicy
from src.models.heads import HeadConfig
from src.training.config import (
    DataConfig,
    DeviceConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Métricas — casos analíticos
# ---------------------------------------------------------------------------


def test_win_rate_on_synthetic() -> None:
    returns = np.array([1.0, -1.0, 1.0, -1.0, 1.0])  # 3 wins de 5
    assert win_rate(returns) == pytest.approx(0.6)


def test_win_rate_empty() -> None:
    assert win_rate(np.empty(0)) == 0.0


def test_profit_factor_finite() -> None:
    returns = np.array([2.0, -1.0, 3.0, -2.0])
    # gains=5, losses=3 → 5/3
    assert profit_factor(returns) == pytest.approx(5.0 / 3.0)


def test_profit_factor_all_wins_is_inf() -> None:
    assert profit_factor(np.array([1.0, 2.0, 3.0])) == float("inf")


def test_sharpe_ratio_known_series() -> None:
    returns = np.array([0.1, -0.05, 0.2, -0.1, 0.05])
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    assert sharpe_ratio(returns) == pytest.approx(mean / std)


def test_sortino_only_downside() -> None:
    returns = np.array([0.1, -0.05, 0.2, -0.1, 0.05])
    excess = returns
    downside = excess[excess < 0]
    expected = float(np.mean(excess) / np.std(downside, ddof=1))
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_max_drawdown_on_known_curve() -> None:
    equity = np.array([0.0, 1.0, 3.0, 2.0, 1.0, 4.0])
    # Peak=3 en idx=2, trough=1 en idx=4 → dd=2, duration=2
    dd, dur = max_drawdown(equity)
    assert dd == pytest.approx(2.0)
    assert dur == 2


def test_max_drawdown_monotonic_is_zero() -> None:
    equity = np.cumsum(np.ones(10))
    dd, dur = max_drawdown(equity)
    assert dd == 0.0
    assert dur == 0


def test_compute_metrics_full_breakdown() -> None:
    returns = np.array([0.1, -0.05, 0.2, -0.1, 0.05])
    per_contract = {"CALLPUT": returns, "HIGHERLOWER": returns * 0.5}
    m = compute_metrics(returns, per_contract_returns=per_contract)
    assert m.n_trades == 5
    assert m.total_return == pytest.approx(0.2)
    assert set(m.per_contract.keys()) == {"CALLPUT", "HIGHERLOWER"}
    # Annualized Sharpe es el Sharpe escalado por √252.
    assert m.annualized_sharpe == pytest.approx(m.sharpe_ratio * np.sqrt(252.0))


# ---------------------------------------------------------------------------
# Engine — PnL determinístico
# ---------------------------------------------------------------------------


class _DeterministicModel(nn.Module):
    """Modelo que emite logits fijos pasados al constructor.

    Logits ``(B, C, H)`` se obtienen broadcasteando los pasados.
    """

    def __init__(self, logits_template: torch.Tensor):
        super().__init__()
        # Reservar al menos un parámetro para que `.parameters()` no esté vacío.
        self.dummy = nn.Parameter(torch.zeros(1))
        self._template = logits_template  # (C, H)

    def forward(self, features, symbol_id=None, granularity_id=None):
        b = features.shape[0]
        out = self._template.to(features.device).unsqueeze(0).expand(b, -1, -1)
        # Sumar un cero diferenciable para mantener el grafo bien formado.
        return out + self.dummy * 0


@pytest.fixture
def tiny_window_dataset(tmp_path):
    """Dataset chico con CALLPUT @ h=1; útil para PnL determinístico."""
    db = tmp_path / "bt.db"
    store = DuckDBStore(db)
    rng = np.random.default_rng(0)
    n = 200
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    rows = [
        CandleRow(
            symbol="R_100", granularity=60,
            epoch=int(epochs[i]),
            open=float(base[i] + rng.standard_normal() * 0.05),
            high=float(base[i] + abs(rng.standard_normal()) * 0.1),
            low=float(base[i] - abs(rng.standard_normal()) * 0.1),
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
    yield ds, emb
    store_ro.close()


def test_backtest_engine_emits_calls_when_logit_strong_positive(tiny_window_dataset) -> None:
    ds, _ = tiny_window_dataset
    # Logit muy alto → p ≈ 1.0 → CALL strong.
    template = torch.tensor([[10.0]])  # (C=1, H=1)
    model = _DeterministicModel(template).eval()
    bundle = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    # bundle.is_fitted=False → calibrate usa sigmoid; sigmoid(10) ≈ 1.0
    engine = BacktestEngine(
        model=model, calibrator=bundle,
        contracts=("CALLPUT",), horizons=(1,),
        policy=SignalPolicy(), config=BacktestConfig(),
    )
    result = engine.run(ds)
    signals = [e.signal for e in result.events if not e.masked]
    # Como p≈1, todos los samples no enmascarados deben ser CALL.
    assert signals
    assert all(s == "CALL" for s in signals)


def test_backtest_engine_emits_puts_when_logit_strong_negative(tiny_window_dataset) -> None:
    ds, _ = tiny_window_dataset
    template = torch.tensor([[-10.0]])
    model = _DeterministicModel(template).eval()
    bundle = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    engine = BacktestEngine(
        model=model, calibrator=bundle,
        contracts=("CALLPUT",), horizons=(1,),
        policy=SignalPolicy(), config=BacktestConfig(),
    )
    result = engine.run(ds)
    signals = [e.signal for e in result.events if not e.masked]
    assert signals and all(s == "PUT" for s in signals)


def test_backtest_engine_no_trade_when_in_dead_band(tiny_window_dataset) -> None:
    ds, _ = tiny_window_dataset
    template = torch.tensor([[0.0]])  # p=0.5 → NO_TRADE
    model = _DeterministicModel(template).eval()
    bundle = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    engine = BacktestEngine(
        model=model, calibrator=bundle,
        contracts=("CALLPUT",), horizons=(1,),
    )
    result = engine.run(ds)
    no_trade = [e for e in result.events if e.signal == "NO_TRADE"]
    assert len(no_trade) == len(result.events)
    assert all(e.pnl == 0.0 for e in no_trade)


def test_backtest_pnl_payout_arithmetic() -> None:
    """Verifica directamente la fórmula de payout en una sola muestra."""
    model = _DeterministicModel(torch.tensor([[10.0]])).eval()
    bundle = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    cfg = BacktestConfig(payout_on_win=0.9, loss_on_lose=1.0, commission=0.01, base_stake=10.0)
    engine = BacktestEngine(
        model=model, calibrator=bundle,
        contracts=("CALLPUT",), horizons=(1,), config=cfg,
    )
    # Simulamos un batch tipo collate manualmente.
    batch = {
        "features": torch.randn(2, 8, 4),
        "symbol_id": torch.tensor([0, 0]),
        "granularity_id": torch.tensor([0, 0]),
        "labels": torch.tensor([[[1]], [[0]]], dtype=torch.int8),  # uno wins, otro loses
        "label_mask": torch.tensor([[[True]], [[True]]]),
        "anchor_epoch": torch.tensor([100, 200], dtype=torch.int64),
    }
    events = list(engine._run_batch(batch))
    assert len(events) == 2
    win, lose = events
    # CALL acierta → stake=10*1.0(normal_sizing for strong threshold 0.80? p=1.0 → strong, sizing=1.5)
    # Como p sigmoide(10)≈1.0 > strong_call_threshold=0.80, sizing=strong_sizing=1.5
    expected_win_pnl = 10.0 * 1.5 * 0.9 - 0.01 * 10.0 * 1.5
    expected_lose_pnl = -(10.0 * 1.5 * 1.0) - 0.01 * 10.0 * 1.5
    assert win.pnl == pytest.approx(expected_win_pnl)
    assert lose.pnl == pytest.approx(expected_lose_pnl)


def test_backtest_skip_masked_labels_default() -> None:
    """skip_masked_labels=True (default): trades sobre labels enmascaradas no entran."""
    model = _DeterministicModel(torch.tensor([[10.0]])).eval()
    bundle = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    engine = BacktestEngine(
        model=model, calibrator=bundle,
        contracts=("CALLPUT",), horizons=(1,),
        config=BacktestConfig(skip_masked_labels=True),
    )
    batch = {
        "features": torch.randn(2, 8, 4),
        "symbol_id": torch.tensor([0, 0]),
        "granularity_id": torch.tensor([0, 0]),
        "labels": torch.tensor([[[1]], [[IGNORE_LABEL]]], dtype=torch.int8),
        "label_mask": torch.tensor([[[True]], [[False]]]),
        "anchor_epoch": torch.tensor([1, 2], dtype=torch.int64),
    }
    events = list(engine._run_batch(batch))
    # Sólo entra el primero; el segundo (mask=False) se skippea.
    assert len(events) == 1
    assert events[0].epoch == 1


# ---------------------------------------------------------------------------
# WalkForward orchestrator
# ---------------------------------------------------------------------------


def test_walk_forward_orchestrator_runs(tiny_window_dataset) -> None:
    ds, emb = tiny_window_dataset

    def factory():
        return BackboneWithHeads(
            num_features=ds.num_features,
            sequence_length=15,
            embedding=emb,
            head_config=HeadConfig(
                contracts=("CALLPUT",), horizons=(1,),
                use_context=True, dropout=0.0,
            ),
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            lstm_layers=1, dropout=0.0, cnn_channels=(8, 16),
        )

    cfg = TrainingConfig(
        epochs=1,
        model=ModelConfig(
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            cnn_channels=(8, 16), dropout=0.0,
        ),
        data=DataConfig(window_size=15, horizons=(1,), batch_size=16),
        optimizer=OptimizerConfig(lr=1e-3),
        device=DeviceConfig(strategy="cpu", seed=0),
    )
    wf = WalkForwardConfig(
        n_folds=2,
        initial_train_fraction=0.5,
        val_fraction_of_block=0.5,
    )
    orch = WalkForwardOrchestrator(
        dataset=ds, model_factory=factory, base_config=cfg,
        contracts=("CALLPUT",), horizons=(1,),
        walk_forward_cfg=wf,
        backtest_cfg=BacktestConfig(),
        signal_policy=SignalPolicy(),
    )
    result = orch.run()
    assert 1 <= len(result.folds) <= 2
    for f in result.folds:
        # Cada fold ejecuta train+val+test sin colapsar.
        assert f.train_range[1] > f.train_range[0]
        assert f.val_range[1] > f.val_range[0]
        assert f.test_range[1] > f.test_range[0]
        # Ranges no se solapan y respetan purga.
        assert f.train_range[1] <= f.val_range[0]
        assert f.val_range[1] <= f.test_range[0]
    agg = result.aggregate_metrics()
    assert agg.n_trades >= 0


def test_walk_forward_config_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        WalkForwardConfig(n_folds=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(initial_train_fraction=1.5)
    with pytest.raises(ValueError):
        WalkForwardConfig(mode="zigzag")


def test_walk_forward_rolling_mode_train_size_fixed(tiny_window_dataset) -> None:
    """En rolling mode, train_size se mantiene <= rolling_window."""
    ds, emb = tiny_window_dataset

    def factory():
        return BackboneWithHeads(
            num_features=ds.num_features, sequence_length=15, embedding=emb,
            head_config=HeadConfig(contracts=("CALLPUT",), horizons=(1,)),
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            lstm_layers=1, dropout=0.0, cnn_channels=(8, 16),
        )

    cfg = TrainingConfig(
        epochs=1,
        model=ModelConfig(
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            cnn_channels=(8, 16), dropout=0.0,
        ),
        data=DataConfig(window_size=15, horizons=(1,), batch_size=16),
        optimizer=OptimizerConfig(lr=1e-3),
        device=DeviceConfig(strategy="cpu", seed=0),
    )
    rolling = 50
    wf = WalkForwardConfig(
        n_folds=2,
        initial_train_fraction=0.5,
        val_fraction_of_block=0.5,
        mode="rolling",
        rolling_window=rolling,
    )
    orch = WalkForwardOrchestrator(
        dataset=ds, model_factory=factory, base_config=cfg,
        contracts=("CALLPUT",), horizons=(1,),
        walk_forward_cfg=wf,
    )
    result = orch.run()
    for f in result.folds:
        train_size = f.train_range[1] - f.train_range[0]
        assert train_size <= rolling


# ---------------------------------------------------------------------------
# CLI scripts/backtest.py
# ---------------------------------------------------------------------------


def _load_backtest_module():
    spec = importlib.util.spec_from_file_location(
        "scripts.backtest",
        Path(__file__).resolve().parent.parent / "scripts" / "backtest.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backtest_cli_walk_forward_smoke(tmp_path) -> None:
    # DB sintético.
    db = tmp_path / "wf.db"
    store = DuckDBStore(db)
    rng = np.random.default_rng(0)
    n = 250
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    store.upsert_candles([
        CandleRow(
            symbol="R_100", granularity=60,
            epoch=int(epochs[i]),
            open=float(base[i] + rng.standard_normal() * 0.05),
            high=float(base[i] + abs(rng.standard_normal()) * 0.1),
            low=float(base[i] - abs(rng.standard_normal()) * 0.1),
            close=float(base[i]),
        )
        for i in range(n)
    ])
    store.close()

    mod = _load_backtest_module()
    out_path = tmp_path / "result.json"
    rc = mod.main([
        "--mode", "walk-forward",
        "--db", str(db),
        "--symbol", "R_100",
        "--kind", "candles", "--granularity", "60",
        "--window-size", "15", "--horizons", "1",
        "--contracts", "CALLPUT",
        "--embedding-dim", "16", "--lstm-hidden", "16",
        "--num-heads", "2", "--cnn-channels", "8", "16",
        "--n-folds", "2", "--initial-train-fraction", "0.5",
        "--epochs-per-fold", "1", "--batch-size", "16",
        "--lr", "1e-3", "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["mode"] == "walk-forward"
    assert payload["n_folds"] >= 1
    assert "aggregated" in payload
    assert payload["aggregated"]["n_trades"] >= 0
