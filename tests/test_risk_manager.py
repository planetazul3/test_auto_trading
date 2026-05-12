"""Tests del RiskManager (A3): drawdown kill-switch, daily caps, exposure."""

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
from src.models.ensemble import SignalPolicy
from src.risk import RiskConfig, RiskDecision, RiskManager


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_risk_config_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        RiskConfig(max_drawdown=0)
    with pytest.raises(ValueError):
        RiskConfig(max_daily_loss=-1)
    with pytest.raises(ValueError):
        RiskConfig(max_trades_per_day=0)
    with pytest.raises(ValueError):
        RiskConfig(max_concurrent_exposure=0)
    with pytest.raises(ValueError):
        RiskConfig(seconds_per_day=0)


def test_risk_manager_default_allows_everything() -> None:
    rm = RiskManager()
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_000,
    )
    assert d.allow and d.adjusted_sizing == 1.0


# ---------------------------------------------------------------------------
# Drawdown kill-switch
# ---------------------------------------------------------------------------


def test_drawdown_kill_switch_engages_after_threshold() -> None:
    rm = RiskManager(RiskConfig(max_drawdown=15.0))
    # 3 wins de +10 → cumulative = 30, peak = 30.
    for i in range(3):
        rm.record_trade(
            contract="CALLPUT", horizon=1, signal="CALL",
            pnl=10.0, epoch=1_700_000_000 + i,
        )
    # 2 losses de -10 → cumulative = 10, drawdown = 20 >= 15 → kill.
    for i in range(2):
        rm.record_trade(
            contract="CALLPUT", horizon=1, signal="CALL",
            pnl=-10.0, epoch=1_700_000_010 + i,
        )
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_100,
    )
    assert not d.allow
    assert d.reason == "drawdown"


def test_drawdown_does_not_trigger_below_threshold() -> None:
    rm = RiskManager(RiskConfig(max_drawdown=100.0))
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=50.0, epoch=1_700_000_000,
    )
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=-30.0, epoch=1_700_000_001,
    )
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_002,
    )
    assert d.allow


# ---------------------------------------------------------------------------
# Daily loss cap
# ---------------------------------------------------------------------------


def test_daily_loss_cap_engages_and_resets_on_new_day() -> None:
    rm = RiskManager(RiskConfig(max_daily_loss=20.0, seconds_per_day=86400))
    epoch_day_one = 1_700_000_000  # day = epoch // 86400
    # Pérdidas dentro del día → daily_pnl = -25 ≤ -20 → kill.
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=-15.0, epoch=epoch_day_one,
    )
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=-10.0, epoch=epoch_day_one + 100,
    )
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=epoch_day_one + 200,
    )
    assert not d.allow
    assert d.reason == "daily_loss"

    # Saltar al día siguiente → kill se levanta automáticamente.
    epoch_day_two = epoch_day_one + 86400
    d2 = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=epoch_day_two,
    )
    assert d2.allow


# ---------------------------------------------------------------------------
# Trades-per-day cap
# ---------------------------------------------------------------------------


def test_max_trades_per_day() -> None:
    rm = RiskManager(RiskConfig(max_trades_per_day=3))
    base_epoch = 1_700_000_000
    for i in range(3):
        rm.record_trade(
            contract="CALLPUT", horizon=1, signal="CALL",
            pnl=0.0, epoch=base_epoch + i,
        )
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=base_epoch + 4,
    )
    assert not d.allow
    assert d.reason == "max_trades_per_day"


def test_per_contract_trade_cap() -> None:
    rm = RiskManager(RiskConfig(
        max_trades_per_contract={"CALLPUT": 2, "HIGHERLOWER": 5},
    ))
    base = 1_700_000_000
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL", pnl=0.0, epoch=base,
    )
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL", pnl=0.0, epoch=base + 1,
    )
    # CALLPUT bloqueado, HIGHERLOWER libre.
    d_cp = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=base + 2,
    )
    d_hl = rm.evaluate(
        contract="HIGHERLOWER", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=base + 2,
    )
    assert not d_cp.allow
    assert d_cp.reason == "max_trades_per_contract:CALLPUT"
    assert d_hl.allow


# ---------------------------------------------------------------------------
# Exposure cap (con reducción de sizing)
# ---------------------------------------------------------------------------


def test_exposure_cap_reduces_sizing() -> None:
    rm = RiskManager(RiskConfig(max_concurrent_exposure=15.0))
    # Marca 10 ya en open_exposure via record_trade con stake.
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=1.0, epoch=1_700_000_000, stake=10.0,
    )
    # Pedimos stake adicional de 10 (base=10, sizing=1) → exceed 15.
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_001,
    )
    assert d.allow
    assert d.adjusted_sizing == pytest.approx(0.5)  # 5 restante / 10 base
    assert d.reason == "reduced_by_exposure"


def test_exposure_cap_blocks_when_fully_used() -> None:
    rm = RiskManager(RiskConfig(max_concurrent_exposure=10.0))
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=0.0, epoch=1_700_000_000, stake=10.0,
    )
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_001,
    )
    assert not d.allow
    assert d.reason == "max_concurrent_exposure"


def test_release_exposure_frees_capacity() -> None:
    rm = RiskManager(RiskConfig(max_concurrent_exposure=20.0))
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=0.0, epoch=1_700_000_000, stake=20.0,
    )
    rm.release_exposure(15.0)
    d = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_001,
    )
    # 15 liberado → 5 ocupado → 15 restante → sizing 1.0 OK.
    assert d.allow
    assert d.adjusted_sizing == 1.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_state() -> None:
    rm = RiskManager(RiskConfig(max_drawdown=10.0))
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=-50.0, epoch=1_700_000_000,
    )
    rm.reset()
    assert rm.state.cumulative_pnl == 0.0
    assert not rm.state.kill_switch_engaged


# ---------------------------------------------------------------------------
# Integración con BacktestEngine
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
def tiny_loss_ds(tmp_path):
    """Dataset cuyos labels CALL/PUT son SIEMPRE 0 (close estrictamente
    decreciente). El engine emitirá CALL (logit positivo) y siempre perderá
    → ideal para forzar drawdown kill-switch."""
    db = tmp_path / "risk.db"
    store = DuckDBStore(db)
    # Generar serie estrictamente decreciente para que CALLPUT @ h=1 siempre = 0.
    n = 200
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 - np.arange(n) * 0.5  # decreciente
    rows = [
        CandleRow(
            symbol="R_100", granularity=60,
            epoch=int(epochs[i]),
            open=float(base[i] + 0.05),
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


def test_backtest_engine_drawdown_kill_switch_stops_losing_streak(tiny_loss_ds) -> None:
    """El engine emite CALL sobre serie decreciente → todos los trades pierden.
    Tras pocas pérdidas, el risk manager debe matar el resto y devolver NO_TRADE."""
    # Logit positivo → CALL. Labels siempre 0 → pierde siempre.
    model = _DeterministicModel(torch.tensor([[10.0]])).eval()
    calibrator = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    # max_drawdown bajo (= 10 unidades) y stake=1 + payout=0.85 → cada pérdida = -1.5
    # ⇒ 7 pérdidas ya rompen drawdown=10.
    rm = RiskManager(RiskConfig(max_drawdown=10.0))
    cfg = BacktestConfig(
        payout_on_win=0.85, loss_on_lose=1.0, base_stake=1.0,
    )
    engine = BacktestEngine(
        model=model, calibrator=calibrator,
        contracts=("CALLPUT",), horizons=(1,),
        policy=SignalPolicy(), config=cfg,
        risk_manager=rm,
    )
    result = engine.run(tiny_loss_ds)
    # Hay un mix de CALL (antes del kill) y NO_TRADE (después).
    signals = [e.signal for e in result.events]
    assert "CALL" in signals
    assert "NO_TRADE" in signals
    # El kill-switch debe haber engaged.
    assert rm.state.kill_switch_engaged
    assert rm.state.kill_switch_reason == "drawdown"
    # Verificar que después del primer NO_TRADE no aparece más ningún CALL.
    seen_no_trade = False
    for s in signals:
        if s == "NO_TRADE":
            seen_no_trade = True
        elif s == "CALL" and seen_no_trade:
            pytest.fail("CALL after kill-switch engaged")


def test_risk_manager_block_does_not_advance_trade_counter() -> None:
    """Los trades bloqueados no consumen el cap de trades_per_day."""
    rm = RiskManager(RiskConfig(max_trades_per_day=2, max_concurrent_exposure=10.0))
    # Bloqueamos la primera evaluación con exposure cap (saturando exposure
    # via record_trade previo con stake mayor que el cap → siempre bloqueado).
    rm.record_trade(
        contract="CALLPUT", horizon=1, signal="CALL",
        pnl=0.0, epoch=1_700_000_000, stake=10.0,
    )
    d_blocked = rm.evaluate(
        contract="CALLPUT", horizon=1, signal="CALL",
        base_stake=10.0, sizing=1.0, epoch=1_700_000_001,
    )
    assert not d_blocked.allow
    # daily_trade_count se incrementó SÓLO por el record_trade (1), no por la evaluación bloqueada.
    assert rm.state.daily_trade_count == 1
