"""Simulador event-driven para contratos binarios Deriv.

El backtester recorre el dataset secuencialmente:

1. Para cada sample ``t`` con su ventana ``(W, F)`` ya calculada y sus
   labels ``(C, H)`` futuras conocidas, ejecuta el modelo + calibrador
   para obtener probabilidades calibradas.
2. Aplica ``SignalPolicy`` por celda (contract, horizon) → decisión
   {CALL, PUT, NO_TRADE} + sizing multiplier.
3. Resuelve el trade contra la label real:
   * CALL acierta si ``label == 1`` (e.g. price up).
   * PUT acierta si ``label == 0`` (price down/flat).
   * NO_TRADE → return 0.
   * Si la label está enmascarada (``IGNORE_LABEL``), el trade
     se cuenta como NO_TRADE.
4. Computa el PnL con el payout binario: ``win → +payout * sizing``,
   ``lose → -stake * sizing``. ``commission`` se resta de cada trade
   no-NO_TRADE.

Diseño:

* **Determinístico**: dado el mismo modelo + dataset + policy + seed,
  produce el mismo resultado bit-exacto.
* **Sin leakage**: el orden temporal del dataset se respeta y nunca se
  mira más allá del horizonte de la label.
* **Multi-contract / multi-horizon**: el resultado lleva una fila por
  (timestamp, contract, horizon) y se puede agregar de varias maneras.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import collate_window_samples
from src.data.labels import IGNORE_LABEL
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.conformal import ConformalBundle
from src.models.ensemble import SignalPolicy


@dataclass(frozen=True)
class BacktestConfig:
    """Parámetros económicos del backtest binario."""

    # Payout neto si la opción gana (típico Deriv 0.85 = 85% return).
    payout_on_win: float = 0.85
    # Pérdida si la opción pierde (1.0 = pierde todo el stake).
    loss_on_lose: float = 1.0
    # Comisión por trade no-NO_TRADE (fracción del stake).
    commission: float = 0.0
    # Stake base; el sizing multiplier de la policy escala sobre esto.
    base_stake: float = 1.0
    # Si True, los trades con label enmascarada se cuentan como skip
    # (no entran en n_trades). Si False, cuentan como NO_TRADE.
    skip_masked_labels: bool = True
    batch_size: int = 64

    def __post_init__(self) -> None:
        if self.payout_on_win <= 0:
            raise ValueError("payout_on_win must be > 0")
        if self.loss_on_lose <= 0:
            raise ValueError("loss_on_lose must be > 0")
        if self.commission < 0:
            raise ValueError("commission must be >= 0")
        if self.base_stake <= 0:
            raise ValueError("base_stake must be > 0")


@dataclass(frozen=True)
class TradeEvent:
    """Un trade resuelto."""

    epoch: int
    contract: str
    horizon: int
    p_calibrated: float
    signal: str             # "CALL" | "PUT" | "NO_TRADE"
    sizing_multiplier: float
    label: int              # 0/1 o IGNORE_LABEL
    pnl: float              # neto, con commission
    masked: bool            # True si label era IGNORE_LABEL


@dataclass
class BacktestResult:
    """Resultado completo del backtest."""

    events: list[TradeEvent]

    def as_dataframe(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=[
                "epoch", "contract", "horizon", "p_calibrated",
                "signal", "sizing_multiplier", "label", "pnl", "masked",
            ])
        return pd.DataFrame([e.__dict__ for e in self.events])

    def returns_by_contract(self) -> dict[str, np.ndarray]:
        df = self.as_dataframe()
        out: dict[str, np.ndarray] = {}
        for c, grp in df.groupby("contract"):
            out[str(c)] = grp["pnl"].to_numpy(dtype=np.float64)
        return out

    def total_returns(self) -> np.ndarray:
        return np.array([e.pnl for e in self.events], dtype=np.float64)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Recorre un dataset y simula trades con modelo + calibrador.

    Parameters
    ----------
    model:
        Modelo entrenado (``BackboneWithHeads`` o cualquier callable
        ``(features, symbol_id, granularity_id) -> (B, C, H) logits``).
    calibrator:
        ``PerContractCalibratorBundle`` ajustado. Si está vacío (no
        ``is_fitted``), se usan los logits como sigmoid directamente
        (fallback razonable para smoke tests).
    contracts / horizons:
        Mismo orden que el output del modelo.
    policy:
        ``SignalPolicy`` con los umbrales para mapear ``p`` → {CALL,PUT,NO_TRADE}.
    config:
        ``BacktestConfig`` con payout/commission/stake.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        calibrator: PerContractCalibratorBundle,
        contracts: Sequence[str],
        horizons: Sequence[int],
        *,
        policy: Optional[SignalPolicy] = None,
        config: Optional[BacktestConfig] = None,
        device: Optional[torch.device] = None,
        conformal_gate: Optional[ConformalBundle] = None,
        risk_manager: Optional["object"] = None,
    ) -> None:
        if len(contracts) == 0 or len(horizons) == 0:
            raise ValueError("contracts and horizons must be non-empty")
        self.model = model
        self.calibrator = calibrator
        self.contracts = tuple(contracts)
        self.horizons = tuple(int(h) for h in horizons)
        self.policy = policy or SignalPolicy()
        self.config = config or BacktestConfig()
        self.device = device or _model_device(model)
        self.conformal_gate = conformal_gate
        self.risk_manager = risk_manager

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, dataset: Dataset) -> BacktestResult:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_window_samples,
            drop_last=False,
        )
        events: list[TradeEvent] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                events.extend(self._run_batch(batch))
        return BacktestResult(events=events)

    # ------------------------------------------------------------------
    # Batch handler
    # ------------------------------------------------------------------

    def _run_batch(self, batch) -> Iterator[TradeEvent]:
        features = batch["features"].to(self.device, non_blocking=True)
        sym = batch["symbol_id"].to(self.device, non_blocking=True)
        gran = batch["granularity_id"].to(self.device, non_blocking=True)
        labels = batch["labels"].cpu().numpy()           # (B, C, H) int8
        mask = batch["label_mask"].cpu().numpy()         # (B, C, H) bool
        epochs = batch["anchor_epoch"].cpu().numpy().astype(np.int64)

        try:
            logits = self.model(features, sym, gran)
        except TypeError:
            # Modelos sin contexto (use_context=False).
            logits = self.model(features)
        logits_np = logits.detach().cpu().numpy().astype(np.float64)  # (B, C, H)

        # Calibrar: si el bundle aún no tiene curvas, usa sigmoid neutro.
        probs = self.calibrator.calibrate(logits_np)

        # Filtro conformal opcional: si el set conformal no es {0} ni {1}
        # (ambivalente o vacío), la celda se fuerza a NO_TRADE.
        if self.conformal_gate is not None:
            conformal_confident = self.conformal_gate.is_confident(probs)
        else:
            conformal_confident = None

        for bi in range(logits_np.shape[0]):
            for ci, contract in enumerate(self.contracts):
                for hi, horizon in enumerate(self.horizons):
                    p = float(probs[bi, ci, hi])
                    label_val = int(labels[bi, ci, hi])
                    is_masked = (not bool(mask[bi, ci, hi])) or label_val == IGNORE_LABEL
                    signal, sizing = _classify(p, self.policy)
                    # Conformal gate: derribar señales sin coverage garantizado.
                    if conformal_confident is not None and not bool(
                        conformal_confident[bi, ci, hi]
                    ):
                        signal, sizing = "NO_TRADE", self.policy.no_trade_sizing
                    # Risk manager opcional: ajustar sizing o forzar NO_TRADE.
                    if self.risk_manager is not None and signal != "NO_TRADE":
                        decision = self.risk_manager.evaluate(
                            contract=contract,
                            horizon=int(horizon),
                            signal=signal,
                            base_stake=self.config.base_stake,
                            sizing=sizing,
                            epoch=int(epochs[bi]),
                        )
                        if not decision.allow:
                            signal, sizing = "NO_TRADE", self.policy.no_trade_sizing
                        else:
                            sizing = decision.adjusted_sizing
                    pnl = self._resolve_pnl(signal, sizing, label_val, is_masked)
                    if is_masked and self.config.skip_masked_labels and signal != "NO_TRADE":
                        # Trade que apuntaba a una label inválida → skip (no entra).
                        continue
                    # Notificar al risk manager del PnL realizado para que
                    # actualice su estado interno (drawdown, exposure, etc).
                    if self.risk_manager is not None and signal != "NO_TRADE":
                        self.risk_manager.record_trade(
                            contract=contract,
                            horizon=int(horizon),
                            signal=signal,
                            pnl=pnl,
                            epoch=int(epochs[bi]),
                        )
                    yield TradeEvent(
                        epoch=int(epochs[bi]),
                        contract=contract,
                        horizon=horizon,
                        p_calibrated=p,
                        signal=signal,
                        sizing_multiplier=sizing,
                        label=label_val if not is_masked else IGNORE_LABEL,
                        pnl=pnl,
                        masked=is_masked,
                    )

    # ------------------------------------------------------------------
    # PnL resolution
    # ------------------------------------------------------------------

    def _resolve_pnl(
        self, signal: str, sizing: float, label: int, masked: bool
    ) -> float:
        if signal == "NO_TRADE":
            return 0.0
        if masked:
            return 0.0
        stake = self.config.base_stake * sizing
        commission = self.config.commission * stake
        # Determinar acierto.
        won = (signal == "CALL" and label == 1) or (signal == "PUT" and label == 0)
        if won:
            return float(stake * self.config.payout_on_win - commission)
        return float(-stake * self.config.loss_on_lose - commission)


# ---------------------------------------------------------------------------
# Helpers locales
# ---------------------------------------------------------------------------


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _classify(p: float, policy: SignalPolicy) -> tuple[str, float]:
    if p >= policy.call_threshold:
        sizing = policy.strong_sizing if p >= policy.strong_call_threshold else policy.normal_sizing
        return "CALL", sizing
    if p <= policy.put_threshold:
        sizing = policy.strong_sizing if p <= policy.strong_put_threshold else policy.normal_sizing
        return "PUT", sizing
    return "NO_TRADE", policy.no_trade_sizing


__all__ = ["BacktestConfig", "BacktestEngine", "BacktestResult", "TradeEvent"]
