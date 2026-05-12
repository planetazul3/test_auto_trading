"""Risk manager con kill-switch y caps configurables.

Filosofía: el manager es **declarativo** — los límites se especifican
en ``RiskConfig`` y el manager los aplica en tiempo real sin que el
caller tenga que duplicar lógica de chequeo.

Reglas soportadas:

* **Max drawdown absoluto**: si el peak-to-trough excede ``max_drawdown``,
  todos los trades subsiguientes son bloqueados hasta ``reset``.
* **Pérdida diaria máxima**: si la suma de PnL del "día" actual cae
  bajo ``-max_daily_loss``, kill-switch hasta el próximo día.
* **Trades por día**: cap a ``max_trades_per_day``.
* **Exposición concurrente**: cap al stake total simultáneo
  ``max_concurrent_exposure``. (Para contratos binarios Deriv resueltos
  instantáneamente en backtest, esto equivale al stake del último trade;
  en live loop puede haber ``H>1`` trades abiertos al mismo tiempo.)
* **Cap por símbolo/contrato**: cap diferenciado por contrato.

Diseño:

* Sin estado global compartido: cada ``RiskManager`` es independiente
  y reseteable.
* Thread-safe vía un único ``Lock``: el live loop puede consultar y
  actualizar concurrentemente.
* ``epoch`` siempre lo pasa el caller (no se usa wall-clock interno)
  para mantener determinismo en backtest y reproducibilidad en replay.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Mapping, Optional


# ---------------------------------------------------------------------------
# Config + decisión
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    """Reglas declarativas del manager.

    Cualquier campo ``None`` o ``inf`` deshabilita esa regla. Los caps
    de PnL son **valores absolutos** (no porcentajes) para que el caller
    sea explícito sobre la moneda/escala.
    """

    max_drawdown: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_trades_per_day: Optional[int] = None
    max_concurrent_exposure: Optional[float] = None
    max_trades_per_contract: Optional[Mapping[str, int]] = None
    seconds_per_day: int = 86_400

    def __post_init__(self) -> None:
        if self.max_drawdown is not None and self.max_drawdown <= 0:
            raise ValueError("max_drawdown must be > 0 or None")
        if self.max_daily_loss is not None and self.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be > 0 or None")
        if self.max_trades_per_day is not None and self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be > 0 or None")
        if (
            self.max_concurrent_exposure is not None
            and self.max_concurrent_exposure <= 0
        ):
            raise ValueError("max_concurrent_exposure must be > 0 or None")
        if self.seconds_per_day <= 0:
            raise ValueError("seconds_per_day must be > 0")


@dataclass(frozen=True)
class RiskDecision:
    """Resultado de ``RiskManager.evaluate``."""

    allow: bool
    adjusted_sizing: float = 0.0
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class RiskState:
    """Estado mutable mantenido por el manager."""

    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    daily_trade_count_by_contract: dict[str, int] = field(default_factory=dict)
    open_exposure: float = 0.0
    last_epoch_day: Optional[int] = None
    kill_switch_engaged: bool = False
    kill_switch_reason: Optional[str] = None

    def reset_day(self) -> None:
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.daily_trade_count_by_contract = {}
        self.kill_switch_engaged = False
        self.kill_switch_reason = None

    def reset_all(self) -> None:
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.daily_trade_count_by_contract = {}
        self.open_exposure = 0.0
        self.last_epoch_day = None
        self.kill_switch_engaged = False
        self.kill_switch_reason = None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class RiskManager:
    """Aplica las reglas de ``RiskConfig`` sobre el estado mutable."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self.state = RiskState()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        contract: str,
        horizon: int,
        signal: str,
        base_stake: float,
        sizing: float,
        epoch: int,
    ) -> RiskDecision:
        """Pre-trade: decide si permitir, ajustar sizing o bloquear.

        ``signal`` se asume ya en ``{"CALL", "PUT"}`` — los NO_TRADE no
        deberían llamar aquí.
        """
        if base_stake <= 0:
            raise ValueError("base_stake must be > 0")
        if sizing < 0:
            raise ValueError("sizing must be >= 0")
        with self._lock:
            self._roll_day(epoch)
            # Drawdown kill-switch.
            if self.state.kill_switch_engaged:
                return RiskDecision(
                    allow=False, adjusted_sizing=0.0,
                    reason=self.state.kill_switch_reason,
                )
            if self._exceeds_max_drawdown():
                self._engage_kill_switch("drawdown")
                return RiskDecision(
                    allow=False, adjusted_sizing=0.0,
                    reason="drawdown",
                )
            # Daily loss kill-switch.
            if (
                self.config.max_daily_loss is not None
                and self.state.daily_pnl <= -float(self.config.max_daily_loss)
            ):
                self._engage_kill_switch("daily_loss")
                return RiskDecision(
                    allow=False, adjusted_sizing=0.0,
                    reason="daily_loss",
                )
            # Trades-per-day cap.
            if (
                self.config.max_trades_per_day is not None
                and self.state.daily_trade_count >= int(self.config.max_trades_per_day)
            ):
                return RiskDecision(
                    allow=False, adjusted_sizing=0.0,
                    reason="max_trades_per_day",
                )
            # Cap por contrato.
            if self.config.max_trades_per_contract is not None:
                limit = self.config.max_trades_per_contract.get(contract)
                if limit is not None:
                    used = self.state.daily_trade_count_by_contract.get(contract, 0)
                    if used >= int(limit):
                        return RiskDecision(
                            allow=False, adjusted_sizing=0.0,
                            reason=f"max_trades_per_contract:{contract}",
                        )
            # Exposure cap: si el stake propuesto excede lo restante,
            # reducir el sizing (no bloquear). Si el restante es 0, bloquear.
            proposed_stake = base_stake * sizing
            if self.config.max_concurrent_exposure is not None:
                remaining = (
                    float(self.config.max_concurrent_exposure)
                    - self.state.open_exposure
                )
                if remaining <= 0:
                    return RiskDecision(
                        allow=False, adjusted_sizing=0.0,
                        reason="max_concurrent_exposure",
                    )
                if proposed_stake > remaining:
                    new_sizing = remaining / base_stake
                    return RiskDecision(
                        allow=True, adjusted_sizing=float(new_sizing),
                        reason="reduced_by_exposure",
                    )
            return RiskDecision(allow=True, adjusted_sizing=float(sizing), reason=None)

    def record_trade(
        self,
        *,
        contract: str,
        horizon: int,
        signal: str,
        pnl: float,
        epoch: int,
        stake: Optional[float] = None,
    ) -> None:
        """Post-trade: actualiza el estado (pnl, contadores, drawdown)."""
        with self._lock:
            self._roll_day(epoch)
            self.state.cumulative_pnl += float(pnl)
            self.state.peak_pnl = max(self.state.peak_pnl, self.state.cumulative_pnl)
            self.state.current_drawdown = max(
                0.0, self.state.peak_pnl - self.state.cumulative_pnl
            )
            self.state.daily_pnl += float(pnl)
            self.state.daily_trade_count += 1
            self.state.daily_trade_count_by_contract[contract] = (
                self.state.daily_trade_count_by_contract.get(contract, 0) + 1
            )
            if stake is not None:
                # En backtest binario el trade se resuelve instantáneamente
                # (la exposición no persiste). En live loop con H > 1 paso,
                # el caller puede llamar release_exposure cuando el contrato
                # expira.
                self.state.open_exposure += float(stake)

    def release_exposure(self, stake: float) -> None:
        """Libera ``stake`` de la exposición concurrente (live loop)."""
        with self._lock:
            self.state.open_exposure = max(0.0, self.state.open_exposure - float(stake))

    def reset(self) -> None:
        with self._lock:
            self.state.reset_all()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _roll_day(self, epoch: int) -> None:
        day = epoch // int(self.config.seconds_per_day)
        if self.state.last_epoch_day is None:
            self.state.last_epoch_day = day
            return
        if day != self.state.last_epoch_day:
            self.state.reset_day()
            self.state.last_epoch_day = day

    def _exceeds_max_drawdown(self) -> bool:
        if self.config.max_drawdown is None:
            return False
        return self.state.current_drawdown >= float(self.config.max_drawdown)

    def _engage_kill_switch(self, reason: str) -> None:
        self.state.kill_switch_engaged = True
        self.state.kill_switch_reason = reason


__all__ = ["RiskConfig", "RiskDecision", "RiskManager", "RiskState"]
