"""Risk management runtime.

* ``RiskConfig``: límites declarativos (drawdown, exposición, pérdida diaria, etc).
* ``RiskState``: estado mutable que el manager actualiza tras cada trade.
* ``RiskDecision``: resultado de ``evaluate`` — permite, ajusta sizing o
  bloquea con razón.
* ``RiskManager``: clase principal, integrable con ``BacktestEngine`` y
  con el live inference loop.
"""

from .manager import (
    RiskConfig,
    RiskDecision,
    RiskManager,
    RiskState,
)

__all__ = [
    "RiskConfig",
    "RiskDecision",
    "RiskManager",
    "RiskState",
]
