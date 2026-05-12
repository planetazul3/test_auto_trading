"""Backtester walk-forward para contratos binarios Deriv.

Componentes:

* ``engine``: simulador event-driven que materializa una serie de
  trades sobre un dataset histórico ya etiquetado (``WindowDataset``)
  + modelo entrenado + calibrador. Aplica una ``SignalPolicy`` para
  decidir lado/sizing y un payout simple para los contratos binarios.
* ``metrics``: PnL acumulado, Sharpe, Sortino, max drawdown, win rate,
  profit factor, hit ratio por contrato/horizonte.
* ``walk_forward``: orquestrador que, dado un dataset largo, hace
  splits temporales rolling (purga + embargo) entrenando y evaluando
  iterativamente. Devuelve una tabla de métricas por fold para
  diagnosticar drift y over-tuning.

El backtester **no** ejecuta órdenes reales: trabaja sobre las labels
del dataset (``IGNORE_LABEL`` mascable) y reusa los mismos labelers que
se usaron al entrenar, así no hay leakage por inconsistencia de
definición.
"""

from .engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    TradeEvent,
)
from .metrics import BacktestMetrics, compute_metrics
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardOrchestrator,
    WalkForwardResult,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "TradeEvent",
    "WalkForwardConfig",
    "WalkForwardOrchestrator",
    "WalkForwardResult",
    "compute_metrics",
]
