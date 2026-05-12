"""Métricas estándar de backtest sobre series de PnL.

Todas las funciones operan sobre ``np.ndarray`` 1-D (returns per-trade
o per-step). Sin estado, sin side-effects.

Convenciones:
* ``returns`` son retornos por trade (no por periodo de tiempo). El
  "annualization factor" lo elige el caller (e.g. ``trades_per_year``).
* ``equity_curve`` es ``cumsum(returns)``; el max drawdown se calcula
  sobre esa curva (no sobre returns).
* Win rate y profit factor ignoran trades con ``return == 0``
  (NO_TRADE) por construcción — el caller filtra antes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num / den) if den != 0.0 else default


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestMetrics:
    """Conjunto de métricas calculadas en `compute_metrics`."""

    n_trades: int
    total_return: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    avg_return: float
    std_return: float
    best_trade: float
    worst_trade: float
    annualized_sharpe: float
    annualized_return: float
    per_contract: dict[str, "BacktestMetrics"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_contract"] = {k: v.to_dict() for k, v in self.per_contract.items()}
        return d


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------


def total_return(returns: np.ndarray) -> float:
    return float(np.sum(returns))


def win_rate(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    winning = (returns > 0).sum()
    return float(winning / returns.size)


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return _safe_div(gains, losses, default=float("inf") if gains > 0 else 0.0)


def sharpe_ratio(returns: np.ndarray, *, risk_free: float = 0.0) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free
    std = float(np.std(excess, ddof=1))
    return _safe_div(float(np.mean(excess)), std)


def sortino_ratio(returns: np.ndarray, *, risk_free: float = 0.0) -> float:
    if returns.size < 2:
        return 0.0
    excess = returns - risk_free
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1))
    return _safe_div(float(np.mean(excess)), downside_std)


def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int]:
    """Devuelve ``(max_drawdown, duration_in_steps)``.

    ``max_drawdown`` es positivo (caída desde el peak). ``duration`` es
    la cantidad de pasos desde el peak hasta el trough.
    """
    if equity_curve.size == 0:
        return 0.0, 0
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve
    if drawdown.size == 0:
        return 0.0, 0
    trough_idx = int(np.argmax(drawdown))
    peak_idx = int(np.argmax(equity_curve[: trough_idx + 1])) if trough_idx > 0 else 0
    return float(drawdown[trough_idx]), int(trough_idx - peak_idx)


# ---------------------------------------------------------------------------
# Aggregador
# ---------------------------------------------------------------------------


def compute_metrics(
    returns: np.ndarray,
    *,
    trades_per_year: float = 252.0,
    per_contract_returns: Mapping[str, np.ndarray] | None = None,
) -> BacktestMetrics:
    """Compone todas las métricas. Si ``per_contract_returns`` se provee,
    calcula también el desglose por contrato."""
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 1:
        raise ValueError("returns must be 1-D")

    equity = np.cumsum(returns)
    dd, dd_dur = max_drawdown(equity)
    sr = sharpe_ratio(returns)

    ann_sr = sr * float(np.sqrt(trades_per_year)) if returns.size > 1 else 0.0
    if returns.size > 0:
        ann_ret = float(np.mean(returns)) * trades_per_year
    else:
        ann_ret = 0.0

    per_contract = {}
    if per_contract_returns is not None:
        for name, arr in per_contract_returns.items():
            per_contract[name] = compute_metrics(
                np.asarray(arr, dtype=np.float64),
                trades_per_year=trades_per_year,
                per_contract_returns=None,  # avoid recursion
            )

    return BacktestMetrics(
        n_trades=int(returns.size),
        total_return=total_return(returns),
        win_rate=win_rate(returns),
        profit_factor=profit_factor(returns),
        sharpe_ratio=sr,
        sortino_ratio=sortino_ratio(returns),
        max_drawdown=dd,
        max_drawdown_duration=dd_dur,
        avg_return=float(np.mean(returns)) if returns.size > 0 else 0.0,
        std_return=float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        best_trade=float(np.max(returns)) if returns.size > 0 else 0.0,
        worst_trade=float(np.min(returns)) if returns.size > 0 else 0.0,
        annualized_sharpe=ann_sr,
        annualized_return=ann_ret,
        per_contract=per_contract,
    )


__all__ = [
    "BacktestMetrics",
    "compute_metrics",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "win_rate",
]
