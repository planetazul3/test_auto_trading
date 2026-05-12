"""Labelers para contratos Deriv (CALL/PUT, HIGHER/LOWER, ONETOUCH/NOTOUCH…).

Cada función toma:

* ``prices``: ``np.ndarray`` 1-D cronológica (close para candles, quote
  para ticks).
* ``horizons``: lista de horizontes ``H`` (en pasos).
* Parámetros específicos del contrato (e.g. barrera para HIGHER/LOWER).

Y devuelve un diccionario ``{horizon: labels}`` donde ``labels`` es un
``np.ndarray`` ``{0, 1}`` (1 = CALL/HIGHER/TOUCH/EVEN). Las muestras
sin suficiente futuro disponible se etiquetan como ``-1`` (ignored
mask), de modo que la loss puede enmascararlas trivialmente.

Cero hardcodes: el horizonte siempre lo decide el caller. La barrera
en HIGHER/LOWER y TOUCH/NOTOUCH se especifica como diferencia relativa
(fracción del precio actual).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


IGNORE_LABEL = -1


ContractLabeler = Callable[..., Mapping[int, np.ndarray]]


def _validate_prices_horizons(prices: np.ndarray, horizons: Sequence[int]) -> None:
    if prices.ndim != 1:
        raise ValueError("prices must be 1-D")
    if not horizons:
        raise ValueError("horizons must be non-empty")
    if any(h <= 0 for h in horizons):
        raise ValueError("all horizons must be > 0")


# ---------------------------------------------------------------------------
# CALL / PUT (sign del retorno absoluto a horizonte H)
# ---------------------------------------------------------------------------


def callput_labeler(
    prices: np.ndarray,
    horizons: Sequence[int],
    *,
    epsilon: float = 0.0,
) -> dict[int, np.ndarray]:
    """1 si ``price[t+h] > price[t] + epsilon``; 0 si ``< -epsilon``; else IGNORE.

    ``epsilon`` permite un dead-band para ignorar movimientos
    micro-ruidosos (útil para forex spreads / synthetic indices).
    """
    _validate_prices_horizons(prices, horizons)
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    n = prices.shape[0]
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        labels = np.full(n, IGNORE_LABEL, dtype=np.int8)
        valid = n - int(h)
        if valid <= 0:
            out[int(h)] = labels
            continue
        diff = prices[h : h + valid] - prices[:valid]
        if epsilon > 0:
            up = diff > epsilon
            down = diff < -epsilon
            labels[:valid] = np.where(up, 1, np.where(down, 0, IGNORE_LABEL)).astype(np.int8)
        else:
            labels[:valid] = (diff > 0).astype(np.int8)
        out[int(h)] = labels
    return out


# ---------------------------------------------------------------------------
# HIGHER / LOWER (cruce de una barrier absoluta o relativa al precio de entrada)
# ---------------------------------------------------------------------------


def higherlower_labeler(
    prices: np.ndarray,
    horizons: Sequence[int],
    *,
    barrier_pct: float = 0.0,
) -> dict[int, np.ndarray]:
    """HIGHER si ``price[t+h] > price[t] * (1 + barrier_pct)``.

    ``barrier_pct=0`` equivale al CALL/PUT estricto. Valores positivos
    exigen movimiento mínimo para etiquetar HIGHER; el equivalente
    negativo gobierna LOWER (se mantiene simétrico).
    """
    _validate_prices_horizons(prices, horizons)
    n = prices.shape[0]
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        labels = np.full(n, IGNORE_LABEL, dtype=np.int8)
        valid = n - int(h)
        if valid <= 0:
            out[int(h)] = labels
            continue
        future = prices[h : h + valid]
        anchor = prices[:valid]
        high_barrier = anchor * (1.0 + barrier_pct)
        low_barrier = anchor * (1.0 - barrier_pct)
        higher = future > high_barrier
        lower = future < low_barrier
        labels[:valid] = np.where(higher, 1, np.where(lower, 0, IGNORE_LABEL)).astype(np.int8)
        out[int(h)] = labels
    return out


# ---------------------------------------------------------------------------
# ONETOUCH / NOTOUCH (toca una barrier dentro del horizonte)
# ---------------------------------------------------------------------------


def touch_notouch_labeler(
    prices: np.ndarray,
    horizons: Sequence[int],
    *,
    barrier_pct: float = 0.01,
    direction: str = "up",
) -> dict[int, np.ndarray]:
    """1 si dentro de ``[t+1, t+h]`` la serie toca la barrier; 0 caso contrario.

    ``direction='up'``: barrier = ``anchor * (1 + barrier_pct)`` y se
    busca ``max(prices[t+1:t+h+1]) >= barrier``.
    ``direction='down'``: barrier = ``anchor * (1 - barrier_pct)`` y se
    busca ``min(prices[t+1:t+h+1]) <= barrier``.
    """
    _validate_prices_horizons(prices, horizons)
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    if barrier_pct <= 0:
        raise ValueError("barrier_pct must be > 0")
    n = prices.shape[0]
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        labels = np.full(n, IGNORE_LABEL, dtype=np.int8)
        valid = n - int(h)
        if valid <= 0:
            out[int(h)] = labels
            continue
        anchor = prices[:valid]
        # Ventana móvil del futuro: max/min de prices[t+1 .. t+h].
        # Implementación vectorizada con strides.
        if direction == "up":
            barrier = anchor * (1.0 + barrier_pct)
            future_max = _rolling_future_max(prices, int(h))
            labels[:valid] = (future_max[:valid] >= barrier).astype(np.int8)
        else:
            barrier = anchor * (1.0 - barrier_pct)
            future_min = _rolling_future_min(prices, int(h))
            labels[:valid] = (future_min[:valid] <= barrier).astype(np.int8)
        out[int(h)] = labels
    return out


def _rolling_future_max(prices: np.ndarray, horizon: int) -> np.ndarray:
    """``out[t] = max(prices[t+1 .. t+horizon])``; positions sin futuro = -inf."""
    n = prices.shape[0]
    out = np.full(n, -np.inf, dtype=prices.dtype)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if end > t + 1:
            out[t] = float(np.max(prices[t + 1 : end]))
    return out


def _rolling_future_min(prices: np.ndarray, horizon: int) -> np.ndarray:
    n = prices.shape[0]
    out = np.full(n, np.inf, dtype=prices.dtype)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if end > t + 1:
            out[t] = float(np.min(prices[t + 1 : end]))
    return out


# ---------------------------------------------------------------------------
# DIGITEVEN / DIGITODD — paridad del último dígito a horizonte H
# ---------------------------------------------------------------------------


def digit_even_odd_labeler(
    prices: np.ndarray,
    horizons: Sequence[int],
    *,
    pip_scale: float = 100.0,
) -> dict[int, np.ndarray]:
    """1 = EVEN, 0 = ODD a horizonte ``h``.

    ``pip_scale`` define la potencia de 10 que separa el último dígito
    relevante (e.g. 100 para 2 decimales = último dígito = redondeo *
    100 mod 10).
    """
    _validate_prices_horizons(prices, horizons)
    if pip_scale <= 0:
        raise ValueError("pip_scale must be > 0")
    n = prices.shape[0]
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        labels = np.full(n, IGNORE_LABEL, dtype=np.int8)
        valid = n - int(h)
        if valid <= 0:
            out[int(h)] = labels
            continue
        future = prices[h : h + valid]
        last_digit = np.floor(np.abs(future) * pip_scale + 1e-9).astype(np.int64) % 10
        labels[:valid] = ((last_digit % 2) == 0).astype(np.int8)
        out[int(h)] = labels
    return out


# Catálogo: clave = contrato del cabezal MultiContractMultiHorizonHead.
DERIV_LABELERS: dict[str, ContractLabeler] = {
    "CALLPUT": callput_labeler,
    "HIGHERLOWER": higherlower_labeler,
    "TOUCHNOTOUCH": touch_notouch_labeler,
    "DIGITEVENODD": digit_even_odd_labeler,
}


__all__ = [
    "DERIV_LABELERS",
    "IGNORE_LABEL",
    "ContractLabeler",
    "callput_labeler",
    "digit_even_odd_labeler",
    "higherlower_labeler",
    "touch_notouch_labeler",
]
