"""Lecturas eficientes desde el ``DuckDBStore`` del conector Deriv.

Centraliza el SQL para obtener ``DataFrame``/``ndarray`` ordenados
temporalmente, evitando que las capas superiores acoplen su lógica al
schema. Si el schema evoluciona, sólo este archivo necesita cambiar.

Las consultas son **read-only** y no abren conexiones nuevas: reciben
una instancia ``DuckDBStore`` ya inicializada por el caller, lo que
permite mantener pools de conexiones y compartir cache entre módulos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.connectors.deriv.storage import DuckDBStore


@dataclass(frozen=True)
class StoreView:
    """Identifica un slice canónico del store (símbolo + tipo + granularidad)."""

    symbol: str
    kind: str  # "ticks" | "candles"
    granularity: Optional[int] = None  # None para ticks
    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ("ticks", "candles"):
            raise ValueError("kind must be 'ticks' or 'candles'")
        if self.kind == "candles" and self.granularity is None:
            raise ValueError("granularity is required for candles")
        if self.kind == "ticks" and self.granularity not in (None, 0):
            raise ValueError("ticks view must have granularity in {None, 0}")
        if self.start_epoch is not None and self.end_epoch is not None:
            if self.end_epoch < self.start_epoch:
                raise ValueError("end_epoch must be >= start_epoch")


def _range_clause(
    start: Optional[int], end: Optional[int]
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start is not None:
        clauses.append("epoch >= ?")
        params.append(int(start))
    if end is not None:
        clauses.append("epoch <= ?")
        params.append(int(end))
    return (" AND ".join(clauses), params)


def load_ticks(
    store: DuckDBStore,
    symbol: str,
    *,
    start_epoch: Optional[int] = None,
    end_epoch: Optional[int] = None,
) -> pd.DataFrame:
    """Devuelve un ``DataFrame`` ordenado cronológicamente."""
    extra, params = _range_clause(start_epoch, end_epoch)
    where = "symbol = ?"
    if extra:
        where = f"{where} AND {extra}"
    sql = (
        "SELECT epoch, quote, bid, ask, pip_size, tick_id "
        f"FROM ticks WHERE {where} ORDER BY epoch"
    )
    with store._lock:  # pylint: disable=protected-access
        df = store._conn.execute(sql, [symbol, *params]).fetchdf()  # type: ignore[attr-defined]
    return df


def load_candles(
    store: DuckDBStore,
    symbol: str,
    granularity: int,
    *,
    start_epoch: Optional[int] = None,
    end_epoch: Optional[int] = None,
) -> pd.DataFrame:
    """Devuelve OHLC + epoch en orden cronológico estricto."""
    if granularity <= 0:
        raise ValueError("granularity must be > 0")
    extra, params = _range_clause(start_epoch, end_epoch)
    where = "symbol = ? AND granularity = ?"
    if extra:
        where = f"{where} AND {extra}"
    sql = (
        "SELECT epoch, open, high, low, close "
        f"FROM candles WHERE {where} ORDER BY epoch"
    )
    with store._lock:  # pylint: disable=protected-access
        df = store._conn.execute(
            sql, [symbol, int(granularity), *params]
        ).fetchdf()  # type: ignore[attr-defined]
    return df


def load_view(store: DuckDBStore, view: StoreView) -> pd.DataFrame:
    """Carga el slice canónico identificado por ``StoreView``."""
    if view.kind == "ticks":
        return load_ticks(
            store, view.symbol,
            start_epoch=view.start_epoch, end_epoch=view.end_epoch,
        )
    return load_candles(
        store, view.symbol, int(view.granularity),  # type: ignore[arg-type]
        start_epoch=view.start_epoch, end_epoch=view.end_epoch,
    )


def list_available_views(store: DuckDBStore) -> list[StoreView]:
    """Inventaria qué ``(symbol, kind, granularity)`` hay materializados."""
    out: list[StoreView] = []
    with store._lock:  # pylint: disable=protected-access
        ticks_syms = [
            r[0] for r in store._conn.execute(  # type: ignore[attr-defined]
                "SELECT DISTINCT symbol FROM ticks ORDER BY 1"
            ).fetchall()
        ]
        candle_rows = store._conn.execute(  # type: ignore[attr-defined]
            "SELECT DISTINCT symbol, granularity FROM candles ORDER BY 1, 2"
        ).fetchall()
    for sym in ticks_syms:
        out.append(StoreView(symbol=sym, kind="ticks"))
    for sym, gran in candle_rows:
        out.append(StoreView(symbol=sym, kind="candles", granularity=int(gran)))
    return out


__all__ = [
    "StoreView",
    "list_available_views",
    "load_candles",
    "load_ticks",
    "load_view",
]
