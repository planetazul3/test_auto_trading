"""Almacén embebido en DuckDB para ticks y velas Deriv.

Diseño:

* Una sola tabla por tipo de dato (``ticks``, ``candles``) con claves
  primarias compuestas para evitar duplicados al re-ejecutar backfills.
* UPSERTs idempotentes: ``DO NOTHING`` para ticks (inmutables) y
  ``DO UPDATE`` para candles (la última vela del intervalo se actualiza
  hasta cerrarse).
* Tabla ``ingest_runs`` con auditoría completa por lote (rango,
  estado, error) que actúa como checkpoint reanudable.
* Detección de huecos en candles vía SQL window functions sobre
  ``range(start, end+1, granularity)``.
* Export a Parquet particionado por símbolo (+ granularidad) para
  archivado a largo plazo.

Concurrencia: una conexión DuckDB envuelta en un ``threading.Lock``.
Las operaciones I/O se ejecutan vía ``asyncio.to_thread`` desde el
ingester, manteniendo el bucle async libre.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import duckdb
except ImportError:  # pragma: no cover - dependencia opcional
    duckdb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version    INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ticks (
        symbol      VARCHAR NOT NULL,
        epoch       BIGINT  NOT NULL,
        quote       DOUBLE  NOT NULL,
        bid         DOUBLE,
        ask         DOUBLE,
        pip_size    DOUBLE,
        tick_id     VARCHAR,
        received_at TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (symbol, epoch)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candles (
        symbol       VARCHAR NOT NULL,
        granularity  INTEGER NOT NULL,
        epoch        BIGINT  NOT NULL,
        open         DOUBLE  NOT NULL,
        high         DOUBLE  NOT NULL,
        low          DOUBLE  NOT NULL,
        close        DOUBLE  NOT NULL,
        received_at  TIMESTAMP DEFAULT current_timestamp,
        PRIMARY KEY (symbol, granularity, epoch)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_runs (
        run_id        BIGINT PRIMARY KEY,
        symbol        VARCHAR NOT NULL,
        kind          VARCHAR NOT NULL,
        granularity   INTEGER,
        range_start   BIGINT,
        range_end     BIGINT,
        rows_received BIGINT  DEFAULT 0,
        rows_inserted BIGINT  DEFAULT 0,
        batches       INTEGER DEFAULT 0,
        started_at    TIMESTAMP DEFAULT current_timestamp,
        finished_at   TIMESTAMP,
        status        VARCHAR DEFAULT 'running',
        error         VARCHAR
    )
    """,
    """CREATE SEQUENCE IF NOT EXISTS ingest_runs_seq START 1""",
    """CREATE INDEX IF NOT EXISTS ix_ticks_symbol_epoch ON ticks(symbol, epoch)""",
    """CREATE INDEX IF NOT EXISTS ix_candles_skey ON candles(symbol, granularity, epoch)""",
    """CREATE INDEX IF NOT EXISTS ix_runs_symbol_kind ON ingest_runs(symbol, kind, granularity)""",
)


@dataclass(frozen=True, slots=True)
class TickRow:
    symbol: str
    epoch: int
    quote: float
    bid: float | None = None
    ask: float | None = None
    pip_size: float | None = None
    tick_id: str | None = None


@dataclass(frozen=True, slots=True)
class CandleRow:
    symbol: str
    granularity: int
    epoch: int
    open: float
    high: float
    low: float
    close: float


class DuckDBStore:
    """Wrapper thread-safe sobre una conexión DuckDB con el esquema Deriv."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        if duckdb is None:
            raise RuntimeError(
                "duckdb is required; install the 'deriv-ingest' extra: "
                "pip install '.[deriv-ingest]'"
            )
        self.path = str(path)
        self.read_only = read_only
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self._init_schema()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            for ddl in _SCHEMA_DDL:
                self._conn.execute(ddl)
            current = self._conn.execute(
                "SELECT max(version) FROM schema_version"
            ).fetchone()
            if current is None or current[0] is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    [SCHEMA_VERSION],
                )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - duckdb idempotency
                pass

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_ticks(self, rows: Sequence[TickRow]) -> int:
        if not rows:
            return 0
        params = [
            (r.symbol, r.epoch, r.quote, r.bid, r.ask, r.pip_size, r.tick_id)
            for r in rows
        ]
        with self._lock:
            before = self._conn.execute("SELECT count(*) FROM ticks").fetchone()[0]
            self._conn.begin()
            try:
                self._conn.executemany(
                    """
                    INSERT INTO ticks (symbol, epoch, quote, bid, ask, pip_size, tick_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (symbol, epoch) DO NOTHING
                    """,
                    params,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            after = self._conn.execute("SELECT count(*) FROM ticks").fetchone()[0]
            return int(after - before)

    def upsert_candles(self, rows: Sequence[CandleRow]) -> int:
        if not rows:
            return 0
        params = [
            (r.symbol, r.granularity, r.epoch, r.open, r.high, r.low, r.close)
            for r in rows
        ]
        with self._lock:
            before = self._conn.execute("SELECT count(*) FROM candles").fetchone()[0]
            self._conn.begin()
            try:
                self._conn.executemany(
                    """
                    INSERT INTO candles
                        (symbol, granularity, epoch, open, high, low, close)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (symbol, granularity, epoch) DO UPDATE SET
                        open  = excluded.open,
                        high  = excluded.high,
                        low   = excluded.low,
                        close = excluded.close
                    """,
                    params,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            after = self._conn.execute("SELECT count(*) FROM candles").fetchone()[0]
            return int(after - before)

    # ------------------------------------------------------------------
    # Reads / boundaries
    # ------------------------------------------------------------------

    def latest_tick_epoch(self, symbol: str) -> int | None:
        return self._scalar(
            "SELECT max(epoch) FROM ticks WHERE symbol = ?", [symbol]
        )

    def earliest_tick_epoch(self, symbol: str) -> int | None:
        return self._scalar(
            "SELECT min(epoch) FROM ticks WHERE symbol = ?", [symbol]
        )

    def latest_candle_epoch(self, symbol: str, granularity: int) -> int | None:
        return self._scalar(
            "SELECT max(epoch) FROM candles WHERE symbol = ? AND granularity = ?",
            [symbol, granularity],
        )

    def earliest_candle_epoch(self, symbol: str, granularity: int) -> int | None:
        return self._scalar(
            "SELECT min(epoch) FROM candles WHERE symbol = ? AND granularity = ?",
            [symbol, granularity],
        )

    def count_ticks(self, symbol: str) -> int:
        return int(
            self._scalar("SELECT count(*) FROM ticks WHERE symbol = ?", [symbol]) or 0
        )

    def count_candles(self, symbol: str, granularity: int) -> int:
        return int(
            self._scalar(
                "SELECT count(*) FROM candles WHERE symbol = ? AND granularity = ?",
                [symbol, granularity],
            )
            or 0
        )

    def _scalar(self, sql: str, params: Sequence[Any]) -> int | None:
        with self._lock:
            row = self._conn.execute(sql, list(params)).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_candle_gaps(
        self,
        symbol: str,
        granularity: int,
        start: int,
        end: int,
    ) -> list[tuple[int, int]]:
        """Devuelve ``[(gap_start, gap_end), ...]`` (epochs cerrados) ausentes."""
        if granularity <= 0:
            raise ValueError("granularity must be > 0")
        if end < start:
            raise ValueError("end must be >= start")
        start = (start // granularity) * granularity
        end = (end // granularity) * granularity
        with self._lock:
            rows = self._conn.execute(
                """
                WITH expected AS (
                    SELECT generate_series AS epoch
                    FROM generate_series(?, ?, ?)
                ),
                actual AS (
                    SELECT epoch FROM candles
                    WHERE symbol = ? AND granularity = ?
                      AND epoch BETWEEN ? AND ?
                ),
                missing AS (
                    SELECT epoch FROM expected
                    EXCEPT SELECT epoch FROM actual
                ),
                grouped AS (
                    SELECT epoch,
                        epoch - row_number() OVER (ORDER BY epoch) * ?
                            AS island
                    FROM missing
                )
                SELECT min(epoch), max(epoch)
                FROM grouped GROUP BY island ORDER BY 1
                """,
                [start, end, granularity, symbol, granularity, start, end, granularity],
            ).fetchall()
        return [(int(a), int(b)) for a, b in rows]

    def detect_tick_gaps(
        self,
        symbol: str,
        *,
        max_gap_seconds: int,
        start: int | None = None,
        end: int | None = None,
    ) -> list[tuple[int, int]]:
        """Detecta huecos en ticks consecutivos mayores a ``max_gap_seconds``.

        Útil para feeds con cadencia conocida (índices sintéticos = 1 tick/s).
        """
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be > 0")
        clauses = ["symbol = ?"]
        params: list[Any] = [symbol]
        if start is not None:
            clauses.append("epoch >= ?")
            params.append(start)
        if end is not None:
            clauses.append("epoch <= ?")
            params.append(end)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(
                f"""
                WITH ordered AS (
                    SELECT epoch,
                        lag(epoch) OVER (ORDER BY epoch) AS prev_epoch
                    FROM ticks
                    WHERE {where}
                )
                SELECT prev_epoch + 1 AS gap_start, epoch - 1 AS gap_end
                FROM ordered
                WHERE prev_epoch IS NOT NULL
                  AND epoch - prev_epoch > ?
                ORDER BY gap_start
                """,
                params + [max_gap_seconds],
            ).fetchall()
        return [(int(a), int(b)) for a, b in rows]

    # ------------------------------------------------------------------
    # Run tracking (resumable checkpoints)
    # ------------------------------------------------------------------

    def start_run(
        self,
        *,
        symbol: str,
        kind: str,
        granularity: int | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
    ) -> int:
        if kind not in ("ticks", "candles"):
            raise ValueError("kind must be 'ticks' or 'candles'")
        with self._lock:
            run_id = int(
                self._conn.execute(
                    "SELECT nextval('ingest_runs_seq')"
                ).fetchone()[0]
            )
            self._conn.execute(
                """
                INSERT INTO ingest_runs
                    (run_id, symbol, kind, granularity, range_start, range_end)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [run_id, symbol, kind, granularity, range_start, range_end],
            )
        return run_id

    def update_run(
        self,
        run_id: int,
        *,
        rows_received_delta: int = 0,
        rows_inserted_delta: int = 0,
        batches_delta: int = 0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE ingest_runs SET
                    rows_received = rows_received + ?,
                    rows_inserted = rows_inserted + ?,
                    batches       = batches + ?
                WHERE run_id = ?
                """,
                [rows_received_delta, rows_inserted_delta, batches_delta, run_id],
            )

    def finish_run(
        self,
        run_id: int,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        if status not in ("ok", "partial", "failed", "cancelled"):
            raise ValueError("invalid status")
        with self._lock:
            self._conn.execute(
                """
                UPDATE ingest_runs
                SET finished_at = current_timestamp,
                    status = ?,
                    error = ?
                WHERE run_id = ?
                """,
                [status, error, run_id],
            )

    def last_successful_run(
        self,
        symbol: str,
        *,
        kind: str,
        granularity: int | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["symbol = ?", "kind = ?", "status = 'ok'"]
        params: list[Any] = [symbol, kind]
        if granularity is None:
            clauses.append("granularity IS NULL")
        else:
            clauses.append("granularity = ?")
            params.append(granularity)
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT run_id, range_start, range_end, rows_inserted, finished_at
                FROM ingest_runs
                WHERE {' AND '.join(clauses)}
                ORDER BY finished_at DESC NULLS LAST, run_id DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": int(row[0]),
            "range_start": int(row[1]) if row[1] is not None else None,
            "range_end": int(row[2]) if row[2] is not None else None,
            "rows_inserted": int(row[3]) if row[3] is not None else 0,
            "finished_at": row[4],
        }

    # ------------------------------------------------------------------
    # Archive (Parquet export)
    # ------------------------------------------------------------------

    def export_parquet(
        self,
        out_dir: str | Path,
        *,
        symbol: str | None = None,
        compression: str = "zstd",
    ) -> list[Path]:
        """Vuelca ``ticks`` y ``candles`` a Parquet particionado.

        Estructura de salida::

            out_dir/ticks/symbol=<S>/data.parquet
            out_dir/candles/symbol=<S>/granularity=<G>/data.parquet
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        symbols_filter = f"WHERE symbol = '{_escape(symbol)}'" if symbol else ""

        with self._lock:
            tick_symbols = [
                r[0]
                for r in self._conn.execute(
                    f"SELECT DISTINCT symbol FROM ticks {symbols_filter} ORDER BY 1"
                ).fetchall()
            ]
            for sym in tick_symbols:
                dest = out / "ticks" / f"symbol={sym}"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / "data.parquet"
                self._conn.execute(
                    f"""
                    COPY (
                        SELECT epoch, quote, bid, ask, pip_size, tick_id, received_at
                        FROM ticks WHERE symbol = ?
                        ORDER BY epoch
                    ) TO '{path.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION '{compression}')
                    """,
                    [sym],
                )
                written.append(path)

            candle_rows = self._conn.execute(
                f"""
                SELECT DISTINCT symbol, granularity FROM candles
                {symbols_filter} ORDER BY 1, 2
                """
            ).fetchall()
            for sym, gran in candle_rows:
                dest = out / "candles" / f"symbol={sym}" / f"granularity={int(gran)}"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / "data.parquet"
                self._conn.execute(
                    f"""
                    COPY (
                        SELECT epoch, open, high, low, close, received_at
                        FROM candles
                        WHERE symbol = ? AND granularity = ?
                        ORDER BY epoch
                    ) TO '{path.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION '{compression}')
                    """,
                    [sym, int(gran)],
                )
                written.append(path)
        return written


def _escape(value: str) -> str:
    return value.replace("'", "''")
