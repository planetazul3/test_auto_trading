"""Ingester resiliente para Deriv: backfill histórico + streaming + reparación.

Reúne tres capacidades en un solo objeto:

1. **Backfill** paginado de ``ticks_history`` (ticks o velas) con
   cursor reanudable, anti-loop, y *checkpoints* en ``ingest_runs``.
2. **Streaming** en vivo vía ``ticks_stream`` y
   ``ticks_history_stream(style='candles')`` con persistencia idempotente.
3. **Reparación de huecos**: detecta intervalos faltantes en candles
   (SQL window functions) y los rellena en lotes.

Resiliencia transversal:

* Reintentos con back-off exponencial + jitter.
* Adaptive rate limiting (AIMD): la tasa baja a la mitad ante
  ``RateLimit`` y sube aditivamente tras N éxitos.
* ``asyncio.Semaphore`` para acotar peticiones concurrentes.
* ``asyncio.to_thread`` para todas las escrituras DuckDB.
* Cancelación cooperativa: si el caller cancela, los runs se marcan
  ``cancelled`` y se persiste lo ya descargado.

Funciona con cualquier ``underlying_symbol`` (no hay valores hardcoded).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .client import CANDLE_GRANULARITIES, DerivWebSocketClient
from .exceptions import (
    DerivAPIError,
    DerivAuthError,
    DerivConnectionError,
    DerivRateLimitError,
    DerivSubscriptionError,
)
from .storage import CandleRow, DuckDBStore, TickRow

logger = logging.getLogger(__name__)


DEFAULT_BATCH = 5000
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_MAX = 60.0
DEFAULT_MAX_RETRIES = 8


@dataclass(slots=True)
class IngestStats:
    rows_received: int = 0
    rows_inserted: int = 0
    batches: int = 0
    retries: int = 0


@dataclass(slots=True)
class AdaptiveRateLimiter:
    """AIMD: multiplicative decrease on rate-limit, additive increase on success.

    Modifica directamente el ``rate`` del ``AsyncTokenBucket`` del cliente,
    de modo que el throttling se aplica en el siguiente ``acquire``.
    """

    initial: float = 100.0
    floor: float = 5.0
    ceiling: float = 100.0
    increase_step: float = 5.0
    increase_every: int = 50
    rate: float = field(init=False)
    _success_streak: int = 0

    def __post_init__(self) -> None:
        if not (self.floor <= self.initial <= self.ceiling):
            raise ValueError("require floor <= initial <= ceiling")
        if self.increase_step <= 0 or self.increase_every <= 0:
            raise ValueError("increase_step and increase_every must be > 0")
        self.rate = float(self.initial)

    def on_success(self) -> bool:
        self._success_streak += 1
        if (
            self._success_streak >= self.increase_every
            and self.rate < self.ceiling
        ):
            self.rate = min(self.ceiling, self.rate + self.increase_step)
            self._success_streak = 0
            return True
        return False

    def on_rate_limited(self) -> None:
        self._success_streak = 0
        self.rate = max(self.floor, self.rate / 2.0)


class MarketDataIngester:
    """Orquesta backfill + streaming + reparación sobre un símbolo cualquiera.

    Parameters
    ----------
    client:
        Conector ya configurado (``DerivWebSocketClient``). Para
        backfills históricos el endpoint ``public`` es suficiente; el
        autenticado solo es necesario para datos de cuenta.
    store:
        Instancia ``DuckDBStore`` abierta en modo lectura/escritura.
    max_concurrent_requests:
        Tope de peticiones ``ticks_history`` paralelas (con un mismo
        WebSocket Deriv permite ~5 conexiones simultáneas y 100 req/s).
    backoff_base / backoff_max / max_retries:
        Parámetros del back-off exponencial con jitter.
    adaptive_limiter:
        Si se provee, se sincroniza con el token bucket del cliente.
        Por defecto se construye uno con el rate actual del cliente.
    """

    def __init__(
        self,
        *,
        client: DerivWebSocketClient,
        store: DuckDBStore,
        max_concurrent_requests: int = 4,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        max_retries: int = DEFAULT_MAX_RETRIES,
        adaptive_limiter: AdaptiveRateLimiter | None = None,
    ) -> None:
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.client = client
        self.store = store
        self._sem = asyncio.Semaphore(max_concurrent_requests)
        self._backoff_base = float(backoff_base)
        self._backoff_max = float(backoff_max)
        self._max_retries = int(max_retries)
        current_rate = getattr(client, "_rate_limiter").rate
        self._limiter = adaptive_limiter or AdaptiveRateLimiter(
            initial=current_rate, ceiling=current_rate
        )

    # ------------------------------------------------------------------
    # Streaming en vivo
    # ------------------------------------------------------------------

    async def subscribe_ticks(self, symbol: str) -> IngestStats:
        """Persiste ticks en vivo hasta que el caller cancele."""
        if not symbol:
            raise ValueError("symbol is required")
        stats = IngestStats()
        run_id = self.store.start_run(symbol=symbol, kind="ticks")
        try:
            async for envelope in self.client.ticks_stream(symbol):
                row = _tick_from_payload(symbol, envelope.get("tick") or {})
                if row is None:
                    continue
                stats.rows_received += 1
                inserted = await asyncio.to_thread(
                    self.store.upsert_ticks, [row]
                )
                stats.rows_inserted += inserted
                stats.batches += 1
                self.store.update_run(
                    run_id,
                    rows_received_delta=1,
                    rows_inserted_delta=inserted,
                    batches_delta=1,
                )
            self.store.finish_run(run_id, status="ok")
        except asyncio.CancelledError:
            self.store.finish_run(run_id, status="cancelled")
            raise
        except Exception as exc:
            self.store.finish_run(run_id, status="failed", error=repr(exc))
            raise
        return stats

    async def subscribe_candles(self, symbol: str, granularity: int) -> IngestStats:
        """Persiste actualizaciones OHLC en vivo."""
        if not symbol:
            raise ValueError("symbol is required")
        if granularity not in CANDLE_GRANULARITIES:
            raise ValueError(f"granularity must be one of {sorted(CANDLE_GRANULARITIES)}")
        stats = IngestStats()
        run_id = self.store.start_run(
            symbol=symbol, kind="candles", granularity=granularity
        )
        stream = self.client.ticks_history_stream(
            symbol,
            end="latest",
            count=1,
            style="candles",
            granularity=granularity,
        )
        try:
            async for envelope in stream:
                rows = _candles_from_payload(symbol, granularity, envelope)
                if not rows:
                    continue
                stats.rows_received += len(rows)
                inserted = await asyncio.to_thread(
                    self.store.upsert_candles, rows
                )
                stats.rows_inserted += inserted
                stats.batches += 1
                self.store.update_run(
                    run_id,
                    rows_received_delta=len(rows),
                    rows_inserted_delta=inserted,
                    batches_delta=1,
                )
            self.store.finish_run(run_id, status="ok")
        except asyncio.CancelledError:
            self.store.finish_run(run_id, status="cancelled")
            raise
        except Exception as exc:
            self.store.finish_run(run_id, status="failed", error=repr(exc))
            raise
        return stats

    # ------------------------------------------------------------------
    # Backfill paginado
    # ------------------------------------------------------------------

    async def backfill_ticks(
        self,
        symbol: str,
        *,
        start: int,
        end: int | None = None,
        batch_size: int = DEFAULT_BATCH,
        resume_from_db: bool = True,
    ) -> IngestStats:
        """Descarga ticks históricos cronológicamente desde ``start``.

        Si ``resume_from_db`` y ya hay datos del símbolo, ``start`` se
        adelanta automáticamente a ``max(epoch) + 1`` (idempotente).
        """
        if not symbol:
            raise ValueError("symbol is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        end_epoch = int(end) if end is not None else int(time.time())
        cursor = int(start)
        if resume_from_db:
            latest = self.store.latest_tick_epoch(symbol)
            if latest is not None and latest >= cursor:
                cursor = latest + 1
        stats = IngestStats()
        run_id = self.store.start_run(
            symbol=symbol,
            kind="ticks",
            range_start=cursor,
            range_end=end_epoch,
        )
        status = "ok"
        error: str | None = None
        try:
            while cursor <= end_epoch:
                payload = await self._call_with_retry(
                    self.client.ticks_history,
                    symbol,
                    start=cursor,
                    end=end_epoch,
                    count=batch_size,
                    style="ticks",
                    _stats=stats,
                )
                rows = _ticks_from_history(symbol, payload)
                if not rows:
                    break
                stats.rows_received += len(rows)
                inserted = await asyncio.to_thread(self.store.upsert_ticks, rows)
                stats.rows_inserted += inserted
                stats.batches += 1
                self.store.update_run(
                    run_id,
                    rows_received_delta=len(rows),
                    rows_inserted_delta=inserted,
                    batches_delta=1,
                )
                last_epoch = rows[-1].epoch
                if last_epoch < cursor:
                    # Servidor devolvió un bloque anterior al cursor:
                    # señal de glitch — abortar para evitar bucle infinito.
                    status = "partial"
                    error = "non-monotonic batch"
                    break
                if last_epoch == cursor and len(rows) <= 1:
                    # No avanzamos: protección anti-stuck.
                    status = "partial"
                    error = "cursor stalled"
                    break
                cursor = last_epoch + 1
                if len(rows) < batch_size:
                    break
        except asyncio.CancelledError:
            self.store.finish_run(run_id, status="cancelled")
            raise
        except Exception as exc:
            self.store.finish_run(run_id, status="failed", error=repr(exc))
            raise
        self.store.finish_run(run_id, status=status, error=error)
        return stats

    async def backfill_candles(
        self,
        symbol: str,
        granularity: int,
        *,
        start: int,
        end: int | None = None,
        batch_size: int = DEFAULT_BATCH,
        resume_from_db: bool = True,
    ) -> IngestStats:
        if not symbol:
            raise ValueError("symbol is required")
        if granularity not in CANDLE_GRANULARITIES:
            raise ValueError(f"granularity must be one of {sorted(CANDLE_GRANULARITIES)}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        end_epoch = int(end) if end is not None else int(time.time())
        cursor = int(start)
        if resume_from_db:
            latest = self.store.latest_candle_epoch(symbol, granularity)
            if latest is not None and latest >= cursor:
                cursor = latest + granularity
        stats = IngestStats()
        run_id = self.store.start_run(
            symbol=symbol,
            kind="candles",
            granularity=granularity,
            range_start=cursor,
            range_end=end_epoch,
        )
        status = "ok"
        error: str | None = None
        try:
            while cursor <= end_epoch:
                payload = await self._call_with_retry(
                    self.client.ticks_history,
                    symbol,
                    start=cursor,
                    end=end_epoch,
                    count=batch_size,
                    style="candles",
                    granularity=granularity,
                    _stats=stats,
                )
                rows = _candles_from_history(symbol, granularity, payload)
                if not rows:
                    break
                stats.rows_received += len(rows)
                inserted = await asyncio.to_thread(self.store.upsert_candles, rows)
                stats.rows_inserted += inserted
                stats.batches += 1
                self.store.update_run(
                    run_id,
                    rows_received_delta=len(rows),
                    rows_inserted_delta=inserted,
                    batches_delta=1,
                )
                last_epoch = rows[-1].epoch
                if last_epoch < cursor:
                    status = "partial"
                    error = "non-monotonic batch"
                    break
                if last_epoch == cursor and len(rows) <= 1:
                    status = "partial"
                    error = "cursor stalled"
                    break
                cursor = last_epoch + granularity
                if len(rows) < batch_size:
                    break
        except asyncio.CancelledError:
            self.store.finish_run(run_id, status="cancelled")
            raise
        except Exception as exc:
            self.store.finish_run(run_id, status="failed", error=repr(exc))
            raise
        self.store.finish_run(run_id, status=status, error=error)
        return stats

    # ------------------------------------------------------------------
    # Reparación de huecos
    # ------------------------------------------------------------------

    async def repair_candle_gaps(
        self,
        symbol: str,
        granularity: int,
        *,
        start: int,
        end: int,
        batch_size: int = DEFAULT_BATCH,
    ) -> IngestStats:
        """Detecta y rellena candles ausentes en ``[start, end]``."""
        gaps = self.store.detect_candle_gaps(symbol, granularity, start, end)
        total = IngestStats()
        for gap_start, gap_end in gaps:
            sub = await self.backfill_candles(
                symbol,
                granularity,
                start=gap_start,
                end=gap_end,
                batch_size=batch_size,
                resume_from_db=False,
            )
            total.rows_received += sub.rows_received
            total.rows_inserted += sub.rows_inserted
            total.batches += sub.batches
            total.retries += sub.retries
        return total

    async def repair_tick_gaps(
        self,
        symbol: str,
        *,
        max_gap_seconds: int,
        start: int | None = None,
        end: int | None = None,
        batch_size: int = DEFAULT_BATCH,
    ) -> IngestStats:
        gaps = self.store.detect_tick_gaps(
            symbol, max_gap_seconds=max_gap_seconds, start=start, end=end
        )
        total = IngestStats()
        for gap_start, gap_end in gaps:
            sub = await self.backfill_ticks(
                symbol,
                start=gap_start,
                end=gap_end,
                batch_size=batch_size,
                resume_from_db=False,
            )
            total.rows_received += sub.rows_received
            total.rows_inserted += sub.rows_inserted
            total.batches += sub.batches
            total.retries += sub.retries
        return total

    # ------------------------------------------------------------------
    # Multi-asset helpers
    # ------------------------------------------------------------------

    async def backfill_many(
        self,
        plans: Iterable[
            tuple[str, str, int | None, int, int | None]
            # (symbol, kind, granularity, start, end)
        ],
        *,
        batch_size: int = DEFAULT_BATCH,
    ) -> dict[tuple[str, str, int | None], IngestStats]:
        """Lanza backfills concurrentes acotados por ``max_concurrent_requests``."""

        async def _one(
            symbol: str, kind: str, granularity: int | None, start: int, end: int | None
        ) -> tuple[tuple[str, str, int | None], IngestStats]:
            if kind == "ticks":
                stats = await self.backfill_ticks(
                    symbol, start=start, end=end, batch_size=batch_size
                )
            elif kind == "candles":
                if granularity is None:
                    raise ValueError("granularity required for candles")
                stats = await self.backfill_candles(
                    symbol,
                    granularity,
                    start=start,
                    end=end,
                    batch_size=batch_size,
                )
            else:
                raise ValueError(f"unknown kind: {kind}")
            return (symbol, kind, granularity), stats

        tasks = [
            asyncio.create_task(_one(s, k, g, st, en))
            for s, k, g, st, en in plans
        ]
        results: dict[tuple[str, str, int | None], IngestStats] = {}
        for fut in asyncio.as_completed(tasks):
            key, stats = await fut
            results[key] = stats
        return results

    # ------------------------------------------------------------------
    # Retry + back-off
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        fn: Any,
        *args: Any,
        _stats: IngestStats | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                async with self._sem:
                    result = await fn(*args, **kwargs)
                if self._limiter.on_success():
                    self._sync_rate_to_client()
                return result
            except DerivAuthError:
                raise  # nunca tiene sentido reintentar
            except DerivRateLimitError as exc:
                self._limiter.on_rate_limited()
                self._sync_rate_to_client()
                attempt += 1
                if _stats is not None:
                    _stats.retries += 1
                if attempt > self._max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt))
                logger.warning("rate-limited, backing off: %s", exc)
            except (DerivConnectionError, DerivAPIError, DerivSubscriptionError) as exc:
                attempt += 1
                if _stats is not None:
                    _stats.retries += 1
                if attempt > self._max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt))
                logger.warning(
                    "transient error (%s), retry %d/%d",
                    exc,
                    attempt,
                    self._max_retries,
                )

    def _backoff(self, attempt: int) -> float:
        base = min(self._backoff_max, self._backoff_base * (2 ** (attempt - 1)))
        # full jitter
        return random.uniform(0.0, base)

    def _sync_rate_to_client(self) -> None:
        bucket = getattr(self.client, "_rate_limiter", None)
        if bucket is not None:
            bucket.rate = self._limiter.rate


# ---------------------------------------------------------------------------
# Payload parsers
# ---------------------------------------------------------------------------


def _tick_from_payload(symbol: str, tick: dict[str, Any]) -> TickRow | None:
    if not isinstance(tick, dict):
        return None
    epoch = tick.get("epoch")
    quote = tick.get("quote")
    if epoch is None or quote is None:
        return None
    try:
        return TickRow(
            symbol=symbol,
            epoch=int(epoch),
            quote=float(quote),
            bid=_maybe_float(tick.get("bid")),
            ask=_maybe_float(tick.get("ask")),
            pip_size=_maybe_float(tick.get("pip_size")),
            tick_id=str(tick["id"]) if tick.get("id") is not None else None,
        )
    except (TypeError, ValueError):
        return None


def _ticks_from_history(symbol: str, payload: dict[str, Any]) -> list[TickRow]:
    history = payload.get("history") or {}
    times = history.get("times") or []
    prices = history.get("prices") or []
    rows: list[TickRow] = []
    for epoch, price in zip(times, prices):
        try:
            rows.append(
                TickRow(symbol=symbol, epoch=int(epoch), quote=float(price))
            )
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r.epoch)
    return rows


def _candle_from_dict(
    symbol: str, granularity: int, candle: dict[str, Any]
) -> CandleRow | None:
    try:
        return CandleRow(
            symbol=symbol,
            granularity=granularity,
            epoch=int(candle["epoch"]),
            open=float(candle["open"]),
            high=float(candle["high"]),
            low=float(candle["low"]),
            close=float(candle["close"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _candles_from_history(
    symbol: str, granularity: int, payload: dict[str, Any]
) -> list[CandleRow]:
    raw = payload.get("candles") or []
    rows: list[CandleRow] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        row = _candle_from_dict(symbol, granularity, entry)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r.epoch)
    return rows


def _candles_from_payload(
    symbol: str, granularity: int, envelope: dict[str, Any]
) -> list[CandleRow]:
    """Convierte tanto respuestas iniciales (lista) como updates OHLC."""
    if envelope.get("msg_type") == "candles" or "candles" in envelope:
        return _candles_from_history(symbol, granularity, envelope)
    ohlc = envelope.get("ohlc")
    if isinstance(ohlc, dict):
        candle = _candle_from_dict(symbol, granularity, ohlc)
        return [candle] if candle is not None else []
    return []


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
