"""Tests para storage DuckDB + MarketDataIngester (sin red, sin pytest-asyncio)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from src.connectors.deriv import (
    AdaptiveRateLimiter,
    CandleRow,
    DerivAuthError,
    DerivWebSocketClient,
    DuckDBStore,
    IngestStats,
    MarketDataIngester,
    TickRow,
)
from src.connectors.deriv.exceptions import DerivAPIError


# ---------------------------------------------------------------------------
# Fake WebSocket idéntico al usado en tests del cliente
# ---------------------------------------------------------------------------


Responder = Callable[[dict[str, Any]], list[dict[str, Any]]]


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False
        self.responder: Responder | None = None

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        msg = json.loads(data)
        self.sent.append(msg)
        if self.responder is not None:
            for response in self.responder(msg):
                await self._incoming.put(json.dumps(response))

    async def push(self, message: dict[str, Any]) -> None:
        await self._incoming.put(json.dumps(message))

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await self._incoming.put(None)


def make_client(
    responder: Responder | None = None,
) -> tuple[DerivWebSocketClient, FakeWebSocket]:
    fake = FakeWebSocket()
    fake.responder = responder

    async def factory(_url: str) -> FakeWebSocket:
        return fake

    client = DerivWebSocketClient(
        "wss://test.local/public",
        ws_factory=factory,
        request_timeout=2.0,
        ping_interval=None,
        rate_limit_per_second=1000.0,
    )
    return client, fake


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ===========================================================================
# DuckDB store
# ===========================================================================


def test_store_schema_initializes_in_memory() -> None:
    store = DuckDBStore(":memory:")
    try:
        with store._lock:
            tables = {
                row[0]
                for row in store._conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema()"
                ).fetchall()
            }
        assert {"ticks", "candles", "ingest_runs", "schema_version"} <= tables
    finally:
        store.close()


def test_ticks_upsert_is_idempotent() -> None:
    store = DuckDBStore(":memory:")
    try:
        rows = [
            TickRow("R_100", 100, 1.0),
            TickRow("R_100", 101, 1.1),
            TickRow("R_100", 102, 1.2),
        ]
        assert store.upsert_ticks(rows) == 3
        # Reinsertar exactamente lo mismo: no debe duplicar nada.
        assert store.upsert_ticks(rows) == 0
        # Una fila nueva: +1.
        assert store.upsert_ticks([TickRow("R_100", 103, 1.3)]) == 1
        assert store.count_ticks("R_100") == 4
        assert store.latest_tick_epoch("R_100") == 103
        assert store.earliest_tick_epoch("R_100") == 100
    finally:
        store.close()


def test_candles_upsert_updates_ohlc_on_conflict() -> None:
    store = DuckDBStore(":memory:")
    try:
        store.upsert_candles([CandleRow("R_100", 60, 60, 1.0, 2.0, 0.5, 1.5)])
        # Misma vela con close actualizado (vela todavía abierta).
        store.upsert_candles([CandleRow("R_100", 60, 60, 1.0, 2.5, 0.5, 2.4)])
        assert store.count_candles("R_100", 60) == 1
        with store._lock:
            row = store._conn.execute(
                "SELECT high, close FROM candles WHERE symbol='R_100' AND epoch=60"
            ).fetchone()
        assert row == (2.5, 2.4)
    finally:
        store.close()


def test_detect_candle_gaps_returns_inclusive_islands() -> None:
    store = DuckDBStore(":memory:")
    try:
        # Tenemos 60, 120, 360, 420 → faltan 180, 240, 300 (isla) y nada después.
        store.upsert_candles(
            [
                CandleRow("R_100", 60, 60, 1, 1, 1, 1),
                CandleRow("R_100", 60, 120, 1, 1, 1, 1),
                CandleRow("R_100", 60, 360, 1, 1, 1, 1),
                CandleRow("R_100", 60, 420, 1, 1, 1, 1),
            ]
        )
        gaps = store.detect_candle_gaps("R_100", 60, 60, 420)
        assert gaps == [(180, 300)]
    finally:
        store.close()


def test_detect_tick_gaps_finds_long_pauses() -> None:
    store = DuckDBStore(":memory:")
    try:
        store.upsert_ticks(
            [
                TickRow("R_100", 100, 1.0),
                TickRow("R_100", 101, 1.0),
                TickRow("R_100", 105, 1.0),  # huge gap before
                TickRow("R_100", 106, 1.0),
                TickRow("R_100", 200, 1.0),  # another gap
            ]
        )
        gaps = store.detect_tick_gaps("R_100", max_gap_seconds=2)
        assert gaps == [(102, 104), (107, 199)]
    finally:
        store.close()


def test_run_tracking_and_resume_lookup() -> None:
    store = DuckDBStore(":memory:")
    try:
        run_id = store.start_run(
            symbol="R_100",
            kind="ticks",
            range_start=100,
            range_end=200,
        )
        store.update_run(
            run_id, rows_received_delta=50, rows_inserted_delta=48, batches_delta=2
        )
        store.finish_run(run_id, status="ok")
        last = store.last_successful_run("R_100", kind="ticks")
        assert last is not None
        assert last["range_start"] == 100
        assert last["rows_inserted"] == 48
    finally:
        store.close()


def test_export_parquet_writes_partitioned_files(tmp_path: Path) -> None:
    store = DuckDBStore(":memory:")
    try:
        store.upsert_ticks([TickRow("R_100", 100, 1.0), TickRow("R_50", 100, 2.0)])
        store.upsert_candles(
            [CandleRow("R_100", 60, 60, 1, 2, 0.5, 1.5)]
        )
        written = store.export_parquet(tmp_path)
        assert any(p.name == "data.parquet" for p in written)
        assert (tmp_path / "ticks" / "symbol=R_100" / "data.parquet").exists()
        assert (tmp_path / "ticks" / "symbol=R_50" / "data.parquet").exists()
        assert (
            tmp_path / "candles" / "symbol=R_100" / "granularity=60" / "data.parquet"
        ).exists()
    finally:
        store.close()


def test_store_persists_across_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "deriv.duckdb"
    store = DuckDBStore(db_path)
    store.upsert_ticks([TickRow("R_100", 100, 1.0)])
    store.close()
    reopened = DuckDBStore(db_path, read_only=True)
    try:
        assert reopened.count_ticks("R_100") == 1
    finally:
        reopened.close()


# ===========================================================================
# Adaptive rate limiter (pure unit tests)
# ===========================================================================


def test_adaptive_limiter_halves_on_rate_limit() -> None:
    lim = AdaptiveRateLimiter(initial=100, floor=5, ceiling=100)
    assert lim.rate == 100
    lim.on_rate_limited()
    assert lim.rate == 50
    lim.on_rate_limited()
    assert lim.rate == 25
    for _ in range(20):
        lim.on_rate_limited()
    assert lim.rate == lim.floor


def test_adaptive_limiter_increases_after_success_streak() -> None:
    lim = AdaptiveRateLimiter(
        initial=50, floor=5, ceiling=100, increase_step=5, increase_every=10
    )
    for _ in range(9):
        assert not lim.on_success()
    assert lim.on_success()  # 10th success → bump
    assert lim.rate == 55


# ===========================================================================
# Ingester
# ===========================================================================


def _make_history_responder(times: list[int], prices: list[float]) -> Responder:
    """Sirve un único bloque ticks_history y luego vacío."""
    served = {"done": False}

    def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
        if "ticks_history" not in msg:
            return []
        if served["done"]:
            return [
                {
                    "msg_type": "history",
                    "history": {"times": [], "prices": []},
                    "req_id": msg["req_id"],
                }
            ]
        served["done"] = True
        return [
            {
                "msg_type": "history",
                "history": {"times": times, "prices": prices},
                "req_id": msg["req_id"],
            }
        ]

    return responder


def test_backfill_ticks_inserts_and_advances_cursor() -> None:
    async def _run() -> None:
        client, fake = make_client(
            _make_history_responder([100, 101, 102], [1.0, 1.1, 1.2])
        )
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                stats = await ingester.backfill_ticks(
                    "R_100", start=100, end=200, batch_size=10
                )
            assert stats.rows_received == 3
            assert stats.rows_inserted == 3
            assert store.count_ticks("R_100") == 3
            sent = [m for m in fake.sent if "ticks_history" in m]
            assert sent[0]["start"] == 100
            assert sent[0]["count"] == 10
        finally:
            store.close()

    run(_run())


def test_backfill_ticks_resumes_from_db(tmp_path: Path) -> None:
    async def _run() -> None:
        store = DuckDBStore(tmp_path / "deriv.duckdb")
        try:
            store.upsert_ticks(
                [TickRow("R_100", t, float(t)) for t in (100, 101, 102)]
            )
            client, fake = make_client(
                _make_history_responder([103, 104], [1.3, 1.4])
            )
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                await ingester.backfill_ticks(
                    "R_100", start=100, end=200, batch_size=10
                )
            # El cursor inicial debe ser max(epoch)+1 = 103 (no 100).
            assert any(
                "ticks_history" in m and m["start"] == 103 for m in fake.sent
            )
            assert store.count_ticks("R_100") == 5
        finally:
            store.close()

    run(_run())


def test_backfill_ticks_terminates_on_stalled_cursor() -> None:
    """Si el servidor devuelve siempre el mismo epoch, debemos parar."""

    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks_history" not in msg:
                return []
            return [
                {
                    "msg_type": "history",
                    "history": {"times": [100], "prices": [1.0]},
                    "req_id": msg["req_id"],
                }
            ]

        client, _ = make_client(responder)
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                stats = await asyncio.wait_for(
                    ingester.backfill_ticks(
                        "R_100", start=100, end=10_000, batch_size=10
                    ),
                    timeout=3.0,
                )
            assert stats.rows_inserted == 1
            with store._lock:
                row = store._conn.execute(
                    "SELECT status, error FROM ingest_runs ORDER BY run_id DESC LIMIT 1"
                ).fetchone()
            assert row[0] == "partial"
            assert row[1] == "cursor stalled"
        finally:
            store.close()

    run(_run())


def test_backfill_candles_persists_ohlc() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks_history" not in msg:
                return []
            if msg.get("style") != "candles":
                return []
            req_id = msg["req_id"]
            if msg["start"] > 240:
                return [
                    {"msg_type": "candles", "candles": [], "req_id": req_id}
                ]
            return [
                {
                    "msg_type": "candles",
                    "candles": [
                        {"epoch": 60, "open": 1, "high": 2, "low": 0.5, "close": 1.5},
                        {"epoch": 120, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0},
                        {"epoch": 180, "open": 2.0, "high": 3.0, "low": 1.5, "close": 2.5},
                        {"epoch": 240, "open": 2.5, "high": 3.5, "low": 2.0, "close": 3.0},
                    ],
                    "req_id": req_id,
                }
            ]

        client, _ = make_client(responder)
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                stats = await ingester.backfill_candles(
                    "R_100", 60, start=60, end=240, batch_size=4
                )
            assert stats.rows_inserted == 4
            assert store.count_candles("R_100", 60) == 4
            assert store.latest_candle_epoch("R_100", 60) == 240
        finally:
            store.close()

    run(_run())


def test_repair_candle_gaps_fills_islands() -> None:
    async def _run() -> None:
        # Pre-cargamos huecos: tenemos 60 y 240, faltan 120 y 180.
        store = DuckDBStore(":memory:")
        try:
            store.upsert_candles(
                [
                    CandleRow("R_100", 60, 60, 1, 1, 1, 1),
                    CandleRow("R_100", 60, 240, 1, 1, 1, 1),
                ]
            )

            def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
                if "ticks_history" not in msg:
                    return []
                # Devolvemos exactamente los candles del rango pedido.
                start = msg["start"]
                end = msg["end"]
                gran = msg["granularity"]
                candles = []
                e = start
                while e <= end:
                    candles.append(
                        {"epoch": e, "open": 1, "high": 1, "low": 1, "close": 1}
                    )
                    e += gran
                return [
                    {
                        "msg_type": "candles",
                        "candles": candles,
                        "req_id": msg["req_id"],
                    }
                ]

            client, _ = make_client(responder)
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                stats = await ingester.repair_candle_gaps(
                    "R_100", 60, start=60, end=240
                )
            assert stats.rows_inserted == 2  # 120 y 180
            assert store.count_candles("R_100", 60) == 4
        finally:
            store.close()

    run(_run())


def test_subscribe_ticks_persists_until_cancelled() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks" in msg and "ticks_history" not in msg:
                req_id = msg["req_id"]
                return [
                    {
                        "msg_type": "tick",
                        "tick": {
                            "epoch": 100,
                            "quote": 1.0,
                            "symbol": "R_100",
                            "id": "t1",
                        },
                        "subscription": {"id": "sub-1"},
                        "req_id": req_id,
                    },
                    {
                        "msg_type": "tick",
                        "tick": {
                            "epoch": 101,
                            "quote": 1.1,
                            "symbol": "R_100",
                            "id": "t2",
                        },
                        "subscription": {"id": "sub-1"},
                    },
                ]
            if "forget" in msg:
                return [
                    {"msg_type": "forget", "forget": 1, "req_id": msg["req_id"]}
                ]
            return []

        client, _ = make_client(responder)
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                task = asyncio.create_task(ingester.subscribe_ticks("R_100"))
                # Esperamos a que se materialicen las 2 filas.
                for _ in range(50):
                    if store.count_ticks("R_100") >= 2:
                        break
                    await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert store.count_ticks("R_100") == 2
            with store._lock:
                status = store._conn.execute(
                    "SELECT status FROM ingest_runs ORDER BY run_id DESC LIMIT 1"
                ).fetchone()[0]
            assert status == "cancelled"
        finally:
            store.close()

    run(_run())


def test_adaptive_limiter_triggers_on_rate_limit_error() -> None:
    async def _run() -> None:
        call_count = {"n": 0}

        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks_history" not in msg:
                return []
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [
                    {
                        "msg_type": "history",
                        "error": {
                            "code": "RateLimit",
                            "message": "slow down",
                        },
                        "req_id": msg["req_id"],
                    }
                ]
            return [
                {
                    "msg_type": "history",
                    "history": {"times": [100], "prices": [1.0]},
                    "req_id": msg["req_id"],
                }
            ]

        client, _ = make_client(responder)
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(
                    client=client,
                    store=store,
                    backoff_base=0.001,
                    backoff_max=0.01,
                )
                initial_rate = ingester._limiter.rate
                # Una fila + parada (epoch=100 y end=100 = stall)
                await ingester.backfill_ticks(
                    "R_100", start=100, end=100, batch_size=5
                )
            assert ingester._limiter.rate < initial_rate  # se redujo
            assert call_count["n"] == 2  # un reintento exitoso
        finally:
            store.close()

    run(_run())


def test_auth_errors_short_circuit_retries() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks_history" not in msg:
                return []
            return [
                {
                    "msg_type": "history",
                    "error": {
                        "code": "AuthorizationRequired",
                        "message": "please log in",
                    },
                    "req_id": msg["req_id"],
                }
            ]

        client, _ = make_client(responder)
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                with pytest.raises(DerivAuthError):
                    await ingester.backfill_ticks(
                        "R_100", start=100, end=200, batch_size=10
                    )
        finally:
            store.close()

    run(_run())


def test_backfill_many_works_for_multiple_assets() -> None:
    async def _run() -> None:
        served: dict[str, bool] = {}

        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks_history" not in msg:
                return []
            symbol = msg["ticks_history"]
            if served.get(symbol):
                return [
                    {
                        "msg_type": "history",
                        "history": {"times": [], "prices": []},
                        "req_id": msg["req_id"],
                    }
                ]
            served[symbol] = True
            return [
                {
                    "msg_type": "history",
                    "history": {
                        "times": [100, 101],
                        "prices": [1.0, 1.1],
                    },
                    "req_id": msg["req_id"],
                }
            ]

        client, _ = make_client(responder)
        store = DuckDBStore(":memory:")
        try:
            async with client:
                ingester = MarketDataIngester(client=client, store=store)
                plans = [
                    ("R_100", "ticks", None, 100, 200),
                    ("R_50", "ticks", None, 100, 200),
                    ("R_25", "ticks", None, 100, 200),
                ]
                results = await ingester.backfill_many(plans, batch_size=10)
            assert len(results) == 3
            for key, stats in results.items():
                assert isinstance(stats, IngestStats)
                assert stats.rows_inserted == 2
            assert store.count_ticks("R_100") == 2
            assert store.count_ticks("R_50") == 2
            assert store.count_ticks("R_25") == 2
        finally:
            store.close()

    run(_run())
