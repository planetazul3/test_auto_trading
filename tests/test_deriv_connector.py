"""Tests para el conector Deriv (auth + WebSocket client + rate limiter).

No requieren conectividad: usan un fake WebSocket programable inyectado
vía ``ws_factory``. Compatible con ``pytest`` sin ``pytest-asyncio``: cada
test asíncrono se ejecuta con ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import pytest

from src.connectors.deriv import (
    CANDLE_GRANULARITIES,
    DerivAPIError,
    DerivAuthError,
    DerivConnectionError,
    DerivOAuth2,
    DerivOTPClient,
    DerivRateLimitError,
    DerivSubscriptionError,
    DerivWebSocketClient,
    PKCEParameters,
)
from src.connectors.deriv.rate_limiter import AsyncTokenBucket


# ---------------------------------------------------------------------------
# Helpers de testing
# ---------------------------------------------------------------------------


Responder = Callable[[dict[str, Any]], list[dict[str, Any]]]


class FakeWebSocket:
    """Fake WebSocket con responder programable."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False
        self.responder: Responder | None = None
        self.send_hook: Callable[[dict[str, Any]], None] | None = None

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        msg = json.loads(data)
        self.sent.append(msg)
        if self.send_hook is not None:
            self.send_hook(msg)
        if self.responder is not None:
            for response in self.responder(msg):
                await self._incoming.put(json.dumps(response))

    async def push(self, message: dict[str, Any]) -> None:
        await self._incoming.put(json.dumps(message))

    async def push_raw(self, raw: str | None) -> None:
        await self._incoming.put(raw)

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
    **client_kwargs: Any,
) -> tuple[DerivWebSocketClient, FakeWebSocket]:
    fake = FakeWebSocket()
    fake.responder = responder

    async def factory(_url: str) -> FakeWebSocket:
        return fake

    kwargs = {
        "request_timeout": 2.0,
        "ping_interval": None,
        "rate_limit_per_second": 1000.0,
    }
    kwargs.update(client_kwargs)
    client = DerivWebSocketClient(
        "wss://test.local/public", ws_factory=factory, **kwargs
    )
    return client, fake


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Auth / PKCE
# ---------------------------------------------------------------------------


def test_pkce_challenge_matches_verifier() -> None:
    pkce = DerivOAuth2.generate_pkce()
    assert isinstance(pkce, PKCEParameters)
    assert 43 <= len(pkce.code_verifier) <= 128
    assert pkce.code_challenge_method == "S256"

    import base64

    expected = base64.urlsafe_b64encode(
        hashlib.sha256(pkce.code_verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    assert pkce.code_challenge == expected


def test_pkce_rejects_bad_length() -> None:
    with pytest.raises(ValueError):
        DerivOAuth2.generate_pkce(length=10)
    with pytest.raises(ValueError):
        DerivOAuth2.generate_pkce(length=200)


def test_authorization_url_contains_all_required_params() -> None:
    oauth = DerivOAuth2(client_id="cid", redirect_uri="https://app/cb")
    url = oauth.build_authorization_url(
        state="abc", code_challenge="chal"
    )
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    assert qs["response_type"] == ["code"]
    assert qs["client_id"] == ["cid"]
    assert qs["redirect_uri"] == ["https://app/cb"]
    assert qs["scope"] == ["trade"]
    assert qs["state"] == ["abc"]
    assert qs["code_challenge"] == ["chal"]
    assert qs["code_challenge_method"] == ["S256"]
    assert "prompt" not in qs


def test_signup_url_adds_registration_and_affiliate_params() -> None:
    oauth = DerivOAuth2(client_id="cid", redirect_uri="https://app/cb")
    url = oauth.build_signup_url(
        state="s",
        code_challenge="c",
        sidc="GUID",
        utm_campaign="camp",
        utm_source="CU303219",
    )
    qs = parse_qs(urlparse(url).query)
    assert qs["prompt"] == ["registration"]
    assert qs["sidc"] == ["GUID"]
    assert qs["utm_medium"] == ["affiliate"]
    assert qs["utm_campaign"] == ["camp"]
    assert qs["utm_source"] == ["CU303219"]


def test_authorization_url_rejects_invalid_scope() -> None:
    oauth = DerivOAuth2(client_id="cid", redirect_uri="https://app/cb")
    with pytest.raises(ValueError):
        oauth.build_authorization_url(
            state="s", code_challenge="c", scope="invalid"
        )


# ---------------------------------------------------------------------------
# OTP REST client (using a fake httpx-style async client)
# ---------------------------------------------------------------------------


class FakeHTTPResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self) -> Any:
        return self._payload


class FakeHTTPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.responses: dict[tuple[str, str], FakeHTTPResponse] = {}

    def expect(self, method: str, url: str, payload: Any, status: int = 200) -> None:
        self.responses[(method.upper(), url)] = FakeHTTPResponse(payload, status)

    async def get(
        self, url: str, *, headers: dict[str, str] | None = None
    ) -> FakeHTTPResponse:
        self.calls.append(("GET", url, {"headers": headers or {}}))
        return self.responses[("GET", url)]

    async def post(
        self,
        url: str,
        *,
        data: Any | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHTTPResponse:
        self.calls.append(
            ("POST", url, {"data": data, "json": json, "headers": headers or {}})
        )
        return self.responses[("POST", url)]


def test_otp_request_returns_ws_url() -> None:
    http = FakeHTTPClient()
    http.expect(
        "POST",
        "https://api.derivws.com/trading/v1/options/accounts/DOT1/otp",
        {
            "data": {
                "url": "wss://api.derivws.com/trading/v1/options/ws/demo?otp=xyz"
            }
        },
    )
    otp_client = DerivOTPClient(
        http_client=http, access_token="t0k", app_id="app42"
    )
    url = run(otp_client.request_otp("DOT1"))
    assert url == "wss://api.derivws.com/trading/v1/options/ws/demo?otp=xyz"
    method, called_url, kwargs = http.calls[0]
    assert method == "POST"
    assert called_url.endswith("/accounts/DOT1/otp")
    assert kwargs["headers"]["Authorization"] == "Bearer t0k"
    assert kwargs["headers"]["Deriv-App-ID"] == "app42"


def test_otp_request_validates_payload() -> None:
    http = FakeHTTPClient()
    http.expect(
        "POST",
        "https://api.derivws.com/trading/v1/options/accounts/DOT1/otp",
        {"data": {}},
    )
    otp_client = DerivOTPClient(
        http_client=http, access_token="t0k", app_id="app42"
    )
    with pytest.raises(ValueError):
        run(otp_client.request_otp("DOT1"))


def test_create_account_validates_account_type() -> None:
    otp_client = DerivOTPClient(
        http_client=FakeHTTPClient(), access_token="t", app_id="a"
    )
    with pytest.raises(ValueError):
        run(otp_client.create_account(account_type="paper"))


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def test_token_bucket_throttles() -> None:
    async def _exercise() -> float:
        bucket = AsyncTokenBucket(rate=20.0, capacity=2.0)
        await bucket.acquire()
        await bucket.acquire()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await bucket.acquire()
        return loop.time() - t0

    waited = run(_exercise())
    assert waited >= 0.04


def test_token_bucket_rejects_oversized_request() -> None:
    async def _exercise() -> None:
        bucket = AsyncTokenBucket(rate=10.0, capacity=2.0)
        with pytest.raises(ValueError):
            await bucket.acquire(5.0)

    run(_exercise())


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def test_demo_constructor_requires_otp_in_url() -> None:
    with pytest.raises(ValueError):
        DerivWebSocketClient.demo("wss://api.derivws.com/trading/v1/options/ws/demo")


def test_demo_constructor_accepts_otp_url() -> None:
    client = DerivWebSocketClient.demo(
        "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc"
    )
    assert "otp=abc" in client.url


def test_public_constructor_uses_default_url() -> None:
    client = DerivWebSocketClient.public()
    assert client.url.endswith("/public")


# ---------------------------------------------------------------------------
# Endpoints: System
# ---------------------------------------------------------------------------


def _echo(msg_type: str, body: dict[str, Any]) -> Responder:
    def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
        return [{**body, "msg_type": msg_type, "req_id": msg["req_id"]}]

    return responder


def test_ping_request_and_response() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("ping", {"ping": "pong"}))
        async with client:
            result = await client.ping()
        assert fake.sent[0]["ping"] == 1
        assert "req_id" in fake.sent[0]
        assert result["ping"] == "pong"
        assert result["msg_type"] == "ping"

    run(_run())


def test_time_endpoint() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("time", {"time": 1_700_000_000}))
        async with client:
            result = await client.time()
        assert fake.sent[0] == {"time": 1, "req_id": fake.sent[0]["req_id"]}
        assert result["time"] == 1_700_000_000

    run(_run())


def test_trading_times_default_date() -> None:
    async def _run() -> None:
        client, fake = make_client(
            _echo("trading_times", {"trading_times": {"markets": []}})
        )
        async with client:
            await client.trading_times()
        assert fake.sent[0]["trading_times"] == "today"

    run(_run())


def test_trading_times_rejects_empty() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.trading_times("")

    run(_run())


# ---------------------------------------------------------------------------
# Endpoints: Market data
# ---------------------------------------------------------------------------


def test_active_symbols_validates_mode() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.active_symbols(active_symbols="medium")

    run(_run())


def test_active_symbols_brief_strips_none_params() -> None:
    async def _run() -> None:
        client, fake = make_client(
            _echo("active_symbols", {"active_symbols": []})
        )
        async with client:
            await client.active_symbols(active_symbols="brief")
        sent = fake.sent[0]
        assert sent["active_symbols"] == "brief"
        assert "contract_type" not in sent  # ``None`` removed

    run(_run())


def test_contracts_for_passes_optional_fields() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("contracts_for", {}))
        async with client:
            await client.contracts_for(
                "1HZ100V", currency="USD", product_type="basic"
            )
        sent = fake.sent[0]
        assert sent["contracts_for"] == "1HZ100V"
        assert sent["currency"] == "USD"
        assert sent["product_type"] == "basic"
        assert "landing_company" not in sent

    run(_run())


def test_contracts_list_simple() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("contracts_list", {"contracts_list": []}))
        async with client:
            await client.contracts_list()
        assert fake.sent[0]["contracts_list"] == 1

    run(_run())


def test_ticks_history_validates_style_and_granularity() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.ticks_history("X", style="lines")
            with pytest.raises(ValueError):
                await client.ticks_history("X", style="candles", granularity=45)

    run(_run())
    assert 60 in CANDLE_GRANULARITIES
    assert 86400 in CANDLE_GRANULARITIES


def test_ticks_history_emits_expected_payload() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("candles", {"candles": []}))
        async with client:
            await client.ticks_history(
                "1HZ100V",
                end="latest",
                start=1,
                count=50,
                style="candles",
                granularity=300,
            )
        sent = fake.sent[0]
        assert sent["ticks_history"] == "1HZ100V"
        assert sent["style"] == "candles"
        assert sent["granularity"] == 300

    run(_run())


def test_ticks_stream_yields_messages_and_unsubscribes() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks" in msg:
                req_id = msg["req_id"]
                return [
                    {
                        "msg_type": "tick",
                        "tick": {"quote": 100.0},
                        "subscription": {"id": "sub-1"},
                        "req_id": req_id,
                    },
                    {
                        "msg_type": "tick",
                        "tick": {"quote": 100.5},
                        "subscription": {"id": "sub-1"},
                    },
                ]
            if "forget" in msg:
                return [
                    {
                        "msg_type": "forget",
                        "forget": 1,
                        "req_id": msg["req_id"],
                    }
                ]
            return []

        client, fake = make_client(responder)
        async with client:
            stream = client.ticks_stream("1HZ100V")
            collected: list[float] = []
            async for tick in stream:
                collected.append(tick["tick"]["quote"])
                if len(collected) == 2:
                    break
            # Explicit aclose ensures the generator's finally block runs and
            # we observe the auto-emitted ``forget`` deterministically.
            await stream.aclose()
        assert collected == [100.0, 100.5]
        sent_types = [next(iter(s)) for s in fake.sent]
        assert "ticks" in sent_types
        assert "forget" in sent_types  # auto-cleanup on iterator close

    run(_run())


# ---------------------------------------------------------------------------
# Endpoints: Account
# ---------------------------------------------------------------------------


def test_balance_one_shot_disables_subscribe() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("balance", {"balance": {"balance": 0}}))
        async with client:
            await client.balance()
        assert fake.sent[0]["subscribe"] == 0

    run(_run())


def test_profit_table_rejects_bad_sort() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.profit_table(sort="random")

    run(_run())


def test_statement_omits_unset_fields() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("statement", {"statement": {}}))
        async with client:
            await client.statement(limit=10)
        sent = fake.sent[0]
        assert sent["statement"] == 1
        assert sent["limit"] == 10
        assert "description" not in sent
        assert "action_type" not in sent

    run(_run())


# ---------------------------------------------------------------------------
# Endpoints: Trading
# ---------------------------------------------------------------------------


def test_proposal_requires_duration_or_date_expiry() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.proposal(
                    contract_type="CALL",
                    currency="USD",
                    underlying_symbol="1HZ100V",
                )

    run(_run())


def test_proposal_rejects_duration_and_date_expiry_together() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.proposal(
                    contract_type="CALL",
                    currency="USD",
                    underlying_symbol="1HZ100V",
                    duration=5,
                    duration_unit="t",
                    date_expiry=1_700_000_000,
                )

    run(_run())


def test_proposal_validates_enums() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.proposal(
                    contract_type="CALL",
                    currency="USD",
                    underlying_symbol="1HZ100V",
                    duration=5,
                    duration_unit="x",
                )
            with pytest.raises(ValueError):
                await client.proposal(
                    contract_type="CALL",
                    currency="USD",
                    underlying_symbol="1HZ100V",
                    duration=5,
                    duration_unit="t",
                    basis="bogus",
                )

    run(_run())


def test_proposal_serializes_full_payload() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("proposal", {"proposal": {"id": "p1"}}))
        async with client:
            await client.proposal(
                contract_type="MULTUP",
                currency="USD",
                underlying_symbol="1HZ100V",
                amount=10,
                basis="stake",
                duration=5,
                duration_unit="t",
                multiplier=20,
                limit_order={"stop_loss": 5, "take_profit": 25},
            )
        sent = fake.sent[0]
        assert sent["proposal"] == 1
        assert sent["contract_type"] == "MULTUP"
        assert sent["limit_order"] == {"stop_loss": 5, "take_profit": 25}
        assert "barrier" not in sent
        assert "cancellation" not in sent

    run(_run())


def test_buy_sends_proposal_id_and_price() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("buy", {"buy": {"contract_id": 1}}))
        async with client:
            await client.buy("propid", 10.5)
        sent = fake.sent[0]
        assert sent["buy"] == "propid"
        assert sent["price"] == 10.5
        assert "parameters" not in sent  # None stripped

    run(_run())


def test_buy_rejects_empty_proposal_id() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.buy("", 10.0)

    run(_run())


def test_sell_serializes_request() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("sell", {"sell": {"sold_for": 12}}))
        async with client:
            await client.sell(12345, 0)
        assert fake.sent[0] == {"sell": 12345, "price": 0, "req_id": fake.sent[0]["req_id"]}

    run(_run())


def test_contract_update_requires_at_least_one_field() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.contract_update(12345)

    run(_run())


def test_contract_update_preserves_explicit_none() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("contract_update", {"contract_update": {}}))
        async with client:
            await client.contract_update(12345, stop_loss=None, take_profit=15.0)
        sent = fake.sent[0]
        assert sent["limit_order"] == {"stop_loss": None, "take_profit": 15.0}

    run(_run())


def test_contract_update_history_limit_bounds() -> None:
    async def _run() -> None:
        client, _ = make_client()
        async with client:
            with pytest.raises(ValueError):
                await client.contract_update_history(1, limit=0)
            with pytest.raises(ValueError):
                await client.contract_update_history(1, limit=1000)

    run(_run())


def test_cancel_endpoint() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("cancel", {"cancel": {"sold_for": 7}}))
        async with client:
            await client.cancel(99)
        assert fake.sent[0]["cancel"] == 99

    run(_run())


# ---------------------------------------------------------------------------
# Subscription management
# ---------------------------------------------------------------------------


def test_forget_serializes_payload() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("forget", {"forget": 1}))
        async with client:
            await client.forget("abc")
        assert fake.sent[0]["forget"] == "abc"

    run(_run())


def test_forget_all_serializes_payload() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("forget_all", {"forget_all": []}))
        async with client:
            await client.forget_all(["ticks", "proposal"])
        assert fake.sent[0]["forget_all"] == ["ticks", "proposal"]

    run(_run())


def test_subscription_limit_enforced() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            if "ticks" in msg:
                return [
                    {
                        "msg_type": "tick",
                        "tick": {"quote": 1.0},
                        "subscription": {"id": f"sub-{msg['req_id']}"},
                        "req_id": msg["req_id"],
                    }
                ]
            if "forget" in msg:
                return [
                    {
                        "msg_type": "forget",
                        "forget": 1,
                        "req_id": msg["req_id"],
                    }
                ]
            return []

        client, _ = make_client(responder, max_subscriptions=2)
        async with client:
            streams = [client.ticks_stream(f"S{i}") for i in range(3)]
            # Drain the first message from the first two streams to register them
            for stream in streams[:2]:
                async for _msg in stream:
                    break
            with pytest.raises(DerivSubscriptionError):
                async for _msg in streams[2]:
                    break

    run(_run())


# ---------------------------------------------------------------------------
# Error mapping & connection lifecycle
# ---------------------------------------------------------------------------


def test_error_response_raises_typed_exception() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {
                    "msg_type": "balance",
                    "error": {
                        "code": "AuthorizationRequired",
                        "message": "Please log in.",
                    },
                    "req_id": msg["req_id"],
                }
            ]

        client, _ = make_client(responder)
        async with client:
            with pytest.raises(DerivAuthError) as excinfo:
                await client.balance()
        assert excinfo.value.code == "AuthorizationRequired"

    run(_run())


def test_generic_error_falls_back_to_base_exception() -> None:
    async def _run() -> None:
        def responder(msg: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                {
                    "msg_type": "balance",
                    "error": {"code": "WeirdError", "message": "boom"},
                    "req_id": msg["req_id"],
                }
            ]

        client, _ = make_client(responder)
        async with client:
            with pytest.raises(DerivAPIError) as excinfo:
                await client.balance()
        assert excinfo.value.code == "WeirdError"
        assert not isinstance(excinfo.value, (DerivAuthError, DerivRateLimitError))

    run(_run())


def test_send_after_close_raises() -> None:
    async def _run() -> None:
        client, _ = make_client(_echo("ping", {"ping": "pong"}))
        await client.connect()
        await client.close()
        with pytest.raises(DerivConnectionError):
            await client.ping()

    run(_run())


def test_request_timeout_surfaces_connection_error() -> None:
    async def _run() -> None:
        # No responder → request never receives a reply → times out.
        client, _ = make_client(responder=None, request_timeout=0.05)
        async with client:
            with pytest.raises(DerivConnectionError):
                await client.ping()

    run(_run())


def test_invalid_json_frame_is_dropped() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("ping", {"ping": "pong"}))
        async with client:
            await fake.push_raw("not json")
            # subsequent request must still succeed
            result = await client.ping()
        assert result["ping"] == "pong"

    run(_run())


def test_req_id_increments_monotonically() -> None:
    async def _run() -> None:
        client, fake = make_client(_echo("ping", {"ping": "pong"}))
        async with client:
            await client.ping()
            await client.ping()
            await client.ping()
        ids = [m["req_id"] for m in fake.sent]
        assert ids == sorted(set(ids)) == [1, 2, 3]

    run(_run())
