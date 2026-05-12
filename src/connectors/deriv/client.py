"""Cliente WebSocket asíncrono para la Deriv API v2.

Implementa los 22 endpoints WebSocket documentados en ``llms.txt``,
respetando el esquema exacto de cada uno (parámetros opcionales, tipos
permitidos, enums). El cliente correlaciona peticiones/respuestas por
``req_id`` y enruta los mensajes de streaming por ``subscription.id``.

Limitaciones operativas aplicadas por defecto (sección Rate Limits del
``llms.txt`` de Deriv):

* 100 peticiones/segundo por conexión (token bucket).
* 100 suscripciones activas como máximo por conexión.
* ``ping_interval`` de 30 s (delegado en la librería ``websockets``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError:  # pragma: no cover - dependencia opcional
    websockets = None  # type: ignore[assignment]
    ClientConnection = Any  # type: ignore[misc, assignment]

from .exceptions import (
    DerivAPIError,
    DerivAuthError,
    DerivConnectionError,
    DerivRateLimitError,
    DerivSubscriptionError,
)
from .rate_limiter import AsyncTokenBucket

logger = logging.getLogger(__name__)

DEFAULT_PUBLIC_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
DEFAULT_DEMO_URL = "wss://api.derivws.com/trading/v1/options/ws/demo"
DEFAULT_REAL_URL = "wss://api.derivws.com/trading/v1/options/ws/real"

#: Granularidades permitidas en ``ticks_history`` cuando ``style='candles'``.
CANDLE_GRANULARITIES: frozenset[int] = frozenset(
    {60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400}
)

_DURATION_UNITS: frozenset[str] = frozenset({"s", "m", "h", "d", "t"})
_BASIS_VALUES: frozenset[str] = frozenset({"stake", "payout"})
_ACTIVE_SYMBOLS_MODES: frozenset[str] = frozenset({"brief", "full"})
_HISTORY_STYLES: frozenset[str] = frozenset({"ticks", "candles"})

_MAX_SUBSCRIPTIONS_DEFAULT = 100
_REQUESTS_PER_SECOND_DEFAULT = 100.0

_ERROR_CODE_TO_EXC: dict[str, type[DerivAPIError]] = {
    "AuthorizationRequired": DerivAuthError,
    "InvalidToken": DerivAuthError,
    "Unauthorized": DerivAuthError,
    "RateLimit": DerivRateLimitError,
}

WSFactory = Callable[[str], Awaitable[Any]]


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<UNSET>"


_UNSET = _Unset()


class _PendingRequest:
    __slots__ = ("future", "subscribe")

    def __init__(self, *, subscribe: bool) -> None:
        self.future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self.subscribe = subscribe


class _Subscription:
    __slots__ = ("id", "queue", "active", "msg_type", "forget_sent")

    def __init__(self, sub_id: str, msg_type: str | None) -> None:
        self.id = sub_id
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.active = True
        self.msg_type = msg_type
        self.forget_sent = False


def _is_otp_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("ws", "wss") and "otp=" in (parsed.query or "")


def _make_error(error: dict[str, Any], envelope: dict[str, Any]) -> DerivAPIError:
    code = str(error.get("code") or "UnknownError")
    message = str(error.get("message") or "")
    cls = _ERROR_CODE_TO_EXC.get(code, DerivAPIError)
    return cls(
        code=code,
        message=message,
        msg_type=envelope.get("msg_type"),
        req_id=envelope.get("req_id"),
    )


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None}


class DerivWebSocketClient:
    """Conector WebSocket asíncrono para Deriv API v2.

    Tres formas habituales de instanciar:

    * :meth:`public` — datos públicos sin autenticación.
    * :meth:`demo` — cuenta demo autenticada vía OTP (URL devuelta por
      :class:`DerivOTPClient.request_otp`).
    * :meth:`real` — cuenta real autenticada vía OTP.

    Uso típico::

        async with DerivWebSocketClient.public() as client:
            print(await client.time())
            async for tick in client.ticks_stream("1HZ100V"):
                ...

    Para flujos autenticados::

        otp_url = await otp_client.request_otp(account_id)
        async with DerivWebSocketClient.demo(otp_url) as client:
            balance = await client.balance()
    """

    def __init__(
        self,
        url: str,
        *,
        ping_interval: float | None = 30.0,
        request_timeout: float = 30.0,
        rate_limit_per_second: float = _REQUESTS_PER_SECOND_DEFAULT,
        max_subscriptions: int = _MAX_SUBSCRIPTIONS_DEFAULT,
        extra_headers: dict[str, str] | None = None,
        ws_factory: WSFactory | None = None,
    ) -> None:
        if not url:
            raise ValueError("url is required")
        self.url = url
        self.ping_interval = ping_interval
        self.request_timeout = request_timeout
        self.max_subscriptions = max_subscriptions
        self.extra_headers = dict(extra_headers or {})
        self._ws_factory = ws_factory
        self._rate_limiter = AsyncTokenBucket(
            rate=rate_limit_per_second, capacity=rate_limit_per_second
        )

        self._ws: Any | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_req_id = 1
        self._pending: dict[int, _PendingRequest] = {}
        self._subscriptions: dict[str, _Subscription] = {}
        self._closed = False
        self._connect_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Constructores convenientes
    # ------------------------------------------------------------------

    @classmethod
    def public(cls, **kwargs: Any) -> "DerivWebSocketClient":
        return cls(DEFAULT_PUBLIC_URL, **kwargs)

    @classmethod
    def demo(cls, otp_url: str, **kwargs: Any) -> "DerivWebSocketClient":
        if not _is_otp_url(otp_url):
            raise ValueError(
                "demo() expects the OTP-authenticated WebSocket URL returned "
                "by DerivOTPClient.request_otp(account_id)"
            )
        return cls(otp_url, **kwargs)

    @classmethod
    def real(cls, otp_url: str, **kwargs: Any) -> "DerivWebSocketClient":
        if not _is_otp_url(otp_url):
            raise ValueError(
                "real() expects the OTP-authenticated WebSocket URL returned "
                "by DerivOTPClient.request_otp(account_id)"
            )
        return cls(otp_url, **kwargs)

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._closed:
                raise DerivConnectionError("client has been closed")
            if self._ws is not None and not self._is_ws_closed():
                return
            await self._open_socket()

    async def _open_socket(self) -> None:
        if self._ws_factory is not None:
            self._ws = await self._ws_factory(self.url)
        else:
            if websockets is None:
                raise DerivConnectionError(
                    "the 'websockets' package is required; install the optional "
                    "extra: pip install '.[deriv]'"
                )
            kwargs: dict[str, Any] = {}
            if self.ping_interval is not None:
                kwargs["ping_interval"] = self.ping_interval
            if self.extra_headers:
                kwargs["additional_headers"] = self.extra_headers
            self._ws = await websockets.connect(self.url, **kwargs)
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="deriv-ws-reader"
        )

    def _is_ws_closed(self) -> bool:
        ws = self._ws
        if ws is None:
            return True
        closed = getattr(ws, "closed", None)
        if closed is not None:
            return bool(closed)
        return getattr(ws, "close_code", None) is not None

    async def close(self) -> None:
        self._closed = True
        reader = self._reader_task
        self._reader_task = None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        self._fail_all(DerivConnectionError("connection closed"))

    async def __aenter__(self) -> "DerivWebSocketClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Lector y dispatcher
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Discarded non-JSON frame from Deriv: %r", raw[:200])
                    continue
                if not isinstance(msg, dict):
                    logger.warning("Discarded non-object frame from Deriv: %r", msg)
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - depende del transporte
            logger.info("Deriv WS reader stopped: %s", exc)
        finally:
            self._fail_all(DerivConnectionError("connection closed"))

    def _dispatch(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("req_id")
        sub_block = msg.get("subscription")
        sub_id = (
            sub_block.get("id") if isinstance(sub_block, dict) else None
        )
        error = msg.get("error")

        if isinstance(req_id, int) and req_id in self._pending:
            pending = self._pending.pop(req_id)
            if pending.subscribe and sub_id and not error:
                sub = self._subscriptions.get(sub_id) or _Subscription(
                    sub_id, msg.get("msg_type")
                )
                self._subscriptions[sub_id] = sub
                sub.queue.put_nowait(msg)
            if not pending.future.done():
                if error:
                    pending.future.set_exception(_make_error(error, msg))
                else:
                    pending.future.set_result(msg)
            return

        if isinstance(sub_id, str) and sub_id in self._subscriptions:
            sub = self._subscriptions[sub_id]
            if error:
                sub.queue.put_nowait({"__error__": _make_error(error, msg)})
            else:
                sub.queue.put_nowait(msg)
            return

        logger.debug(
            "Dropping unmatched Deriv frame: msg_type=%s req_id=%s",
            msg.get("msg_type"),
            req_id,
        )

    def _fail_all(self, exc: Exception) -> None:
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(exc)
        self._pending.clear()
        for sub in list(self._subscriptions.values()):
            if sub.active:
                sub.active = False
                sub.queue.put_nowait(None)

    # ------------------------------------------------------------------
    # Núcleo: send / subscribe
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        if self._closed:
            raise DerivConnectionError("client has been closed")
        if self._ws is None or self._is_ws_closed():
            await self.connect()

    def _allocate_req_id(self) -> int:
        rid = self._next_req_id
        self._next_req_id += 1
        return rid

    async def send(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Envía una petición de un solo turno y devuelve la respuesta correlada."""
        return await self._request(payload, subscribe=False, timeout=timeout)

    async def _request(
        self,
        payload: dict[str, Any],
        *,
        subscribe: bool,
        timeout: float | None,
    ) -> dict[str, Any]:
        await self._ensure_connected()
        if subscribe and len(self._subscriptions) >= self.max_subscriptions:
            raise DerivSubscriptionError(
                f"subscription limit reached ({self.max_subscriptions} active)"
            )
        req_id = self._allocate_req_id()
        out = _compact(payload)
        out["req_id"] = req_id
        if subscribe:
            out["subscribe"] = 1
        await self._rate_limiter.acquire()
        pending = _PendingRequest(subscribe=subscribe)
        self._pending[req_id] = pending
        assert self._ws is not None
        try:
            await self._ws.send(json.dumps(out))
        except Exception as exc:
            self._pending.pop(req_id, None)
            raise DerivConnectionError(f"send failed: {exc}") from exc
        effective_timeout = timeout if timeout is not None else self.request_timeout
        try:
            return await asyncio.wait_for(pending.future, timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(req_id, None)
            raise DerivConnectionError(
                f"request timed out after {effective_timeout}s "
                f"(msg_type={out.get('msg_type') or next(iter(payload))})"
            ) from exc

    async def _subscribe(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        first = await self._request(
            payload, subscribe=True, timeout=self.request_timeout
        )
        sub_block = first.get("subscription") or {}
        sub_id = sub_block.get("id") if isinstance(sub_block, dict) else None
        if not isinstance(sub_id, str):
            yield first
            return
        sub = self._subscriptions[sub_id]
        try:
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                if isinstance(item, dict) and "__error__" in item:
                    raise item["__error__"]
                yield item
        finally:
            sub.active = False
            self._subscriptions.pop(sub_id, None)
            if (
                not sub.forget_sent
                and self._ws is not None
                and not self._is_ws_closed()
                and not self._closed
            ):
                with contextlib.suppress(Exception):
                    await self.forget(sub_id)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def balance(self) -> dict[str, Any]:
        return await self.send({"balance": 1, "subscribe": 0})

    def balance_stream(self) -> AsyncIterator[dict[str, Any]]:
        return self._subscribe({"balance": 1})

    async def portfolio(
        self, *, contract_type: list[str] | None = None
    ) -> dict[str, Any]:
        return await self.send(
            {"portfolio": 1, "contract_type": contract_type}
        )

    async def profit_table(
        self,
        *,
        description: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
        contract_type: list[str] | None = None,
    ) -> dict[str, Any]:
        if sort is not None and sort not in ("ASC", "DESC"):
            raise ValueError("sort must be 'ASC' or 'DESC'")
        return await self.send(
            {
                "profit_table": 1,
                "description": description,
                "limit": limit,
                "offset": offset,
                "sort": sort,
                "date_from": date_from,
                "date_to": date_to,
                "contract_type": contract_type,
            }
        )

    async def statement(
        self,
        *,
        description: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
        action_type: str | None = None,
        date_from: int | None = None,
        date_to: int | None = None,
    ) -> dict[str, Any]:
        return await self.send(
            {
                "statement": 1,
                "description": description,
                "limit": limit,
                "offset": offset,
                "action_type": action_type,
                "date_from": date_from,
                "date_to": date_to,
            }
        )

    def transaction_stream(self) -> AsyncIterator[dict[str, Any]]:
        return self._subscribe({"transaction": 1})

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    async def active_symbols(
        self,
        *,
        active_symbols: str = "brief",
        contract_type: list[str] | None = None,
    ) -> dict[str, Any]:
        if active_symbols not in _ACTIVE_SYMBOLS_MODES:
            raise ValueError("active_symbols must be 'brief' or 'full'")
        return await self.send(
            {"active_symbols": active_symbols, "contract_type": contract_type}
        )

    async def contracts_for(
        self,
        symbol: str,
        *,
        currency: str | None = None,
        landing_company: str | None = None,
        product_type: str | None = None,
    ) -> dict[str, Any]:
        if not symbol:
            raise ValueError("symbol is required")
        return await self.send(
            {
                "contracts_for": symbol,
                "currency": currency,
                "landing_company": landing_company,
                "product_type": product_type,
            }
        )

    async def contracts_list(self) -> dict[str, Any]:
        return await self.send({"contracts_list": 1})

    def ticks_stream(
        self, symbols: str | list[str]
    ) -> AsyncIterator[dict[str, Any]]:
        if not symbols:
            raise ValueError("at least one symbol is required")
        return self._subscribe({"ticks": symbols})

    async def ticks_history(
        self,
        symbol: str,
        *,
        end: str | int = "latest",
        start: int | None = None,
        count: int | None = None,
        style: str = "ticks",
        granularity: int | None = None,
        adjust_start_time: int | None = None,
    ) -> dict[str, Any]:
        self._validate_ticks_history(style, granularity)
        return await self.send(
            {
                "ticks_history": symbol,
                "end": end,
                "start": start,
                "count": count,
                "style": style,
                "granularity": granularity,
                "adjust_start_time": adjust_start_time,
            }
        )

    def ticks_history_stream(
        self,
        symbol: str,
        *,
        end: str | int = "latest",
        start: int | None = None,
        count: int | None = None,
        style: str = "ticks",
        granularity: int | None = None,
        adjust_start_time: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self._validate_ticks_history(style, granularity)
        return self._subscribe(
            {
                "ticks_history": symbol,
                "end": end,
                "start": start,
                "count": count,
                "style": style,
                "granularity": granularity,
                "adjust_start_time": adjust_start_time,
            }
        )

    @staticmethod
    def _validate_ticks_history(style: str, granularity: int | None) -> None:
        if style not in _HISTORY_STYLES:
            raise ValueError("style must be 'ticks' or 'candles'")
        if granularity is not None and granularity not in CANDLE_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {sorted(CANDLE_GRANULARITIES)}"
            )

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    async def proposal(
        self,
        *,
        contract_type: str,
        currency: str,
        underlying_symbol: str,
        amount: float | None = None,
        basis: str | None = None,
        duration: int | None = None,
        duration_unit: str | None = None,
        date_expiry: int | None = None,
        date_start: int | None = None,
        barrier: str | None = None,
        barrier2: str | None = None,
        growth_rate: float | None = None,
        cancellation: str | None = None,
        limit_order: dict[str, float | None] | None = None,
        multiplier: float | None = None,
        payout_per_point: float | None = None,
        selected_tick: int | None = None,
        product_type: str | None = None,
        trading_period_start: int | None = None,
    ) -> dict[str, Any]:
        payload = self._build_proposal_payload(
            contract_type=contract_type,
            currency=currency,
            underlying_symbol=underlying_symbol,
            amount=amount,
            basis=basis,
            duration=duration,
            duration_unit=duration_unit,
            date_expiry=date_expiry,
            date_start=date_start,
            barrier=barrier,
            barrier2=barrier2,
            growth_rate=growth_rate,
            cancellation=cancellation,
            limit_order=limit_order,
            multiplier=multiplier,
            payout_per_point=payout_per_point,
            selected_tick=selected_tick,
            product_type=product_type,
            trading_period_start=trading_period_start,
        )
        return await self.send(payload)

    def proposal_stream(
        self,
        *,
        contract_type: str,
        currency: str,
        underlying_symbol: str,
        amount: float | None = None,
        basis: str | None = None,
        duration: int | None = None,
        duration_unit: str | None = None,
        date_expiry: int | None = None,
        date_start: int | None = None,
        barrier: str | None = None,
        barrier2: str | None = None,
        growth_rate: float | None = None,
        cancellation: str | None = None,
        limit_order: dict[str, float | None] | None = None,
        multiplier: float | None = None,
        payout_per_point: float | None = None,
        selected_tick: int | None = None,
        product_type: str | None = None,
        trading_period_start: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._build_proposal_payload(
            contract_type=contract_type,
            currency=currency,
            underlying_symbol=underlying_symbol,
            amount=amount,
            basis=basis,
            duration=duration,
            duration_unit=duration_unit,
            date_expiry=date_expiry,
            date_start=date_start,
            barrier=barrier,
            barrier2=barrier2,
            growth_rate=growth_rate,
            cancellation=cancellation,
            limit_order=limit_order,
            multiplier=multiplier,
            payout_per_point=payout_per_point,
            selected_tick=selected_tick,
            product_type=product_type,
            trading_period_start=trading_period_start,
        )
        return self._subscribe(payload)

    @staticmethod
    def _build_proposal_payload(
        *,
        contract_type: str,
        currency: str,
        underlying_symbol: str,
        amount: float | None,
        basis: str | None,
        duration: int | None,
        duration_unit: str | None,
        date_expiry: int | None,
        date_start: int | None,
        barrier: str | None,
        barrier2: str | None,
        growth_rate: float | None,
        cancellation: str | None,
        limit_order: dict[str, float | None] | None,
        multiplier: float | None,
        payout_per_point: float | None,
        selected_tick: int | None,
        product_type: str | None,
        trading_period_start: int | None,
    ) -> dict[str, Any]:
        if not contract_type:
            raise ValueError("contract_type is required")
        if not currency:
            raise ValueError("currency is required")
        if not underlying_symbol:
            raise ValueError("underlying_symbol is required")
        if basis is not None and basis not in _BASIS_VALUES:
            raise ValueError("basis must be 'stake' or 'payout'")
        if duration_unit is not None and duration_unit not in _DURATION_UNITS:
            raise ValueError("duration_unit must be one of: s, m, h, d, t")
        if duration is None and date_expiry is None:
            raise ValueError(
                "proposal requires either 'duration' or 'date_expiry'"
            )
        if duration is not None and date_expiry is not None:
            raise ValueError(
                "proposal accepts either 'duration' or 'date_expiry', not both"
            )
        return {
            "proposal": 1,
            "contract_type": contract_type,
            "currency": currency,
            "underlying_symbol": underlying_symbol,
            "amount": amount,
            "basis": basis,
            "duration": duration,
            "duration_unit": duration_unit,
            "date_expiry": date_expiry,
            "date_start": date_start,
            "barrier": barrier,
            "barrier2": barrier2,
            "growth_rate": growth_rate,
            "cancellation": cancellation,
            "limit_order": limit_order,
            "multiplier": multiplier,
            "payout_per_point": payout_per_point,
            "selected_tick": selected_tick,
            "product_type": product_type,
            "trading_period_start": trading_period_start,
        }

    async def buy(
        self,
        proposal_id: str,
        price: float,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not proposal_id:
            raise ValueError("proposal_id is required")
        return await self.send(
            {"buy": proposal_id, "price": price, "parameters": parameters}
        )

    def buy_stream(
        self,
        proposal_id: str,
        price: float,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not proposal_id:
            raise ValueError("proposal_id is required")
        return self._subscribe(
            {"buy": proposal_id, "price": price, "parameters": parameters}
        )

    async def sell(self, contract_id: int, price: float) -> dict[str, Any]:
        return await self.send({"sell": contract_id, "price": price})

    async def proposal_open_contract(
        self, *, contract_id: int | None = None
    ) -> dict[str, Any]:
        return await self.send(
            {"proposal_open_contract": 1, "contract_id": contract_id}
        )

    def proposal_open_contract_stream(
        self, *, contract_id: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        return self._subscribe(
            {"proposal_open_contract": 1, "contract_id": contract_id}
        )

    async def contract_update(
        self,
        contract_id: int,
        *,
        stop_loss: float | None | _Unset = _UNSET,
        take_profit: float | None | _Unset = _UNSET,
    ) -> dict[str, Any]:
        """Actualiza ``stop_loss``/``take_profit`` de un contrato abierto.

        Para cancelar un límite existente, pasa explícitamente ``None``.
        Omitir el argumento deja el valor previo sin cambios.
        """
        limit_order: dict[str, float | None] = {}
        if not isinstance(stop_loss, _Unset):
            limit_order["stop_loss"] = stop_loss
        if not isinstance(take_profit, _Unset):
            limit_order["take_profit"] = take_profit
        if not limit_order:
            raise ValueError(
                "contract_update requires at least one of 'stop_loss' or 'take_profit'"
            )
        return await self.send(
            {
                "contract_update": 1,
                "contract_id": contract_id,
                "limit_order": limit_order,
            }
        )

    async def contract_update_history(
        self,
        contract_id: int,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        if limit is not None and not 1 <= limit <= 999:
            raise ValueError("limit must be between 1 and 999")
        return await self.send(
            {
                "contract_update_history": 1,
                "contract_id": contract_id,
                "limit": limit,
            }
        )

    async def cancel(self, contract_id: int) -> dict[str, Any]:
        return await self.send({"cancel": contract_id})

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def forget(self, subscription_id: str) -> dict[str, Any]:
        if not subscription_id:
            raise ValueError("subscription_id is required")
        result = await self.send({"forget": subscription_id})
        sub = self._subscriptions.get(subscription_id)
        if sub is not None:
            sub.forget_sent = True
            if sub.active:
                sub.active = False
                sub.queue.put_nowait(None)
        return result

    async def forget_all(self, types: str | list[str]) -> dict[str, Any]:
        if not types:
            raise ValueError("types is required")
        targets: set[str] = {types} if isinstance(types, str) else set(types)
        result = await self.send({"forget_all": types})
        for sub in list(self._subscriptions.values()):
            if sub.msg_type in targets or not sub.msg_type:
                sub.forget_sent = True
                if sub.active:
                    sub.active = False
                    sub.queue.put_nowait(None)
        return result

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    async def ping(self) -> dict[str, Any]:
        return await self.send({"ping": 1})

    async def time(self) -> dict[str, Any]:
        return await self.send({"time": 1})

    async def trading_times(self, date: str = "today") -> dict[str, Any]:
        if not date:
            raise ValueError("date is required ('today' or 'yyyy-mm-dd')")
        return await self.send({"trading_times": date})

    # ------------------------------------------------------------------
    # Introspección
    # ------------------------------------------------------------------

    @property
    def active_subscriptions(self) -> int:
        return sum(1 for s in self._subscriptions.values() if s.active)

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._is_ws_closed()
