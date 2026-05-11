"""Conector asíncrono para la Deriv API v2 (WebSocket + REST OAuth/OTP).

Cubre los 22 endpoints WebSocket documentados en
``https://developers.deriv.com/llms.txt`` agrupados en cinco categorías:

* Account: ``balance``, ``portfolio``, ``profit_table``, ``statement``, ``transaction``
* Market Data: ``active_symbols``, ``contracts_for``, ``contracts_list``,
  ``ticks``, ``ticks_history``
* Trading: ``proposal``, ``buy``, ``sell``, ``proposal_open_contract``,
  ``contract_update``, ``contract_update_history``, ``cancel``
* Subscription: ``forget``, ``forget_all``
* System: ``ping``, ``time``, ``trading_times``

También incluye utilidades para el flujo OAuth2 + PKCE y la obtención del
OTP previo a la conexión a los endpoints autenticados ``demo`` y ``real``.
"""

from .auth import (
    DEFAULT_AUTH_URL,
    DEFAULT_REST_BASE,
    DEFAULT_TOKEN_URL,
    DerivOAuth2,
    DerivOTPClient,
    PKCEParameters,
)
from .client import (
    DEFAULT_DEMO_URL,
    DEFAULT_PUBLIC_URL,
    DEFAULT_REAL_URL,
    CANDLE_GRANULARITIES,
    DerivWebSocketClient,
)
from .exceptions import (
    DerivAPIError,
    DerivAuthError,
    DerivConnectionError,
    DerivProtocolError,
    DerivRateLimitError,
    DerivSubscriptionError,
)
from .ingest import (
    AdaptiveRateLimiter,
    IngestStats,
    MarketDataIngester,
)
from .storage import (
    SCHEMA_VERSION,
    CandleRow,
    DuckDBStore,
    TickRow,
)

__all__ = [
    "AdaptiveRateLimiter",
    "CANDLE_GRANULARITIES",
    "CandleRow",
    "DEFAULT_AUTH_URL",
    "DEFAULT_DEMO_URL",
    "DEFAULT_PUBLIC_URL",
    "DEFAULT_REAL_URL",
    "DEFAULT_REST_BASE",
    "DEFAULT_TOKEN_URL",
    "DerivAPIError",
    "DerivAuthError",
    "DerivConnectionError",
    "DerivOAuth2",
    "DerivOTPClient",
    "DerivProtocolError",
    "DerivRateLimitError",
    "DerivSubscriptionError",
    "DerivWebSocketClient",
    "DuckDBStore",
    "IngestStats",
    "MarketDataIngester",
    "PKCEParameters",
    "SCHEMA_VERSION",
    "TickRow",
]
