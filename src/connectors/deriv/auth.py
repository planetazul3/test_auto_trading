"""Utilidades para el flujo OAuth2 + PKCE y la obtención del OTP de Deriv.

Las clases aquí definidas no realizan I/O por sí mismas: reciben un
cliente HTTP asíncrono compatible con ``httpx.AsyncClient`` (cualquier
objeto con métodos ``get``/``post`` que devuelvan una respuesta con
``raise_for_status`` y ``json``). Esto facilita el testing y permite
reutilizar el cliente entre múltiples conexiones.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlencode

DEFAULT_AUTH_URL = "https://auth.deriv.com/oauth2/auth"
DEFAULT_TOKEN_URL = "https://auth.deriv.com/oauth2/token"
DEFAULT_REST_BASE = "https://api.derivws.com"

_PKCE_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-._~"
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


@dataclass(frozen=True)
class PKCEParameters:
    """Par ``code_verifier`` / ``code_challenge`` derivado por SHA-256."""

    code_verifier: str
    code_challenge: str
    code_challenge_method: str = "S256"


class _AsyncHTTPResponse(Protocol):
    def raise_for_status(self) -> Any: ...
    def json(self) -> Any: ...


class _AsyncHTTPClient(Protocol):
    async def get(self, url: str, *, headers: dict[str, str] | None = ...) -> _AsyncHTTPResponse: ...
    async def post(
        self,
        url: str,
        *,
        data: Any | None = ...,
        json: Any | None = ...,
        headers: dict[str, str] | None = ...,
    ) -> _AsyncHTTPResponse: ...


class DerivOAuth2:
    """Helper para el flujo OAuth2 Authorization Code + PKCE de Deriv."""

    def __init__(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        auth_url: str = DEFAULT_AUTH_URL,
        token_url: str = DEFAULT_TOKEN_URL,
    ) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        if not redirect_uri:
            raise ValueError("redirect_uri is required")
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.auth_url = auth_url
        self.token_url = token_url

    @staticmethod
    def generate_pkce(length: int = 64) -> PKCEParameters:
        if not 43 <= length <= 128:
            raise ValueError("PKCE verifier length must be between 43 and 128")
        verifier = "".join(secrets.choice(_PKCE_ALPHABET) for _ in range(length))
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        return PKCEParameters(verifier, challenge)

    @staticmethod
    def generate_state(n_bytes: int = 16) -> str:
        if n_bytes <= 0:
            raise ValueError("n_bytes must be > 0")
        return secrets.token_hex(n_bytes)

    def build_authorization_url(
        self,
        *,
        state: str,
        code_challenge: str,
        scope: str = "trade",
        code_challenge_method: str = "S256",
        prompt: str | None = None,
        sidc: str | None = None,
        utm_campaign: str | None = None,
        utm_medium: str | None = None,
        utm_source: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> str:
        if scope not in ("trade", "admin"):
            raise ValueError("scope must be 'trade' or 'admin'")
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
        optional = {
            "prompt": prompt,
            "sidc": sidc,
            "utm_campaign": utm_campaign,
            "utm_medium": utm_medium,
            "utm_source": utm_source,
        }
        for key, value in optional.items():
            if value is not None:
                params[key] = value
        if extra:
            params.update(extra)
        return f"{self.auth_url}?{urlencode(params)}"

    def build_signup_url(
        self,
        *,
        state: str,
        code_challenge: str,
        scope: str = "trade",
        sidc: str | None = None,
        utm_campaign: str | None = None,
        utm_medium: str | None = "affiliate",
        utm_source: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> str:
        return self.build_authorization_url(
            state=state,
            code_challenge=code_challenge,
            scope=scope,
            prompt="registration",
            sidc=sidc,
            utm_campaign=utm_campaign,
            utm_medium=utm_medium,
            utm_source=utm_source,
            extra=extra,
        )

    def build_token_payload(self, *, code: str, code_verifier: str) -> dict[str, str]:
        return {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": self.redirect_uri,
        }

    async def exchange_code(
        self,
        *,
        code: str,
        code_verifier: str,
        http_client: _AsyncHTTPClient,
    ) -> dict[str, Any]:
        response = await http_client.post(
            self.token_url,
            data=self.build_token_payload(code=code, code_verifier=code_verifier),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())


class DerivOTPClient:
    """Cliente REST mínimo para gestión de cuentas Options y emisión de OTP."""

    def __init__(
        self,
        *,
        http_client: _AsyncHTTPClient,
        access_token: str,
        app_id: str,
        base_url: str = DEFAULT_REST_BASE,
    ) -> None:
        if not access_token:
            raise ValueError("access_token is required")
        if not app_id:
            raise ValueError("app_id is required")
        self.http = http_client
        self.access_token = access_token
        self.app_id = app_id
        self.base_url = base_url.rstrip("/")

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Deriv-App-ID": self.app_id,
        }

    async def list_accounts(self) -> dict[str, Any]:
        response = await self.http.get(
            f"{self.base_url}/trading/v1/options/accounts",
            headers=self._headers,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def create_account(
        self,
        *,
        currency: str = "USD",
        group: str = "row",
        account_type: str = "demo",
    ) -> dict[str, Any]:
        if account_type not in ("demo", "real"):
            raise ValueError("account_type must be 'demo' or 'real'")
        response = await self.http.post(
            f"{self.base_url}/trading/v1/options/accounts",
            headers={**self._headers, "Content-Type": "application/json"},
            json={
                "currency": currency,
                "group": group,
                "account_type": account_type,
            },
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def reset_demo_balance(self, account_id: str) -> dict[str, Any]:
        response = await self.http.post(
            f"{self.base_url}/trading/v1/options/accounts/{account_id}/reset-demo-balance",
            headers=self._headers,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def request_otp(self, account_id: str) -> str:
        """Solicita un OTP y devuelve la URL WebSocket lista para conectar."""
        response = await self.http.post(
            f"{self.base_url}/trading/v1/options/accounts/{account_id}/otp",
            headers=self._headers,
        )
        response.raise_for_status()
        payload = response.json()
        url = payload.get("data", {}).get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("OTP response missing data.url")
        return url

    async def health(self) -> dict[str, Any]:
        response = await self.http.get(f"{self.base_url}/v1/health")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
