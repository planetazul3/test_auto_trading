"""Excepciones específicas del conector Deriv."""

from __future__ import annotations


class DerivAPIError(Exception):
    """Error devuelto por la API en el campo ``error`` de un mensaje."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        msg_type: str | None = None,
        req_id: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.msg_type = msg_type
        self.req_id = req_id
        super().__init__(f"{code}: {message}")


class DerivAuthError(DerivAPIError):
    """Token inválido, expirado o ausencia de autorización."""


class DerivRateLimitError(DerivAPIError):
    """El servidor reportó saturación del límite de peticiones."""


class DerivConnectionError(Exception):
    """Fallo de transporte WebSocket (timeout, desconexión, cierre)."""


class DerivSubscriptionError(Exception):
    """Se intentó superar el límite de suscripciones por conexión."""


class DerivProtocolError(Exception):
    """Respuesta inesperada o malformada del servidor."""
