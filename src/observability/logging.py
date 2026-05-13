"""Logging JSON estructurado con `correlation_id` por trade/signal.

* ``JsonFormatter``: serializa cada `LogRecord` a una línea JSON con
  los campos canónicos (`ts`, `level`, `name`, `message`, `correlation_id`,
  extras vía `extra={...}`). No usa `json.dumps(record.__dict__)` porque
  pone basura interna; selecciona explícitamente.
* ``correlation_id``: `contextvars.ContextVar` thread-safe + asyncio-safe.
  Cualquier código que loguee dentro del contexto incluye el ID
  automáticamente. Útil para tracear una señal desde el WebSocket hasta
  el JSON de salida.
* ``configure_root``: configura `logging.root` con un handler stdout
  y el `JsonFormatter`. Idempotente; sólo añade el handler si no hay
  ninguno del tipo.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import sys
import uuid
from typing import Any, Optional

# Campos del LogRecord que vamos a omitir del payload JSON
# (son cosas internas de logging que no interesan al consumidor).
_RESERVED_FIELDS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName",
    "taskName",
})


# ContextVar para propagar el correlation_id a través de tareas async y
# threads (Python copia el contexto en task_factory por default).
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def new_correlation_id(prefix: Optional[str] = None) -> str:
    """Genera y setea un nuevo correlation_id en el contexto actual."""
    cid = uuid.uuid4().hex[:16]
    if prefix:
        cid = f"{prefix}-{cid}"
    correlation_id.set(cid)
    return cid


class JsonFormatter(logging.Formatter):
    """Formatter que serializa cada LogRecord como una línea JSON estable."""

    def __init__(
        self,
        *,
        ensure_ascii: bool = False,
        include_extras: bool = True,
    ) -> None:
        super().__init__()
        self.ensure_ascii = bool(ensure_ascii)
        self.include_extras = bool(include_extras)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _dt.datetime.fromtimestamp(
                record.created, tz=_dt.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        cid = correlation_id.get()
        if cid is not None:
            payload["correlation_id"] = cid
        # Stack/exc info si están.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        if self.include_extras:
            for k, v in record.__dict__.items():
                if k in _RESERVED_FIELDS or k.startswith("_"):
                    continue
                if k in payload:
                    continue
                payload[k] = _coerce(v)

        return json.dumps(payload, ensure_ascii=self.ensure_ascii, default=str)


def _coerce(value: Any) -> Any:
    """Convierte tipos no-serializables en algo JSON-compatible."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    return repr(value)


def configure_root(
    *,
    level: int = logging.INFO,
    stream: Optional[Any] = None,
    json_format: bool = True,
) -> None:
    """Configura ``logging.root`` con un handler stdout. Idempotente.

    Si ``json_format=True`` (default), usa ``JsonFormatter``; sino, el
    formato textual estándar con timestamp.
    """
    root = logging.getLogger()
    root.setLevel(level)
    stream = stream or sys.stdout
    formatter: logging.Formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    # Evitar duplicar handlers en re-imports.
    for h in root.handlers:
        if getattr(h, "_observability_managed", False):
            h.setFormatter(formatter)
            h.setLevel(level)
            return
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    # Marker para idempotencia.
    handler._observability_managed = True  # type: ignore[attr-defined]
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Wrapper opinado: devuelve un logger con `propagate=True`."""
    log = logging.getLogger(name)
    log.propagate = True
    return log


__all__ = [
    "JsonFormatter",
    "configure_root",
    "correlation_id",
    "get_logger",
    "new_correlation_id",
]
