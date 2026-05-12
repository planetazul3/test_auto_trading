"""Observabilidad: logging JSON estructurado + métricas Prometheus.

* ``logging``: `JsonFormatter` para logs en formato JSON estructurado,
  `correlation_id` (`contextvars`) que se inyecta automáticamente en
  cada record, y `configure_root` para wiring de una sola llamada.
* ``metrics``: wrapper import-safe sobre `prometheus_client`. Si la
  dependencia no está, todos los `Counter/Histogram/Gauge` degradan a
  no-ops sin romper imports ni emitir warnings.
"""

from .logging import (
    JsonFormatter,
    correlation_id,
    configure_root,
    get_logger,
    new_correlation_id,
)
from .metrics import (
    HAS_PROMETHEUS,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    inference_latency_seconds,
    metrics_registry,
    signals_emitted_total,
    train_batch_duration_seconds,
    train_loss,
)

__all__ = [
    "Counter",
    "Gauge",
    "HAS_PROMETHEUS",
    "Histogram",
    "JsonFormatter",
    "MetricsRegistry",
    "configure_root",
    "correlation_id",
    "get_logger",
    "inference_latency_seconds",
    "metrics_registry",
    "new_correlation_id",
    "signals_emitted_total",
    "train_batch_duration_seconds",
    "train_loss",
]
