"""Wrapper import-safe sobre ``prometheus_client``.

Si la dependencia opcional está instalada, expone `Counter`, `Gauge`
y `Histogram` reales con un `MetricsRegistry` dedicado para no
contaminar el default. Si no está, todos degradan a no-ops sin emitir
warnings ni romper imports.

Métricas estándar pre-definidas (usables por Trainer, BacktestEngine
y live loop):

* ``train_loss``: gauge etiquetado con ``stage`` (train/val).
* ``train_batch_duration_seconds``: histogram.
* ``inference_latency_seconds``: histogram etiquetado con ``mode``
  (single/batch).
* ``signals_emitted_total``: counter etiquetado con ``signal``
  (CALL/PUT/NO_TRADE) y ``contract``.

Para exponer un endpoint HTTP scrapeable:

    from src.observability.metrics import start_http_server
    start_http_server(port=9090)   # no-op si prometheus_client falta
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

try:  # pragma: no cover - dependencia opcional
    import prometheus_client as _prom
    from prometheus_client import (
        CollectorRegistry as _CollectorRegistry,
        Counter as _Counter,
        Gauge as _Gauge,
        Histogram as _Histogram,
    )
    HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    _prom = None  # type: ignore[assignment]
    _CollectorRegistry = object  # type: ignore[assignment,misc]
    _Counter = None  # type: ignore[assignment,misc]
    _Gauge = None    # type: ignore[assignment,misc]
    _Histogram = None  # type: ignore[assignment,misc]
    HAS_PROMETHEUS = False


# ---------------------------------------------------------------------------
# Stubs no-op para entornos sin prometheus_client
# ---------------------------------------------------------------------------


class _NoOpMetric:
    """Stub silencioso compatible con la API de Counter/Gauge/Histogram."""

    def __init__(self, name: str, *_: Any, **__: Any) -> None:
        self.name = name

    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":  # noqa: D401
        return self

    def inc(self, amount: float = 1.0) -> None:
        pass

    def dec(self, amount: float = 1.0) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def time(self) -> Any:
        class _Ctx:
            def __enter__(self_inner: "_Ctx") -> "_Ctx":
                return self_inner

            def __exit__(self_inner: "_Ctx", *a: Any) -> None:
                return None
        return _Ctx()


class MetricsRegistry:
    """Wrapper sobre `prometheus_client.CollectorRegistry`.

    Si prometheus está disponible mantiene su propio registry (no
    contamina el default). Caso contrario es un namespace ligero que
    devuelve stubs no-op.
    """

    def __init__(self, *, namespace: str = "ml_signal_engine") -> None:
        self.namespace = namespace
        self._registry: Optional[Any] = (
            _CollectorRegistry() if HAS_PROMETHEUS else None
        )
        self._metrics: dict[str, Any] = {}

    @property
    def registry(self) -> Optional[Any]:
        return self._registry

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def counter(
        self,
        name: str,
        description: str,
        *,
        labelnames: Iterable[str] = (),
    ) -> Any:
        return self._get_or_create(
            name, description, labelnames, kind="counter"
        )

    def gauge(
        self,
        name: str,
        description: str,
        *,
        labelnames: Iterable[str] = (),
    ) -> Any:
        return self._get_or_create(
            name, description, labelnames, kind="gauge"
        )

    def histogram(
        self,
        name: str,
        description: str,
        *,
        labelnames: Iterable[str] = (),
        buckets: Optional[Iterable[float]] = None,
    ) -> Any:
        return self._get_or_create(
            name, description, labelnames, kind="histogram", buckets=buckets
        )

    def _get_or_create(
        self,
        name: str,
        description: str,
        labelnames: Iterable[str],
        *,
        kind: str,
        buckets: Optional[Iterable[float]] = None,
    ) -> Any:
        full_name = f"{self.namespace}_{name}"
        if full_name in self._metrics:
            return self._metrics[full_name]
        if not HAS_PROMETHEUS:
            metric: Any = _NoOpMetric(full_name)
        else:
            labels = tuple(labelnames)
            if kind == "counter":
                metric = _Counter(
                    full_name, description, labels, registry=self._registry,
                )
            elif kind == "gauge":
                metric = _Gauge(
                    full_name, description, labels, registry=self._registry,
                )
            elif kind == "histogram":
                kwargs: dict[str, Any] = {"registry": self._registry}
                if buckets is not None:
                    kwargs["buckets"] = tuple(buckets)
                metric = _Histogram(
                    full_name, description, labels, **kwargs,
                )
            else:
                raise ValueError(f"unknown kind {kind!r}")
        self._metrics[full_name] = metric
        return metric

    # ------------------------------------------------------------------
    # HTTP scrape endpoint (opcional)
    # ------------------------------------------------------------------

    def start_http_server(self, port: int = 9090, addr: str = "0.0.0.0") -> bool:
        if not HAS_PROMETHEUS or self._registry is None:
            return False
        assert _prom is not None
        _prom.start_http_server(port, addr=addr, registry=self._registry)
        return True


# ---------------------------------------------------------------------------
# Default registry + métricas estándar del pipeline
# ---------------------------------------------------------------------------


metrics_registry = MetricsRegistry()

# Latencia de inferencia: buckets pensados para target p99 < 5ms (F4).
inference_latency_seconds = metrics_registry.histogram(
    "inference_latency_seconds",
    "Latencia end-to-end de generate_signal por modo (single/batch).",
    labelnames=("mode",),
    buckets=(
        1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3,
        1e-2, 2.5e-2, 5e-2, 1e-1, 5e-1, 1.0,
    ),
)
train_batch_duration_seconds = metrics_registry.histogram(
    "train_batch_duration_seconds",
    "Duración de un step de entrenamiento (fwd+bwd+optim).",
    labelnames=("stage",),
)
train_loss = metrics_registry.gauge(
    "train_loss",
    "Última loss reportada (etiquetada por stage train/val y por contract).",
    labelnames=("stage", "contract"),
)
signals_emitted_total = metrics_registry.counter(
    "signals_emitted_total",
    "Cantidad acumulada de señales emitidas por (signal, contract).",
    labelnames=("signal", "contract"),
)


# ---------------------------------------------------------------------------
# Re-exports para uso externo
# ---------------------------------------------------------------------------

# Exponer los símbolos del módulo prom (si están) o los stubs equivalentes
# como ``Counter``/`Gauge`/`Histogram` para que código de usuario haga
# `from src.observability.metrics import Counter` sin depender del flag.
Counter: Any
Gauge: Any
Histogram: Any
if HAS_PROMETHEUS:
    Counter = _Counter
    Gauge = _Gauge
    Histogram = _Histogram
else:  # pragma: no cover - solo si prom no está
    Counter = _NoOpMetric
    Gauge = _NoOpMetric
    Histogram = _NoOpMetric


def start_http_server(port: int = 9090, addr: str = "0.0.0.0") -> bool:
    """Atajo sobre el registry default. Devuelve False si prom falta."""
    return metrics_registry.start_http_server(port=port, addr=addr)


__all__ = [
    "Counter",
    "Gauge",
    "HAS_PROMETHEUS",
    "Histogram",
    "MetricsRegistry",
    "inference_latency_seconds",
    "metrics_registry",
    "signals_emitted_total",
    "start_http_server",
    "train_batch_duration_seconds",
    "train_loss",
]
