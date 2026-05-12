"""Tests para src/observability (E1 + E2)."""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from src.observability.logging import (
    JsonFormatter,
    configure_root,
    correlation_id,
    get_logger,
    new_correlation_id,
)
from src.observability.metrics import (
    HAS_PROMETHEUS,
    MetricsRegistry,
    inference_latency_seconds,
    signals_emitted_total,
    train_loss,
)


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


def test_json_formatter_serializes_record_with_canonical_fields() -> None:
    rec = logging.LogRecord(
        name="test.logger", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hello %s", args=("world",), exc_info=None,
    )
    out = JsonFormatter().format(rec)
    payload = json.loads(out)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert "ts" in payload


def test_json_formatter_includes_correlation_id_when_set() -> None:
    cid = new_correlation_id(prefix="sig")
    try:
        rec = logging.LogRecord(
            name="x", level=logging.INFO, pathname=__file__,
            lineno=1, msg="m", args=(), exc_info=None,
        )
        payload = json.loads(JsonFormatter().format(rec))
        assert payload["correlation_id"] == cid
        assert cid.startswith("sig-")
    finally:
        correlation_id.set(None)


def test_json_formatter_omits_correlation_id_when_unset() -> None:
    correlation_id.set(None)
    rec = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__,
        lineno=1, msg="m", args=(), exc_info=None,
    )
    payload = json.loads(JsonFormatter().format(rec))
    assert "correlation_id" not in payload


def test_json_formatter_includes_extras() -> None:
    log = get_logger("test.extras")
    log.handlers.clear()
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False
    try:
        log.info("trade emitted", extra={"contract": "CALLPUT", "horizon": 3})
        out = buf.getvalue().strip()
        payload = json.loads(out)
        assert payload["contract"] == "CALLPUT"
        assert payload["horizon"] == 3
    finally:
        log.removeHandler(h)


def test_json_formatter_serializes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord(
            name="x", level=logging.ERROR, pathname=__file__,
            lineno=1, msg="oops", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(JsonFormatter().format(rec))
    assert "exception" in payload
    assert "ValueError" in payload["exception"]


def test_configure_root_is_idempotent() -> None:
    """Llamar configure_root dos veces no debe duplicar handlers."""
    root = logging.getLogger()
    initial_handlers = list(root.handlers)
    try:
        root.handlers.clear()
        configure_root(level=logging.INFO)
        n_after_first = len(root.handlers)
        configure_root(level=logging.WARNING)
        n_after_second = len(root.handlers)
        assert n_after_first == n_after_second == 1
        # El re-configure ajusta nivel.
        assert root.handlers[0].level == logging.WARNING
    finally:
        root.handlers = initial_handlers


def test_new_correlation_id_is_unique() -> None:
    a = new_correlation_id()
    b = new_correlation_id()
    assert a != b


# ---------------------------------------------------------------------------
# MetricsRegistry (with prometheus_client available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PROMETHEUS, reason="prometheus_client not installed")
def test_metrics_registry_creates_real_metrics_when_prometheus_available() -> None:
    reg = MetricsRegistry(namespace="test_ns")
    counter = reg.counter("hits", "test counter", labelnames=("kind",))
    counter.labels(kind="a").inc()
    counter.labels(kind="a").inc(2.5)
    # Verificar via collect() del registry interno.
    samples = []
    for fam in reg.registry.collect():
        for s in fam.samples:
            samples.append(s)
    # Buscar el sample correspondiente a kind=a.
    found = [s for s in samples if s.labels.get("kind") == "a" and s.name.endswith("_total")]
    assert any(s.value == pytest.approx(3.5) for s in found)


@pytest.mark.skipif(not HAS_PROMETHEUS, reason="prometheus_client not installed")
def test_metrics_registry_caches_same_metric() -> None:
    reg = MetricsRegistry()
    a = reg.counter("dup", "desc")
    b = reg.counter("dup", "desc")
    assert a is b


@pytest.mark.skipif(not HAS_PROMETHEUS, reason="prometheus_client not installed")
def test_default_metrics_expose_expected_labels() -> None:
    inference_latency_seconds.labels(mode="single").observe(0.001)
    signals_emitted_total.labels(signal="CALL", contract="CALLPUT").inc()
    train_loss.labels(stage="train", contract="CALLPUT").set(0.42)
    # Si se llegaron a observar sin levantar excepción, el wiring es correcto.


# ---------------------------------------------------------------------------
# No-op fallback (simulado mockeando HAS_PROMETHEUS)
# ---------------------------------------------------------------------------


def test_noop_fallback_metric_supports_all_ops(monkeypatch) -> None:
    """Verifica que `_NoOpMetric` no rompe en ninguna API esperada."""
    from src.observability import metrics as metrics_mod
    noop = metrics_mod._NoOpMetric("dummy")
    # Todas las ops deben ser no-op sin excepciones.
    noop.inc()
    noop.inc(5.0)
    noop.dec()
    noop.dec(2.0)
    noop.set(7.0)
    noop.observe(0.1)
    with noop.time():
        pass
    # `.labels(...)` devuelve el mismo objeto.
    assert noop.labels(stage="train") is noop


def test_metrics_registry_falls_back_when_prom_absent(monkeypatch) -> None:
    """Cuando `HAS_PROMETHEUS=False`, todas las métricas son no-ops."""
    from src.observability import metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "HAS_PROMETHEUS", False)
    reg = MetricsRegistry(namespace="fallback")
    counter = reg.counter("hits", "desc", labelnames=("kind",))
    assert isinstance(counter, metrics_mod._NoOpMetric)
    counter.labels(kind="a").inc()  # no-op


def test_start_http_server_returns_false_without_prom(monkeypatch) -> None:
    from src.observability import metrics as metrics_mod
    monkeypatch.setattr(metrics_mod, "HAS_PROMETHEUS", False)
    reg = MetricsRegistry()
    assert reg.start_http_server(port=0) is False
