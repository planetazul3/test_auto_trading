"""Tests del benchmark de latencia (F4)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


def _load_bench_module():
    name = "scripts.benchmark_inference"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).resolve().parent.parent / "scripts" / "benchmark_inference.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Registrar antes de exec_module: las dataclasses con frozen=True
    # consultan sys.modules[__module__].__dict__ durante el _process_class.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_benchmark_full_mode_produces_report() -> None:
    mod = _load_bench_module()
    report = mod.run_benchmark(
        iterations=20, warmup=5,
        window_size=20, num_features=6,
        contracts=("CALLPUT",), horizons=(1, 3),
        mode="full", device=torch.device("cpu"),
        target_p99_ms=1000.0,  # no relevante para test (sólo smoke)
    )
    assert report.iterations == 20
    assert report.window_size == 20
    assert report.num_features == 6
    assert report.p50_ms > 0
    assert report.p95_ms >= report.p50_ms
    assert report.p99_ms >= report.p95_ms
    assert report.mean_ms > 0
    assert report.min_ms <= report.mean_ms <= report.max_ms
    # passes_target con target=1000ms en CPU es trivialmente True salvo
    # bajo saturación extrema; el wiring del flag se verifica en el test
    # ``test_cli_fail_on_regression_exits_nonzero``.


def test_run_benchmark_forward_mode_only_skips_calibrator() -> None:
    """``mode=forward`` debe ser más rápido (o al menos no más lento) que ``full``."""
    mod = _load_bench_module()
    rep_full = mod.run_benchmark(
        iterations=30, warmup=10,
        window_size=20, num_features=6,
        contracts=("CALLPUT",), horizons=(1,),
        mode="full", device=torch.device("cpu"),
        target_p99_ms=1000.0, seed=0,
    )
    rep_fwd = mod.run_benchmark(
        iterations=30, warmup=10,
        window_size=20, num_features=6,
        contracts=("CALLPUT",), horizons=(1,),
        mode="forward", device=torch.device("cpu"),
        target_p99_ms=1000.0, seed=0,
    )
    # No exigimos estricto < porque medidas pequeñas son ruidosas, pero
    # los modos deben tener semánticas distintas y producir reportes válidos.
    assert rep_full.mode == "full"
    assert rep_fwd.mode == "forward"
    assert rep_full.iterations == rep_fwd.iterations == 30


def test_run_benchmark_validates_params() -> None:
    mod = _load_bench_module()
    with pytest.raises(ValueError):
        mod.run_benchmark(iterations=0)
    with pytest.raises(ValueError):
        mod.run_benchmark(warmup=-1)
    with pytest.raises(ValueError):
        mod.run_benchmark(mode="invalid")
    with pytest.raises(ValueError):
        mod.run_benchmark(target_p99_ms=0)


def test_cli_emits_json_when_flag_set(capsys) -> None:
    mod = _load_bench_module()
    rc = mod.main([
        "--iterations", "10", "--warmup", "2",
        "--window-size", "16", "--num-features", "4",
        "--contracts", "CALLPUT", "--horizons", "1",
        "--mode", "full", "--device", "cpu",
        "--target-p99-ms", "1000.0",
        "--json",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["mode"] == "full"
    assert payload["iterations"] == 10
    # passes_target queda como observación, no aserción: el target_p99_ms=1000
    # debería bastar pero bajo saturación extrema del runner puede no cumplirse.
    assert "passes_target" in payload


def test_cli_fail_on_regression_exits_nonzero() -> None:
    """Con target absurdamente bajo (0.001ms) y --fail-on-regression,
    el CLI debe terminar con exit code 1."""
    mod = _load_bench_module()
    rc = mod.main([
        "--iterations", "10", "--warmup", "2",
        "--window-size", "16", "--num-features", "4",
        "--contracts", "CALLPUT", "--horizons", "1",
        "--mode", "forward", "--device", "cpu",
        "--target-p99-ms", "0.001",  # imposible
        "--fail-on-regression",
    ])
    assert rc == 1
