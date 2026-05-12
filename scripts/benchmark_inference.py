"""Benchmark de latencia de inferencia (F4 del audit tracking).

Mide la latencia end-to-end del pipeline:

    features (W, F) → BackboneWithHeads → logits (1, C, H) → calibrate → policy

por iteración, sobre N samples, y reporta p50/p95/p99 + media + stdev.

Modos:

* ``--mode forward``: solo el backbone + heads (núcleo del cómputo).
* ``--mode full``: incluye el `PerContractCalibratorBundle.calibrate`
  y la `SignalPolicy` (lo que efectivamente se mide en producción).

Target documentado (CPU): **p99 < 5ms** por inferencia single-batch.

Salidas: tabla por stdout y, si ``--json``, un payload JSON con todas
las métricas para tracking de regressiones en CI.

Uso ejemplo (smoke):

  python scripts/benchmark_inference.py --mode full --iterations 200 \\
      --window-size 60 --num-features 14 \\
      --contracts CALLPUT HIGHERLOWER --horizons 1 3 5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import BackboneWithHeads
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.ensemble import SignalPolicy
from src.models.heads import HeadConfig
from src.observability.logging import configure_root

log = logging.getLogger("bench")


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkReport:
    mode: str
    device: str
    iterations: int
    window_size: int
    num_features: int
    num_contracts: int
    num_horizons: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    target_p99_ms: float
    passes_target: bool

    def render(self) -> str:
        ok = "PASS" if self.passes_target else "FAIL"
        return (
            f"benchmark mode={self.mode} device={self.device} "
            f"n={self.iterations} W={self.window_size} F={self.num_features} "
            f"C={self.num_contracts} H={self.num_horizons}\n"
            f"  p50={self.p50_ms:.3f}ms  p95={self.p95_ms:.3f}ms  "
            f"p99={self.p99_ms:.3f}ms  mean={self.mean_ms:.3f}ms\n"
            f"  stdev={self.stdev_ms:.3f}ms  min={self.min_ms:.3f}ms  "
            f"max={self.max_ms:.3f}ms\n"
            f"  target p99 < {self.target_p99_ms}ms → {ok}"
        )


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def _percentile_ms(samples_sec: list[float], q: float) -> float:
    arr = np.array(samples_sec, dtype=np.float64) * 1000.0  # ms
    return float(np.percentile(arr, q))


def run_benchmark(
    *,
    iterations: int = 200,
    warmup: int = 30,
    window_size: int = 60,
    num_features: int = 14,
    contracts: tuple[str, ...] = ("CALLPUT",),
    horizons: tuple[int, ...] = (1, 3, 5),
    mode: str = "full",
    device: Optional[torch.device] = None,
    target_p99_ms: float = 5.0,
    seed: int = 0,
) -> BenchmarkReport:
    """Ejecuta el benchmark y devuelve métricas + verdict vs target."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if mode not in ("forward", "full"):
        raise ValueError("mode must be 'forward' | 'full'")
    if target_p99_ms <= 0:
        raise ValueError("target_p99_ms must be > 0")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cpu")

    # Modelo + componentes minimal pero realista.
    embedding = AssetTimeframeEmbedding(embedding_dim=32)
    sym_id = torch.tensor([embedding.register_symbol("R_100")], device=device)
    gran_id = torch.tensor(
        [embedding.register_granularity(60)], device=device
    )
    head_cfg = HeadConfig(
        contracts=contracts, horizons=horizons,
        use_context=True, dropout=0.0,
    )
    model = BackboneWithHeads(
        num_features=num_features, sequence_length=window_size,
        embedding=embedding, head_config=head_cfg,
        embedding_dim=64, lstm_hidden=64, num_attention_heads=4,
        lstm_layers=2, dropout=0.0,
        cnn_channels=(64, 128),
    ).to(device).eval()

    # Bundle calibrador con curvas neutrales (mismo overhead que el de prod
    # pre-calibrado).
    bundle = PerContractCalibratorBundle(
        contracts=contracts, horizons=horizons,
        window_size=2000, min_observations=10,
    )
    policy = SignalPolicy()

    # Pre-allocar tensores para evitar measurement noise de allocations.
    x = torch.randn(1, window_size, num_features, device=device)

    # Warmup (incluye JIT compile/cuDNN benchmark si aplica).
    with torch.inference_mode():
        for _ in range(warmup):
            logits = model(x, sym_id, gran_id)
            if mode == "full":
                bundle.calibrate(logits)

    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(iterations):
            t0 = time.perf_counter()
            logits = model(x, sym_id, gran_id)
            if mode == "full":
                probs = bundle.calibrate(logits)
                # Aplicar policy escalar — overhead despreciable, pero
                # representativo del trabajo real.
                p_val = float(probs[0, 0, 0])
                _ = _classify(p_val, policy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - t0)

    p50 = _percentile_ms(samples, 50)
    p95 = _percentile_ms(samples, 95)
    p99 = _percentile_ms(samples, 99)
    mean_ms = float(np.mean(samples) * 1000.0)
    stdev_ms = float(np.std(samples, ddof=1) * 1000.0) if len(samples) > 1 else 0.0
    return BenchmarkReport(
        mode=mode,
        device=str(device),
        iterations=iterations,
        window_size=window_size,
        num_features=num_features,
        num_contracts=len(contracts),
        num_horizons=len(horizons),
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        mean_ms=mean_ms, stdev_ms=stdev_ms,
        min_ms=float(np.min(samples) * 1000.0),
        max_ms=float(np.max(samples) * 1000.0),
        target_p99_ms=target_p99_ms,
        passes_target=p99 < target_p99_ms,
    )


def _classify(p: float, policy: SignalPolicy) -> tuple[str, float]:
    if p >= policy.call_threshold:
        sizing = policy.strong_sizing if p >= policy.strong_call_threshold else policy.normal_sizing
        return "CALL", sizing
    if p <= policy.put_threshold:
        sizing = policy.strong_sizing if p <= policy.strong_put_threshold else policy.normal_sizing
        return "PUT", sizing
    return "NO_TRADE", policy.no_trade_sizing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--num-features", type=int, default=14)
    p.add_argument(
        "--contracts", type=str, nargs="+", default=["CALLPUT"],
    )
    p.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 3, 5],
    )
    p.add_argument(
        "--mode", choices=("forward", "full"), default="full",
    )
    p.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="cpu",
        help="cpu por default (matches target docs). 'auto' usa CUDA si está.",
    )
    p.add_argument("--target-p99-ms", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p.add_argument(
        "--fail-on-regression", action="store_true",
        help="exit code 1 si p99 >= target (útil en CI)",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def _pick_device(arg: str) -> torch.device:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        return torch.device("cuda:0")
    if arg == "auto":
        return (
            torch.device("cuda:0") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    return torch.device("cpu")


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    configure_root(level=getattr(logging, args.log_level), json_format=False)
    device = _pick_device(args.device)

    report = run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        window_size=args.window_size,
        num_features=args.num_features,
        contracts=tuple(args.contracts),
        horizons=tuple(args.horizons),
        mode=args.mode,
        device=device,
        target_p99_ms=args.target_p99_ms,
        seed=args.seed,
    )

    if args.json:
        sys.stdout.write(json.dumps(asdict(report), indent=2) + "\n")
    else:
        sys.stdout.write(report.render() + "\n")

    if args.fail_on_regression and not report.passes_target:
        log.error("p99=%.3fms exceeds target=%.1fms", report.p99_ms, report.target_p99_ms)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
