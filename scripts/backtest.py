"""CLI de backtest walk-forward.

Modos:

* ``--mode static``: carga un checkpoint + bundle pre-entrenados y los
  evalúa sobre todo el dataset (sin re-entrenar). Útil para
  diagnósticos rápidos.
* ``--mode walk-forward`` (default): orquesta entrenamiento + calibración +
  backtest por fold sobre la serie completa, reportando métricas
  agregadas y por fold.

Uso:

  python scripts/backtest.py --mode walk-forward \\
    --db ./market.duckdb --symbol R_100 --kind candles --granularity 60 \\
    --window-size 30 --horizons 1 3 --contracts CALLPUT \\
    --n-folds 4 --epochs-per-fold 3 \\
    --output ./bt_results.json

  python scripts/backtest.py --mode static \\
    --db ./market.duckdb --symbol R_100 --kind candles --granularity 60 \\
    --window-size 30 --horizons 1 3 --contracts CALLPUT \\
    --checkpoint ./ckpts/best.pt --calibrator-bundle ./ckpts/calibrator_bundle.json \\
    --output ./bt_static.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.backtest.metrics import compute_metrics
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardOrchestrator,
)
from src.connectors.deriv.storage import DuckDBStore
from src.data.dataset import (
    LabelSpec,
    MultiSymbolWindowDataset,
    WindowDataset,
    WindowDatasetConfig,
)
from src.data.store_adapter import StoreView
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import build_model_from_config
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.conformal import ConformalBundle
from src.models.ensemble import SignalPolicy
from src.models.heads import HeadConfig
from src.risk import RiskConfig, RiskManager
from src.training.config import (
    DataConfig,
    DeviceConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)

log = logging.getLogger("backtest")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode", choices=("walk-forward", "static"), default="walk-forward"
    )
    p.add_argument("--db", type=str, required=True)
    p.add_argument("--symbol", action="append", required=True)
    p.add_argument("--kind", choices=("ticks", "candles"), default="candles")
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    p.add_argument("--contracts", type=str, nargs="+", default=["CALLPUT"])

    # Modelo (compartido entre modos).
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--cnn-channels", type=int, nargs="+", default=[64, 128])
    p.add_argument("--dropout", type=float, default=0.1)

    # walk-forward
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--initial-train-fraction", type=float, default=0.4)
    p.add_argument(
        "--val-fraction-of-block", type=float, default=0.5,
        help="Dentro del bloque val+test, qué fracción es val",
    )
    p.add_argument("--epochs-per-fold", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--rolling-window", type=int, default=0,
        help="Tamaño del train en modo rolling (0 = expanding)",
    )

    # static
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--calibrator-bundle", type=str, default=None)

    # Económico
    p.add_argument("--payout-on-win", type=float, default=0.85)
    p.add_argument("--loss-on-lose", type=float, default=1.0)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--base-stake", type=float, default=1.0)

    # Signal policy (overridable)
    p.add_argument("--call-threshold", type=float, default=0.70)
    p.add_argument("--put-threshold", type=float, default=0.30)
    p.add_argument("--strong-call-threshold", type=float, default=0.80)
    p.add_argument("--strong-put-threshold", type=float, default=0.20)

    # Risk manager (A3)
    p.add_argument("--max-drawdown", type=float, default=None,
                   help="kill-switch absoluto si DD >= valor; None = sin límite")
    p.add_argument("--max-daily-loss", type=float, default=None)
    p.add_argument("--max-trades-per-day", type=int, default=None)
    p.add_argument("--max-concurrent-exposure", type=float, default=None)

    # Conformal gate (B3)
    p.add_argument("--conformal-alpha", type=float, default=None,
                   help="si se especifica, se aplica un ConformalBundle gate "
                        "(en walk-forward se calibra sobre val)")

    p.add_argument("--output", type=str, default=None, help="ruta a JSON con métricas")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO")
    return p


def _build_dataset(args: argparse.Namespace, store: DuckDBStore):
    emb = AssetTimeframeEmbedding(embedding_dim=32)
    win_cfg = WindowDatasetConfig(
        window_size=args.window_size,
        horizons=tuple(args.horizons),
        label_specs=tuple(LabelSpec(c) for c in args.contracts),
    )
    sub_datasets = []
    for sym in args.symbol:
        view = StoreView(
            symbol=sym,
            kind=args.kind,
            granularity=args.granularity if args.kind == "candles" else None,
        )
        sub_datasets.append(WindowDataset(store, view, win_cfg, emb))
    if len(sub_datasets) == 1:
        return sub_datasets[0], emb
    return MultiSymbolWindowDataset(sub_datasets), emb


def _build_policy(args: argparse.Namespace) -> SignalPolicy:
    return SignalPolicy(
        call_threshold=args.call_threshold,
        put_threshold=args.put_threshold,
        strong_call_threshold=args.strong_call_threshold,
        strong_put_threshold=args.strong_put_threshold,
    )


def _build_backtest_cfg(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        payout_on_win=args.payout_on_win,
        loss_on_lose=args.loss_on_lose,
        commission=args.commission,
        base_stake=args.base_stake,
        batch_size=args.batch_size,
    )


def _maybe_build_risk_manager(args: argparse.Namespace) -> RiskManager | None:
    if (
        args.max_drawdown is None
        and args.max_daily_loss is None
        and args.max_trades_per_day is None
        and args.max_concurrent_exposure is None
    ):
        return None
    return RiskManager(RiskConfig(
        max_drawdown=args.max_drawdown,
        max_daily_loss=args.max_daily_loss,
        max_trades_per_day=args.max_trades_per_day,
        max_concurrent_exposure=args.max_concurrent_exposure,
    ))


def _maybe_build_conformal_gate(args: argparse.Namespace) -> ConformalBundle | None:
    if args.conformal_alpha is None:
        return None
    return ConformalBundle(
        contracts=tuple(args.contracts),
        horizons=tuple(args.horizons),
        alpha=args.conformal_alpha,
        min_observations=20,
    )


def _build_model_factory(args: argparse.Namespace, embedding: AssetTimeframeEmbedding, num_features: int):
    head = HeadConfig(
        contracts=tuple(args.contracts), horizons=tuple(args.horizons),
        use_context=True, dropout=args.dropout,
    )
    cfg = ModelConfig(
        embedding_dim=args.embedding_dim,
        lstm_hidden=args.lstm_hidden,
        num_attention_heads=args.num_heads,
        cnn_channels=tuple(args.cnn_channels),
        dropout=args.dropout,
        head=head,
    )

    def _factory():
        return build_model_from_config(
            cfg,
            num_features=num_features,
            sequence_length=args.window_size,
            embedding=embedding,
        )
    return _factory


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _run_walk_forward(args: argparse.Namespace) -> dict:
    store = DuckDBStore(args.db, read_only=True)
    try:
        dataset, embedding = _build_dataset(args, store)
        n_features = (
            dataset.num_features
            if hasattr(dataset, "num_features")
            else dataset[0].features.shape[-1]
        )
        factory = _build_model_factory(args, embedding, n_features)
        train_cfg = TrainingConfig(
            epochs=args.epochs_per_fold,
            model=ModelConfig(
                embedding_dim=args.embedding_dim,
                lstm_hidden=args.lstm_hidden,
                num_attention_heads=args.num_heads,
                cnn_channels=tuple(args.cnn_channels),
                dropout=args.dropout,
            ),
            data=DataConfig(
                window_size=args.window_size,
                horizons=tuple(args.horizons),
                contracts=tuple(args.contracts),
                batch_size=args.batch_size,
            ),
            optimizer=OptimizerConfig(lr=args.lr),
            device=DeviceConfig(strategy="cpu", seed=args.seed),
        )
        wf_cfg = WalkForwardConfig(
            n_folds=args.n_folds,
            initial_train_fraction=args.initial_train_fraction,
            val_fraction_of_block=args.val_fraction_of_block,
            mode="rolling" if args.rolling_window > 0 else "expanding",
            rolling_window=args.rolling_window,
        )
        orch = WalkForwardOrchestrator(
            dataset=dataset,
            model_factory=factory,
            base_config=train_cfg,
            contracts=tuple(args.contracts),
            horizons=tuple(args.horizons),
            walk_forward_cfg=wf_cfg,
            backtest_cfg=_build_backtest_cfg(args),
            signal_policy=_build_policy(args),
        )
        result = orch.run()
    finally:
        store.close()

    aggregated = result.aggregate_metrics()
    payload = {
        "mode": "walk-forward",
        "n_folds": len(result.folds),
        "aggregated": aggregated.to_dict(),
        "per_fold": [
            {
                "fold_index": f.fold_index,
                "train_range": list(f.train_range),
                "val_range": list(f.val_range),
                "test_range": list(f.test_range),
                "train_loss": f.train_loss,
                "val_loss": f.val_loss,
                "metrics": f.metrics.to_dict(),
            }
            for f in result.folds
        ],
    }
    return payload


def _run_static(args: argparse.Namespace) -> dict:
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required in static mode")
    store = DuckDBStore(args.db, read_only=True)
    try:
        dataset, embedding = _build_dataset(args, store)
        n_features = (
            dataset.num_features
            if hasattr(dataset, "num_features")
            else dataset[0].features.shape[-1]
        )
        for sym in args.symbol:
            embedding.register_symbol(sym)
        embedding.register_granularity(
            args.granularity if args.kind == "candles" else None
        )

        factory = _build_model_factory(args, embedding, n_features)
        model = factory()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()

        bundle = PerContractCalibratorBundle(
            contracts=tuple(args.contracts), horizons=tuple(args.horizons),
        )
        if args.calibrator_bundle:
            raw = json.loads(Path(args.calibrator_bundle).read_text())
            state = {
                k: {
                    "x_thresholds": np.asarray(v["x_thresholds"], dtype=np.float64),
                    "y_values": np.asarray(v["y_values"], dtype=np.float64),
                }
                for k, v in raw.items()
            }
            bundle.load_state_dict(state)

        engine = BacktestEngine(
            model=model,
            calibrator=bundle,
            contracts=tuple(args.contracts),
            horizons=tuple(args.horizons),
            policy=_build_policy(args),
            config=_build_backtest_cfg(args),
            device=device,
            conformal_gate=_maybe_build_conformal_gate(args),
            risk_manager=_maybe_build_risk_manager(args),
        )
        result = engine.run(dataset)
    finally:
        store.close()

    per_contract = result.returns_by_contract()
    metrics = compute_metrics(
        result.total_returns(),
        per_contract_returns=per_contract if per_contract else None,
    )
    return {
        "mode": "static",
        "n_events": len(result.events),
        "metrics": metrics.to_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mode == "walk-forward":
        payload = _run_walk_forward(args)
    else:
        payload = _run_static(args)

    output_str = json.dumps(payload, indent=2, default=float)
    if args.output:
        Path(args.output).write_text(output_str)
        log.info("wrote results to %s", args.output)
    else:
        print(output_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
