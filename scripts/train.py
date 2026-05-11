"""CLI de entrenamiento end-to-end.

Junta:

* DuckDB store (ticks / candles persistidos por el ingester).
* ``WindowDataset`` por (symbol, kind, granularity).
* ``MultiSymbolWindowDataset`` si se entrena cross-asset.
* ``BackboneWithHeads`` con cabezales multi-contract / multi-horizon.
* ``Trainer`` auto-detect (CPU / single-GPU / DDP-spawn).
* ``PerContractCalibratorBundle`` calibrado con un pass over val tras
  cada epoch.

Uso (CPU, smoke):

  python scripts/train.py --db ./market.duckdb \
    --symbol R_100 --kind candles --granularity 60 \
    --epochs 2 --batch-size 32 --device-strategy cpu \
    --checkpoint-dir ./ckpts

Uso (DDP local, 2 procesos GPU):

  python scripts/train.py --db ./market.duckdb \
    --symbol R_100 --kind candles --granularity 60 \
    --epochs 10 --batch-size 128 --device-strategy ddp --world-size 2

El script es tolerante a falta de GPU: si pides ``--device-strategy
single_gpu`` sin CUDA, baja a CPU con un aviso.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from src.connectors.deriv.storage import DuckDBStore
from src.data.dataset import (
    LabelSpec,
    MultiSymbolWindowDataset,
    WindowDataset,
    WindowDatasetConfig,
    collate_window_samples,
)
from src.data.sampler import DistributedTimeSeriesSampler, purged_split
from src.data.store_adapter import StoreView
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import build_model_from_config
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.heads import HeadConfig
from src.training.config import (
    DataConfig,
    DeviceConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.training.ddp import (
    detect_device_strategy,
    init_distributed,
    is_main_process,
    shutdown_distributed,
)
from src.training.losses import MultiContractLoss
from src.training.trainer import Trainer

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=str, required=True, help="ruta al DuckDB store")
    p.add_argument(
        "--symbol", type=str, action="append", required=True,
        help="símbolo Deriv (repetible para cross-asset)",
    )
    p.add_argument(
        "--kind", type=str, choices=("ticks", "candles"), default="candles"
    )
    p.add_argument(
        "--granularity", type=int, default=60,
        help="segundos; 0 = ticks (ignorado si --kind ticks)",
    )
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument(
        "--contracts", type=str, nargs="+",
        default=["CALLPUT", "HIGHERLOWER"],
        help="lista de contratos a entrenar como cabezales",
    )

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1)

    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--cnn-channels", type=int, nargs="+", default=[64, 128])
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument(
        "--device-strategy", type=str, choices=("auto", "cpu", "single_gpu", "ddp"),
        default="auto",
    )
    p.add_argument("--precision", type=str, choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--world-size", type=int, default=1, help="usado sólo en DDP")
    p.add_argument("--ddp-backend", type=str, default="nccl")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument(
        "--dry-run", action="store_true",
        help="construye datasets/modelo y reporta sin entrenar (smoke)",
    )
    return p


# ---------------------------------------------------------------------------
# Dataset / model assembly
# ---------------------------------------------------------------------------


def build_datasets(
    args: argparse.Namespace,
    embedding: AssetTimeframeEmbedding,
    data_cfg: DataConfig,
) -> tuple[Dataset, Optional[Dataset], Optional[Dataset], list[str]]:
    """Devuelve ``(train_ds, val_ds, test_ds, feature_names)``."""
    store = DuckDBStore(args.db, read_only=True)
    label_specs = tuple(LabelSpec(c) for c in args.contracts)
    win_cfg = WindowDatasetConfig(
        window_size=data_cfg.window_size,
        stride=data_cfg.stride,
        horizons=data_cfg.horizons,
        label_specs=label_specs,
        feature_config=data_cfg.feature_builder,
    )

    per_symbol: list[WindowDataset] = []
    for sym in args.symbol:
        view = StoreView(
            symbol=sym,
            kind=args.kind,
            granularity=args.granularity if args.kind == "candles" else None,
        )
        ds = WindowDataset(store, view, win_cfg, embedding=embedding)
        log.info("dataset %s: %d windows, %d features", sym, len(ds), ds.num_features)
        per_symbol.append(ds)

    if len(per_symbol) == 1:
        full_ds: Dataset = per_symbol[0]
        feature_names = per_symbol[0].feature_names
    else:
        ms = MultiSymbolWindowDataset(per_symbol)
        full_ds = ms
        feature_names = ms.feature_names

    # Purged split a nivel de índices globales (preserva orden temporal por sub-ds).
    n = len(full_ds)
    split = purged_split(
        n,
        val_fraction=data_cfg.val_fraction,
        test_fraction=data_cfg.test_fraction,
        purge=data_cfg.effective_purge(),
        embargo=0,
    )
    log.info(
        "split sizes: train=%d val=%d test=%d (n=%d, purge=%d)",
        split.train_indices.size, split.val_indices.size, split.test_indices.size,
        n, data_cfg.effective_purge(),
    )
    train_ds = _SubsetByIndex(full_ds, split.train_indices)
    val_ds = _SubsetByIndex(full_ds, split.val_indices) if split.val_indices.size else None
    test_ds = _SubsetByIndex(full_ds, split.test_indices) if split.test_indices.size else None
    return train_ds, val_ds, test_ds, feature_names


class _SubsetByIndex(Dataset):
    """Subset Dataset preservando el ``WindowSample`` íntegro."""

    def __init__(self, base: Dataset, indices) -> None:
        self.base = base
        self.indices = list(int(i) for i in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


# ---------------------------------------------------------------------------
# Forward function for Trainer
# ---------------------------------------------------------------------------


def _forward_fn(model, batch):
    return model(batch["features"], batch["symbol_id"], batch["granularity_id"])


# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------


def _run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=args.log_level,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 1000)))
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        init_distributed(backend=args.ddp_backend)

    try:
        cfg = _assemble_config(args)
        embedding = AssetTimeframeEmbedding(
            embedding_dim=cfg.model.asset_timeframe_embedding_dim
        )
        train_ds, val_ds, _test_ds, feature_names = build_datasets(args, embedding, cfg.data)

        head_cfg = HeadConfig(
            contracts=tuple(args.contracts),
            horizons=tuple(args.horizons),
            use_context=cfg.model.use_asset_timeframe_context,
            dropout=cfg.model.dropout,
        )
        model_cfg = replace(cfg.model, head=head_cfg)
        model = build_model_from_config(
            model_cfg,
            num_features=len(feature_names),
            sequence_length=cfg.data.window_size,
            embedding=embedding,
        )
        if is_main_process():
            log.info("model: %s | %.2fM params",
                     model.extra_repr(), model.count_parameters() / 1e6)
            log.info("config:\n%s", cfg.to_json())

        loss = MultiContractLoss(
            contracts=tuple(args.contracts), horizons=tuple(args.horizons)
        )

        train_sampler = val_sampler = None
        if world_size > 1:
            train_sampler = DistributedTimeSeriesSampler(
                list(range(len(train_ds))),
                num_replicas=world_size, rank=rank, shuffle=True,
                seed=cfg.device.seed,
            )
            if val_ds is not None:
                val_sampler = DistributedTimeSeriesSampler(
                    list(range(len(val_ds))),
                    num_replicas=world_size, rank=rank, shuffle=False,
                )

        if args.dry_run:
            if is_main_process():
                log.info("--dry-run set; skipping fit")
            return

        trainer = Trainer(
            model=model,
            loss_fn=loss,
            train_dataset=train_ds,
            val_dataset=val_ds,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            config=cfg,
            forward_fn=_forward_fn,
            collate_fn=collate_window_samples,
        )
        state = trainer.fit()

        if is_main_process():
            log.info("training done. best_val_loss=%.6f epochs=%d",
                     state.best_val_loss, len(state.history))
            # Calibrar bundle sobre el val set.
            if val_ds is not None:
                bundle = _calibrate_bundle(
                    trainer, val_ds, args.contracts, args.horizons,
                    cfg, embedding, feature_names,
                )
                report = bundle.quality_report()
                log.info("calibration report:\n%s", json.dumps(report, indent=2))
                if args.checkpoint_dir:
                    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    _save_bundle(bundle, Path(args.checkpoint_dir) / "calibrator_bundle.json")
    finally:
        shutdown_distributed()


def _assemble_config(args: argparse.Namespace) -> TrainingConfig:
    head = HeadConfig(
        contracts=tuple(args.contracts),
        horizons=tuple(args.horizons),
        use_context=True,
        dropout=args.dropout,
    )
    return TrainingConfig(
        epochs=args.epochs,
        model=ModelConfig(
            embedding_dim=args.embedding_dim,
            lstm_hidden=args.lstm_hidden,
            num_attention_heads=args.num_heads,
            cnn_channels=tuple(args.cnn_channels),
            dropout=args.dropout,
            head=head,
        ),
        data=DataConfig(
            window_size=args.window_size,
            stride=args.stride,
            horizons=tuple(args.horizons),
            contracts=tuple(args.contracts),
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            batch_size=args.batch_size,
        ),
        optimizer=OptimizerConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip,
            gradient_accumulation_steps=args.grad_accum,
        ),
        device=DeviceConfig(
            strategy=args.device_strategy,
            precision=args.precision,
            ddp_backend=args.ddp_backend,
            seed=args.seed,
        ),
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
    )


def _calibrate_bundle(
    trainer: Trainer,
    val_ds: Dataset,
    contracts,
    horizons,
    cfg: TrainingConfig,
    embedding: AssetTimeframeEmbedding,
    feature_names,
) -> PerContractCalibratorBundle:
    """Pass single epoch sobre val_ds para alimentar el calibrador por celda."""
    from torch.utils.data import DataLoader  # local: evitar import en top en DDP workers
    bundle = PerContractCalibratorBundle(
        contracts=contracts, horizons=horizons, window_size=10_000
    )
    loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        collate_fn=collate_window_samples,
    )
    trainer.model.eval()
    device = trainer.device
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            sym = batch["symbol_id"].to(device, non_blocking=True)
            gran = batch["granularity_id"].to(device, non_blocking=True)
            logits = trainer._inner_model(features, sym, gran)  # type: ignore[attr-defined]
            bundle.add_observations(logits, batch["labels"], batch["label_mask"])
    bundle.update_all(background=False)
    return bundle


def _save_bundle(bundle: PerContractCalibratorBundle, path: Path) -> None:
    state = bundle.state_dict()
    serialisable = {
        k: {"x_thresholds": v["x_thresholds"].tolist(), "y_values": v["y_values"].tolist()}
        for k, v in state.items()
    }
    path.write_text(json.dumps(serialisable, indent=2))
    log.info("saved calibrator bundle → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    strategy = detect_device_strategy(args.device_strategy)
    world_size = args.world_size if strategy == "ddp" else 1

    if world_size > 1:
        # Spawn DDP workers.
        mp.spawn(_run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        _run_worker(rank=0, world_size=1, args=args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
