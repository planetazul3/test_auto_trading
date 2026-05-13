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
    p.add_argument(
        "--config", type=str, default=None,
        help="ruta a YAML/JSON con TrainingConfig; CLI flags explícitos sobreescriben",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="reanuda desde un checkpoint .pt; restaura model/optimizer/scheduler/state",
    )

    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument(
        "--contracts", type=str, nargs="+", default=None,
        help="lista de contratos a entrenar como cabezales",
    )

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--grad-accum", type=int, default=None)

    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--test-fraction", type=float, default=None)

    p.add_argument("--embedding-dim", type=int, default=None)
    p.add_argument("--lstm-hidden", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--cnn-channels", type=int, nargs="+", default=None)
    p.add_argument("--dropout", type=float, default=None)

    p.add_argument(
        "--device-strategy", type=str, choices=("auto", "cpu", "single_gpu", "ddp"),
        default=None,
    )
    p.add_argument("--precision", type=str, choices=("fp32", "fp16", "bf16"), default=None)
    p.add_argument("--world-size", type=int, default=1, help="usado sólo en DDP")
    p.add_argument("--ddp-backend", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument(
        "--dry-run", action="store_true",
        help="construye datasets/modelo y reporta sin entrenar (smoke)",
    )
    return p


# Defaults used when neither --config nor an explicit CLI flag provides a value.
_CLI_DEFAULTS: dict = {
    "window_size": 60,
    "stride": 1,
    "horizons": [1, 3, 5, 10],
    "contracts": ["CALLPUT", "HIGHERLOWER"],
    "epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "grad_accum": 1,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "embedding_dim": 64,
    "lstm_hidden": 64,
    "num_heads": 4,
    "cnn_channels": [64, 128],
    "dropout": 0.1,
    "device_strategy": "auto",
    "precision": "fp32",
    "ddp_backend": "nccl",
    "seed": 42,
}


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
    label_specs = tuple(LabelSpec(c) for c in data_cfg.contracts)
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
            contracts=cfg.data.contracts,
            horizons=cfg.data.horizons,
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
            contracts=cfg.data.contracts, horizons=cfg.data.horizons
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
        if args.resume:
            trainer.load_checkpoint(args.resume)
            if is_main_process():
                log.info(
                    "resumed from %s (next epoch=%d, best_val_loss=%.6f)",
                    args.resume, trainer.state.epoch, trainer.state.best_val_loss,
                )
        state = trainer.fit()

        if is_main_process():
            log.info("training done. best_val_loss=%.6f epochs=%d",
                     state.best_val_loss, len(state.history))
            # Calibrar bundle sobre el val set.
            if val_ds is not None:
                bundle = _calibrate_bundle(
                    trainer, val_ds, cfg.data.contracts, cfg.data.horizons,
                    cfg, embedding, feature_names,
                )
                report = bundle.quality_report()
                log.info("calibration report:\n%s", json.dumps(report, indent=2))
                if args.checkpoint_dir:
                    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    _save_bundle(bundle, Path(args.checkpoint_dir) / "calibrator_bundle.json")
    finally:
        shutdown_distributed()


def _pick(cli_value, base_value, default):
    """Resolve a single field: explicit CLI > loaded-config > hardcoded default."""
    if cli_value is not None:
        return cli_value
    if base_value is not None:
        return base_value
    return default


def _assemble_config(args: argparse.Namespace) -> TrainingConfig:
    """Build a TrainingConfig from the layered sources.

    Precedence (highest first): explicit CLI flag → ``--config`` file →
    ``_CLI_DEFAULTS``. Fields not exposed via CLI come from the file when
    present, else the dataclass default.
    """
    base: Optional[TrainingConfig] = (
        TrainingConfig.from_file(args.config) if args.config else None
    )
    base_model = base.model if base else None
    base_data = base.data if base else None
    base_opt = base.optimizer if base else None
    base_dev = base.device if base else None

    def m(field: str, default):  # model
        return _pick(None, getattr(base_model, field, None), default) if base_model else default

    def d(field: str, default):  # data
        return _pick(None, getattr(base_data, field, None), default) if base_data else default

    def o(field: str, default):  # optimizer
        return _pick(None, getattr(base_opt, field, None), default) if base_opt else default

    def v(field: str, default):  # device
        return _pick(None, getattr(base_dev, field, None), default) if base_dev else default

    contracts = tuple(
        _pick(args.contracts, base_data.contracts if base_data else None, _CLI_DEFAULTS["contracts"])
    )
    horizons = tuple(
        _pick(args.horizons, base_data.horizons if base_data else None, _CLI_DEFAULTS["horizons"])
    )
    dropout = _pick(args.dropout, base_model.dropout if base_model else None, _CLI_DEFAULTS["dropout"])
    cnn_channels = tuple(
        _pick(args.cnn_channels, base_model.cnn_channels if base_model else None, _CLI_DEFAULTS["cnn_channels"])
    )

    head = HeadConfig(
        contracts=contracts,
        horizons=horizons,
        use_context=m("use_asset_timeframe_context", True),
        dropout=dropout,
    )

    model_cfg = ModelConfig(
        embedding_dim=_pick(args.embedding_dim, base_model.embedding_dim if base_model else None, _CLI_DEFAULTS["embedding_dim"]),
        lstm_hidden=_pick(args.lstm_hidden, base_model.lstm_hidden if base_model else None, _CLI_DEFAULTS["lstm_hidden"]),
        lstm_layers=m("lstm_layers", ModelConfig.__dataclass_fields__["lstm_layers"].default),
        num_attention_heads=_pick(args.num_heads, base_model.num_attention_heads if base_model else None, _CLI_DEFAULTS["num_heads"]),
        cnn_channels=cnn_channels,
        cnn_kernel_sizes=tuple(m("cnn_kernel_sizes", (3, 3))),
        cnn_dilations=tuple(m("cnn_dilations", (1, 2))),
        dropout=dropout,
        head=head,
        use_asset_timeframe_context=m("use_asset_timeframe_context", True),
        asset_timeframe_embedding_dim=m("asset_timeframe_embedding_dim", 32),
    )

    feature_builder = base_data.feature_builder if base_data else DataConfig.__dataclass_fields__["feature_builder"].default_factory()
    data_cfg = DataConfig(
        window_size=_pick(args.window_size, base_data.window_size if base_data else None, _CLI_DEFAULTS["window_size"]),
        stride=_pick(args.stride, base_data.stride if base_data else None, _CLI_DEFAULTS["stride"]),
        horizons=horizons,
        contracts=contracts,
        val_fraction=_pick(args.val_fraction, base_data.val_fraction if base_data else None, _CLI_DEFAULTS["val_fraction"]),
        test_fraction=_pick(args.test_fraction, base_data.test_fraction if base_data else None, _CLI_DEFAULTS["test_fraction"]),
        purge=d("purge", 0),
        embargo=d("embargo", 0),
        batch_size=_pick(args.batch_size, base_data.batch_size if base_data else None, _CLI_DEFAULTS["batch_size"]),
        num_workers=d("num_workers", 0),
        pin_memory=d("pin_memory", False),
        feature_builder=feature_builder,
    )

    optimizer_cfg = OptimizerConfig(
        lr=_pick(args.lr, base_opt.lr if base_opt else None, _CLI_DEFAULTS["lr"]),
        weight_decay=_pick(args.weight_decay, base_opt.weight_decay if base_opt else None, _CLI_DEFAULTS["weight_decay"]),
        betas=tuple(o("betas", (0.9, 0.999))),
        grad_clip_norm=_pick(args.grad_clip, base_opt.grad_clip_norm if base_opt else None, _CLI_DEFAULTS["grad_clip"]),
        gradient_accumulation_steps=_pick(args.grad_accum, base_opt.gradient_accumulation_steps if base_opt else None, _CLI_DEFAULTS["grad_accum"]),
        lr_scheduler=o("lr_scheduler", None),
        warmup_steps=o("warmup_steps", 0),
    )

    device_cfg = DeviceConfig(
        strategy=_pick(args.device_strategy, base_dev.strategy if base_dev else None, _CLI_DEFAULTS["device_strategy"]),
        precision=_pick(args.precision, base_dev.precision if base_dev else None, _CLI_DEFAULTS["precision"]),
        ddp_backend=_pick(args.ddp_backend, base_dev.ddp_backend if base_dev else None, _CLI_DEFAULTS["ddp_backend"]),
        find_unused_parameters=v("find_unused_parameters", False),
        seed=_pick(args.seed, base_dev.seed if base_dev else None, _CLI_DEFAULTS["seed"]),
    )

    return TrainingConfig(
        epochs=_pick(args.epochs, base.epochs if base else None, _CLI_DEFAULTS["epochs"]),
        model=model_cfg,
        data=data_cfg,
        optimizer=optimizer_cfg,
        device=device_cfg,
        checkpoint_dir=_pick(args.checkpoint_dir, base.checkpoint_dir if base else None, None),
        checkpoint_every_n_epochs=base.checkpoint_every_n_epochs if base else 1,
        early_stopping_patience=_pick(
            args.early_stopping_patience,
            base.early_stopping_patience if base else None,
            None,
        ),
        log_every_n_steps=base.log_every_n_steps if base else 50,
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

    # Resolver la estrategia de device antes de spawn: CLI > config > "auto".
    cfg_preview = TrainingConfig.from_file(args.config) if args.config else None
    raw_strategy = args.device_strategy or (cfg_preview.device.strategy if cfg_preview else "auto")
    strategy = detect_device_strategy(raw_strategy)
    world_size = args.world_size if strategy == "ddp" else 1

    if world_size > 1:
        # Spawn DDP workers.
        mp.spawn(_run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        _run_worker(rank=0, world_size=1, args=args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
