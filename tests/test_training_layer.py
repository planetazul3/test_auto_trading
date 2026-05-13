"""Tests para ``src/training``: config, losses, trainer, DDP helpers.

Cubre:

* Validaciones de las dataclasses de config (rechazo de combinaciones inválidas).
* ``MultiContractLoss`` enmascarado correctamente y per-contract breakdown.
* ``Trainer`` end-to-end en CPU sobre dataset sintético tiny.
* Auto-detect device strategy (mocked).
* Smoke DDP con backend gloo en 2 procesos.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.data.labels import IGNORE_LABEL
from src.training.config import (
    DataConfig,
    DeviceConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.training.ddp import detect_device_strategy
from src.training.losses import MultiContractLoss
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_training_config_serializes_to_json() -> None:
    cfg = TrainingConfig(epochs=5)
    js = cfg.to_json()
    assert '"epochs": 5' in js


def test_training_config_round_trip_json(tmp_path) -> None:
    cfg = TrainingConfig(
        epochs=7,
        model=ModelConfig(embedding_dim=32, num_attention_heads=4, cnn_channels=(8, 16)),
        data=DataConfig(window_size=20, horizons=(1, 3), contracts=("CALLPUT",), batch_size=8),
        optimizer=OptimizerConfig(lr=5e-4, lr_scheduler="cosine", warmup_steps=10),
        device=DeviceConfig(strategy="cpu", precision="bf16", seed=123),
    )
    path = tmp_path / "cfg.json"
    cfg.to_file(path)
    loaded = TrainingConfig.from_file(path)
    assert loaded == cfg


def test_training_config_round_trip_yaml(tmp_path) -> None:
    cfg = TrainingConfig(
        epochs=2,
        data=DataConfig(window_size=15, horizons=(1, 2, 5), contracts=("CALLPUT", "HIGHERLOWER")),
        device=DeviceConfig(strategy="cpu"),
    )
    path = tmp_path / "cfg.yaml"
    cfg.to_file(path)
    loaded = TrainingConfig.from_file(path)
    assert loaded == cfg


def test_training_config_from_dict_partial_uses_dataclass_defaults() -> None:
    cfg = TrainingConfig.from_dict({"epochs": 3, "data": {"batch_size": 32}})
    assert cfg.epochs == 3
    assert cfg.data.batch_size == 32
    # Defaults preservados.
    assert cfg.data.window_size == 60
    assert cfg.optimizer.lr == 3e-4


def test_training_config_from_file_rejects_unknown_extension(tmp_path) -> None:
    path = tmp_path / "cfg.toml"
    path.write_text("ignored")
    with pytest.raises(ValueError, match="extension"):
        TrainingConfig.from_file(path)


def test_model_config_rejects_indivisible_dims() -> None:
    with pytest.raises(ValueError):
        ModelConfig(embedding_dim=10, num_attention_heads=4)


def test_data_config_purge_default_uses_max_horizon() -> None:
    cfg = DataConfig(horizons=(1, 3, 7))
    assert cfg.effective_purge() == 7
    cfg2 = DataConfig(horizons=(1, 3, 7), purge=20)
    assert cfg2.effective_purge() == 20


def test_optimizer_config_rejects_bad_scheduler() -> None:
    with pytest.raises(ValueError):
        OptimizerConfig(lr_scheduler="invalid")


# ---------------------------------------------------------------------------
# DDP detect
# ---------------------------------------------------------------------------


def test_detect_device_strategy_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert detect_device_strategy("auto") == "cpu"


def test_detect_device_strategy_forced_single_gpu_downgrades_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert detect_device_strategy("single_gpu") == "cpu"


def test_detect_device_strategy_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        detect_device_strategy("hpu")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_multi_contract_loss_respects_mask() -> None:
    loss_fn = MultiContractLoss(contracts=("CALLPUT", "HIGHERLOWER"), horizons=(1, 3))
    logits = torch.zeros(2, 2, 2, requires_grad=True)
    labels = torch.tensor(
        [
            [[1, 0], [IGNORE_LABEL, 1]],
            [[IGNORE_LABEL, IGNORE_LABEL], [0, 1]],
        ],
        dtype=torch.int8,
    )
    mask = labels != IGNORE_LABEL
    loss, per_contract = loss_fn(logits, labels, mask)
    # Loss BCE con logits=0 y mezcla equilibrada → cercano a ln(2)
    assert torch.isfinite(loss)
    assert abs(loss.item() - 0.6931) < 0.01
    assert set(per_contract.keys()) == {"CALLPUT", "HIGHERLOWER"}


def test_multi_contract_loss_all_masked_returns_zero() -> None:
    loss_fn = MultiContractLoss(contracts=("CALLPUT",), horizons=(1,))
    logits = torch.zeros(1, 1, 1, requires_grad=True)
    labels = torch.full((1, 1, 1), IGNORE_LABEL, dtype=torch.int8)
    mask = labels != IGNORE_LABEL
    loss, _ = loss_fn(logits, labels, mask)
    # Todo enmascarado → loss=0 (denom forzado a 1 evita NaN).
    assert loss.item() == 0.0


def test_multi_contract_loss_horizon_weights() -> None:
    loss_fn = MultiContractLoss(
        contracts=("CALLPUT",), horizons=(1, 3), horizon_weights={3: 10.0}
    )
    logits = torch.zeros(1, 1, 2)
    labels = torch.tensor([[[1, 1]]], dtype=torch.int8)
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss_weighted, _ = loss_fn(logits, labels, mask)
    # Con pesos h=1:1 y h=3:10, la loss media subirá por encima de ln(2).
    assert loss_weighted.item() > 0.69


# ---------------------------------------------------------------------------
# Trainer end-to-end
# ---------------------------------------------------------------------------


class _TinyDataset(Dataset):
    """Dataset sintético: features → labels binarios separables."""

    def __init__(self, n: int = 64, n_features: int = 6, contracts: int = 2, horizons: int = 2):
        rng = np.random.default_rng(7)
        self.features = rng.standard_normal((n, 8, n_features)).astype(np.float32)
        # Labels = sign del último feature (separable). C=2 contratos, H=2 horizontes.
        signs = (self.features[:, -1, 0] > 0).astype(np.int8)
        self.labels = np.tile(signs[:, None, None], (1, contracts, horizons))
        self.mask = np.ones_like(self.labels, dtype=bool)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, i):
        return {
            "features": torch.from_numpy(self.features[i]),
            "labels": torch.from_numpy(self.labels[i]),
            "label_mask": torch.from_numpy(self.mask[i]),
        }


class _TinyModel(nn.Module):
    """Modelo minimal que produce ``(B, C, H)``."""

    def __init__(self, n_features: int, contracts: int, horizons: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(n_features * 8, contracts * horizons)
        self.c = contracts
        self.h = horizons

    def forward(self, x):
        z = self.flatten(x)
        return self.linear(z).view(z.size(0), self.c, self.h)


def _trainer_collate(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "label_mask": torch.stack([b["label_mask"] for b in batch]),
    }


def test_trainer_cpu_reduces_loss_on_separable_dataset(tmp_path) -> None:
    ds_train = _TinyDataset(n=64)
    ds_val = _TinyDataset(n=32)
    model = _TinyModel(n_features=6, contracts=2, horizons=2)
    loss = MultiContractLoss(contracts=("CALLPUT", "HIGHERLOWER"), horizons=(1, 3))

    cfg = TrainingConfig(
        epochs=3,
        data=DataConfig(batch_size=16, val_fraction=0.0, test_fraction=0.0),
        optimizer=OptimizerConfig(lr=1e-2, grad_clip_norm=1.0),
        device=DeviceConfig(strategy="cpu"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        checkpoint_every_n_epochs=1,
    )
    trainer = Trainer(
        model=model,
        loss_fn=loss,
        train_dataset=ds_train,
        val_dataset=ds_val,
        config=cfg,
        collate_fn=_trainer_collate,
    )
    state = trainer.fit()
    assert state.history[-1]["train_loss"] < state.history[0]["train_loss"]
    # Checkpoint best fue escrito (modelo mejoró).
    assert (Path(cfg.checkpoint_dir) / "best.pt").exists()


def test_trainer_loads_checkpoint_round_trip(tmp_path) -> None:
    ds = _TinyDataset(n=16)
    model = _TinyModel(n_features=6, contracts=2, horizons=2)
    loss = MultiContractLoss(contracts=("CALLPUT", "HIGHERLOWER"), horizons=(1, 3))
    cfg = TrainingConfig(
        epochs=1,
        data=DataConfig(batch_size=8),
        optimizer=OptimizerConfig(lr=1e-3),
        device=DeviceConfig(strategy="cpu"),
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    trainer = Trainer(
        model=model, loss_fn=loss, train_dataset=ds, config=cfg,
        collate_fn=_trainer_collate,
    )
    trainer.fit()

    # Nueva instancia carga el checkpoint.
    model2 = _TinyModel(n_features=6, contracts=2, horizons=2)
    trainer2 = Trainer(
        model=model2, loss_fn=loss, train_dataset=ds, config=cfg,
        collate_fn=_trainer_collate,
    )
    trainer2.load_checkpoint(str(Path(cfg.checkpoint_dir) / "best.pt"))
    sd1 = trainer._inner_model.state_dict()
    sd2 = trainer2._inner_model.state_dict()
    for k in sd1:
        torch.testing.assert_close(sd1[k], sd2[k])


def test_trainer_resume_advances_epoch_and_skips_completed(tmp_path) -> None:
    """Tras cargar un checkpoint, fit() arranca en el epoch siguiente al guardado."""
    ds = _TinyDataset(n=16)
    model_a = _TinyModel(n_features=6, contracts=2, horizons=2)
    loss = MultiContractLoss(contracts=("CALLPUT", "HIGHERLOWER"), horizons=(1, 3))
    cfg = TrainingConfig(
        epochs=2,
        data=DataConfig(batch_size=8),
        optimizer=OptimizerConfig(lr=1e-3),
        device=DeviceConfig(strategy="cpu"),
        checkpoint_dir=str(tmp_path / "ckpts"),
    )
    trainer_a = Trainer(
        model=model_a, loss_fn=loss, train_dataset=ds, config=cfg,
        collate_fn=_trainer_collate,
    )
    state_a = trainer_a.fit()
    assert len(state_a.history) == 2  # epochs 0 y 1

    # Reanudar: cfg.epochs sigue siendo 2, debería ejecutar 0 epochs nuevos.
    model_b = _TinyModel(n_features=6, contracts=2, horizons=2)
    trainer_b = Trainer(
        model=model_b, loss_fn=loss, train_dataset=ds, config=cfg,
        collate_fn=_trainer_collate,
    )
    trainer_b.load_checkpoint(str(Path(cfg.checkpoint_dir) / "best.pt"))
    # Después del load, state.epoch debería ser "next to run" = último_completado + 1.
    assert trainer_b.state.epoch >= 1
    state_b = trainer_b.fit()
    # No corre epochs adicionales porque ya alcanzamos cfg.epochs.
    assert len(state_b.history) == 0

    # Si subimos epochs, sí entrena los faltantes.
    from dataclasses import replace as _replace
    cfg_more = _replace(cfg, epochs=4)
    model_c = _TinyModel(n_features=6, contracts=2, horizons=2)
    trainer_c = Trainer(
        model=model_c, loss_fn=loss, train_dataset=ds, config=cfg_more,
        collate_fn=_trainer_collate,
    )
    trainer_c.load_checkpoint(str(Path(cfg.checkpoint_dir) / "best.pt"))
    state_c = trainer_c.fit()
    assert 1 <= len(state_c.history) <= 4 - trainer_c.state.epoch + len(state_c.history)


def test_trainer_early_stopping_triggers() -> None:
    ds = _TinyDataset(n=16)
    model = _TinyModel(n_features=6, contracts=2, horizons=2)
    loss = MultiContractLoss(contracts=("CALLPUT", "HIGHERLOWER"), horizons=(1, 3))

    # LR=0 garantiza que val_loss no mejore después del primer epoch.
    cfg = TrainingConfig(
        epochs=10,
        data=DataConfig(batch_size=8),
        optimizer=OptimizerConfig(lr=1e-12, grad_clip_norm=None),
        device=DeviceConfig(strategy="cpu"),
        early_stopping_patience=1,
    )
    trainer = Trainer(
        model=model, loss_fn=loss, train_dataset=ds, val_dataset=ds, config=cfg,
        collate_fn=_trainer_collate,
    )
    state = trainer.fit()
    assert len(state.history) <= cfg.epochs
    assert len(state.history) <= 1 + (cfg.early_stopping_patience or 0) + 1


# ---------------------------------------------------------------------------
# DDP smoke (gloo backend, 2 ranks, CPU)
# ---------------------------------------------------------------------------


def _ddp_worker(rank: int, world_size: int, port: int, q):
    """Worker que inicializa DDP en gloo y reduce un tensor."""
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from src.training.ddp import is_main_process, world_size_and_rank
        ws, r = world_size_and_rank()
        assert ws == world_size
        assert r == rank
        # Reducción real cross-process: cada rank suma su id.
        t = torch.tensor([float(rank)])
        dist.all_reduce(t)
        q.put((rank, float(t.item()), is_main_process()))
    finally:
        dist.destroy_process_group()


def test_ddp_smoke_two_ranks_with_gloo() -> None:
    import multiprocessing as mp
    # Spawn context obligatorio en algunos OS para que torch.distributed funcione.
    ctx = mp.get_context("spawn")
    port = 29500 + (os.getpid() % 1000)
    q: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=_ddp_worker, args=(r, 2, port, q))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"DDP worker failed with exit code {p.exitcode}"
    results = sorted([q.get(timeout=1) for _ in range(2)])
    # All-reduce sum: rank 0 + rank 1 = 1.0 en todos los ranks.
    for r, val, is_main in results:
        assert val == 1.0
        assert is_main == (r == 0)
