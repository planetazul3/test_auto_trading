"""Trainer unificado: auto-detect CPU / single GPU / DDP.

El mismo código corre los tres modos con cambios mínimos:

* La estrategia se decide con ``detect_device_strategy()`` o se fuerza
  via ``DeviceConfig.strategy``.
* En modo ``"ddp"`` el caller debe spawnear ``world_size`` procesos
  (``torch.multiprocessing.spawn(run_worker, …)``) y cada uno construir
  su ``Trainer``. ``init_distributed()`` se llama desde el worker.
* En ``"single_gpu"`` / ``"cpu"`` el caller construye el trainer
  directamente sin process group.

Componentes core:

* AMP (autocast bf16/fp16) según ``precision``.
* Grad accumulation y grad clipping.
* Mixed-precision con ``GradScaler`` (sólo si CUDA + fp16; bf16 no
  necesita scaler).
* Checkpoints best + last con ``ckpt_dir``.
* Early stopping opcional sobre val_loss.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from src.training.config import TrainingConfig
from src.training.ddp import (
    detect_device_strategy,
    is_main_process,
)


@dataclass
class TrainState:
    """Estado mutable del entrenamiento."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = math.inf
    bad_epochs: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


class Trainer:
    """Trainer agnóstico al hardware.

    Parameters
    ----------
    model:
        Módulo ya instanciado en CPU. El trainer se encarga del move-to-device
        y del wrapping en DDP si corresponde.
    loss_fn:
        Callable ``(logits, labels, label_mask) -> (loss, per_contract_dict)``.
    train_dataset / val_dataset:
        Datasets. Si vienen con un sampler externo se pasa via ``train_sampler``.
    config:
        ``TrainingConfig`` (model/data/optim/device).
    forward_fn:
        Callable opcional que recibe ``(model, batch_dict)`` y devuelve el
        tensor de logits ``(B, C, H)``. Si es ``None`` se asume que el
        modelo expone ``forward(features, context=None)`` y se pasa
        ``batch["features"]`` directamente. Esto permite acomodar
        backbones que requieran contexto adicional.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]],
        train_dataset: Dataset,
        config: TrainingConfig,
        *,
        val_dataset: Optional[Dataset] = None,
        train_sampler: Optional[Any] = None,
        val_sampler: Optional[Any] = None,
        forward_fn: Optional[Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]] = None,
        collate_fn: Optional[Callable[[Any], dict[str, torch.Tensor]]] = None,
    ) -> None:
        torch.manual_seed(config.device.seed)
        self.config = config
        self.strategy = detect_device_strategy(config.device.strategy)
        self.device = self._pick_device()
        self.precision = config.device.precision
        self.autocast_dtype = _autocast_dtype(self.precision)
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.precision == "fp16" and self.device.type == "cuda"
            else None
        )

        model = model.to(self.device)
        if self.strategy == "ddp":
            if not dist.is_initialized():
                raise RuntimeError(
                    "DDP strategy requires init_distributed() before Trainer()"
                )
            self.model: nn.Module = DDP(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=config.device.find_unused_parameters,
            )
        else:
            self.model = model
        self._inner_model = model

        self.loss_fn = loss_fn
        self.forward_fn = forward_fn or (lambda m, batch: m(batch["features"]))
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory and self.device.type == "cuda",
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.val_loader: Optional[DataLoader] = (
            DataLoader(
                val_dataset,
                batch_size=config.data.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory and self.device.type == "cuda",
                drop_last=False,
                collate_fn=collate_fn,
            )
            if val_dataset is not None
            else None
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
        )
        self.scheduler = self._build_scheduler()

        self.state = TrainState()
        self._ckpt_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else None
        )
        if self._ckpt_dir is not None and is_main_process():
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def _pick_device(self) -> torch.device:
        if self.strategy == "cpu":
            return torch.device("cpu")
        if self.strategy == "single_gpu":
            return torch.device("cuda:0")
        # DDP: LOCAL_RANK guía el binding al GPU físico, si está disponible.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        cfg = self.config.optimizer
        if cfg.lr_scheduler is None:
            return None
        steps_per_epoch = max(1, len(self.train_loader) // cfg.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.config.epochs
        warmup = max(0, int(cfg.warmup_steps))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))
            if cfg.lr_scheduler == "linear":
                remain = max(0, total_steps - step)
                return remain / max(1, total_steps - warmup)
            # cosine
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Batch loop helpers
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _forward_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.forward_fn(self.model, batch)
        return self.loss_fn(logits, batch["labels"], batch.get("label_mask"))

    # ------------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

        cfg = self.config.optimizer
        accum = max(1, cfg.gradient_accumulation_steps)
        running_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, raw_batch in enumerate(self.train_loader):
            batch = self._move_batch(raw_batch)
            if self.autocast_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                    loss, _per_contract = self._forward_loss(batch)
            else:
                loss, _per_contract = self._forward_loss(batch)
            loss_scaled = loss / accum

            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            running_loss += float(loss.detach())
            n_batches += 1

            if (step + 1) % accum == 0:
                if cfg.grad_clip_norm is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip_norm
                    )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.state.global_step += 1

        avg = running_loss / max(1, n_batches)
        return {"train_loss": avg}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        running = 0.0
        n_batches = 0
        for raw_batch in self.val_loader:
            batch = self._move_batch(raw_batch)
            if self.autocast_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                    loss, _ = self._forward_loss(batch)
            else:
                loss, _ = self._forward_loss(batch)
            running += float(loss.detach())
            n_batches += 1
        avg = running / max(1, n_batches)
        return {"val_loss": avg}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> TrainState:
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            metrics = {**train_metrics, **val_metrics}
            self.state.history.append(metrics)

            # Si no hay validation set, usar train_loss como criterio.
            monitor = val_metrics.get("val_loss")
            if monitor is None:
                monitor = train_metrics.get("train_loss", math.inf)
            if monitor < self.state.best_val_loss:
                self.state.best_val_loss = monitor
                self.state.bad_epochs = 0
                self._save_checkpoint("best.pt")
            else:
                self.state.bad_epochs += 1

            if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0:
                self._save_checkpoint("last.pt")

            patience = self.config.early_stopping_patience
            if patience is not None and self.state.bad_epochs >= patience:
                break
        return self.state

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        if self._ckpt_dir is None or not is_main_process():
            return
        path = self._ckpt_dir / name
        torch.save(
            {
                "model": self._inner_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "state": {
                    "epoch": self.state.epoch,
                    "global_step": self.state.global_step,
                    "best_val_loss": self.state.best_val_loss,
                },
                "config": self.config.to_json(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        # weights_only=False explícito: el checkpoint incluye optimizer y
        # scheduler state, que son pickled objects. Confiamos en que el
        # checkpoint fue producido por este mismo Trainer (mismo proceso).
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self._inner_model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        st = payload.get("state", {})
        self.state.epoch = int(st.get("epoch", 0))
        self.state.global_step = int(st.get("global_step", 0))
        self.state.best_val_loss = float(st.get("best_val_loss", math.inf))


__all__ = ["Trainer", "TrainState"]
