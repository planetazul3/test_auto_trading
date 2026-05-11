"""Capa de entrenamiento: auto-detect CPU/GPU/DDP, multi-contract loss, checkpoints."""

from .config import (
    DataConfig,
    DeviceConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from .ddp import (
    detect_device_strategy,
    init_distributed,
    is_main_process,
    shutdown_distributed,
    world_size_and_rank,
)
from .losses import MultiContractLoss
from .trainer import Trainer, TrainState

__all__ = [
    "DataConfig",
    "DeviceConfig",
    "ModelConfig",
    "MultiContractLoss",
    "OptimizerConfig",
    "TrainState",
    "Trainer",
    "TrainingConfig",
    "detect_device_strategy",
    "init_distributed",
    "is_main_process",
    "shutdown_distributed",
    "world_size_and_rank",
]
