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
from .tuning import (
    BackboneObjective,
    Categorical,
    FloatRange,
    IntRange,
    SearchSpace,
    XGBoostMetaLearnerObjective,
    XGBoostSearchSpace,
    tune,
)

__all__ = [
    "BackboneObjective",
    "Categorical",
    "DataConfig",
    "DeviceConfig",
    "FloatRange",
    "IntRange",
    "ModelConfig",
    "MultiContractLoss",
    "OptimizerConfig",
    "SearchSpace",
    "TrainState",
    "Trainer",
    "TrainingConfig",
    "XGBoostMetaLearnerObjective",
    "XGBoostSearchSpace",
    "detect_device_strategy",
    "init_distributed",
    "is_main_process",
    "shutdown_distributed",
    "tune",
    "world_size_and_rank",
]
