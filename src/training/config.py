"""Configuración tipada del pipeline de entrenamiento.

Todo es dataclass: serializable a JSON, hashable y reproducible. Cero
defaults mágicos en código de runtime — si algo no se especifica, se
usa el default explícito de la dataclass, documentado aquí.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional

from src.data.features import FeatureBuilderConfig
from src.models.heads import DERIV_BINARY_CONTRACTS, HeadConfig


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Configuración del backbone + cabezales."""

    embedding_dim: int = 64
    lstm_hidden: int = 64
    lstm_layers: int = 2
    num_attention_heads: int = 4
    cnn_channels: tuple[int, ...] = (64, 128)
    cnn_kernel_sizes: tuple[int, ...] = (3, 3)
    cnn_dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.1
    head: HeadConfig = field(default_factory=HeadConfig)
    use_asset_timeframe_context: bool = True
    asset_timeframe_embedding_dim: int = 32

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if self.embedding_dim % self.num_attention_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_attention_heads"
            )
        if not self.cnn_channels:
            raise ValueError("cnn_channels must be non-empty")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Configuración del dataset y splits."""

    window_size: int = 60
    stride: int = 1
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    contracts: tuple[str, ...] = DERIV_BINARY_CONTRACTS
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    purge: int = 0  # default 0; el trainer lo eleva a max(horizons) si =0
    embargo: int = 0
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    feature_builder: FeatureBuilderConfig = field(default_factory=FeatureBuilderConfig)

    def __post_init__(self) -> None:
        if self.window_size <= 1:
            raise ValueError("window_size must be > 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val + test must be < 1")

    def effective_purge(self) -> int:
        """Purge real: ``max(horizons)`` si el caller dejó 0, else literal."""
        return max(self.horizons) if self.purge == 0 else int(self.purge)


# ---------------------------------------------------------------------------
# Optimizador
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerConfig:
    """Hiperparámetros del optimizador y scheduler."""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1
    lr_scheduler: Optional[str] = None  # "cosine" | "linear" | None
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.lr_scheduler not in (None, "cosine", "linear"):
            raise ValueError("lr_scheduler must be None | 'cosine' | 'linear'")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceConfig:
    """Estrategia de hardware: auto-detect por defecto."""

    strategy: str = "auto"   # "auto" | "ddp" | "single_gpu" | "cpu"
    precision: str = "fp32"  # "fp32" | "fp16" | "bf16"
    ddp_backend: str = "nccl"  # "nccl" en GPU, "gloo" en CPU/tests
    find_unused_parameters: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.strategy not in ("auto", "ddp", "single_gpu", "cpu"):
            raise ValueError(
                "strategy must be 'auto' | 'ddp' | 'single_gpu' | 'cpu'"
            )
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be 'fp32' | 'fp16' | 'bf16'")
        if self.ddp_backend not in ("nccl", "gloo", "mpi"):
            raise ValueError("ddp_backend must be 'nccl' | 'gloo' | 'mpi'")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Configuración top-level del job de entrenamiento."""

    epochs: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    checkpoint_dir: Optional[str] = None
    checkpoint_every_n_epochs: int = 1
    early_stopping_patience: Optional[int] = None
    log_every_n_steps: int = 50

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.checkpoint_every_n_epochs <= 0:
            raise ValueError("checkpoint_every_n_epochs must be > 0")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be > 0 or None")

    def to_json(self) -> str:
        def _default(o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return str(o)
        return json.dumps(dataclasses.asdict(self), default=_default, indent=2)


__all__ = [
    "DataConfig",
    "DeviceConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
]
