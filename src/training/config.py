"""Configuración tipada del pipeline de entrenamiento.

Todo es dataclass: serializable a JSON, hashable y reproducible. Cero
defaults mágicos en código de runtime — si algo no se especifica, se
usa el default explícito de la dataclass, documentado aquí.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
        def _default(o: Any) -> Any:
            if dataclasses.is_dataclass(o) and not isinstance(o, type):
                return dataclasses.asdict(o)
            return str(o)
        return json.dumps(dataclasses.asdict(self), default=_default, indent=2)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_file(self, path: str | Path) -> None:
        p = Path(path)
        payload = self.to_dict()
        if p.suffix in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]
            p.write_text(yaml.safe_dump(payload, sort_keys=False))
        elif p.suffix == ".json":
            p.write_text(json.dumps(payload, indent=2))
        else:
            raise ValueError(f"unsupported config extension: {p.suffix!r}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        return _build_training_config(payload)

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]
            payload = yaml.safe_load(text)
        elif p.suffix == ".json":
            payload = json.loads(text)
        else:
            raise ValueError(f"unsupported config extension: {p.suffix!r}")
        if not isinstance(payload, dict):
            raise ValueError("config file must parse to a mapping at top level")
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Recursive builders (handle nested dataclasses + tuple coercion from YAML/JSON)
# ---------------------------------------------------------------------------


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"expected list/tuple, got {type(value).__name__}")


def _build_head_config(payload: dict[str, Any]) -> HeadConfig:
    kwargs: dict[str, Any] = dict(payload)
    if "contracts" in kwargs:
        kwargs["contracts"] = _as_tuple(kwargs["contracts"])
    if "horizons" in kwargs:
        kwargs["horizons"] = _as_tuple(kwargs["horizons"])
    return HeadConfig(**kwargs)


def _build_model_config(payload: dict[str, Any]) -> ModelConfig:
    kwargs: dict[str, Any] = dict(payload)
    for k in ("cnn_channels", "cnn_kernel_sizes", "cnn_dilations"):
        if k in kwargs:
            kwargs[k] = _as_tuple(kwargs[k])
    if "head" in kwargs and isinstance(kwargs["head"], dict):
        kwargs["head"] = _build_head_config(kwargs["head"])
    return ModelConfig(**kwargs)


def _build_feature_builder_config(payload: dict[str, Any]) -> FeatureBuilderConfig:
    kwargs: dict[str, Any] = dict(payload)
    for k in ("return_windows", "volatility_windows", "momentum_windows"):
        if k in kwargs:
            kwargs[k] = _as_tuple(kwargs[k])
    return FeatureBuilderConfig(**kwargs)


def _build_data_config(payload: dict[str, Any]) -> DataConfig:
    kwargs: dict[str, Any] = dict(payload)
    if "horizons" in kwargs:
        kwargs["horizons"] = _as_tuple(kwargs["horizons"])
    if "contracts" in kwargs:
        kwargs["contracts"] = _as_tuple(kwargs["contracts"])
    if "feature_builder" in kwargs and isinstance(kwargs["feature_builder"], dict):
        kwargs["feature_builder"] = _build_feature_builder_config(kwargs["feature_builder"])
    return DataConfig(**kwargs)


def _build_optimizer_config(payload: dict[str, Any]) -> OptimizerConfig:
    kwargs: dict[str, Any] = dict(payload)
    if "betas" in kwargs:
        kwargs["betas"] = _as_tuple(kwargs["betas"])
    return OptimizerConfig(**kwargs)


def _build_device_config(payload: dict[str, Any]) -> DeviceConfig:
    return DeviceConfig(**payload)


def _build_training_config(payload: dict[str, Any]) -> TrainingConfig:
    kwargs: dict[str, Any] = dict(payload)
    if "model" in kwargs and isinstance(kwargs["model"], dict):
        kwargs["model"] = _build_model_config(kwargs["model"])
    if "data" in kwargs and isinstance(kwargs["data"], dict):
        kwargs["data"] = _build_data_config(kwargs["data"])
    if "optimizer" in kwargs and isinstance(kwargs["optimizer"], dict):
        kwargs["optimizer"] = _build_optimizer_config(kwargs["optimizer"])
    if "device" in kwargs and isinstance(kwargs["device"], dict):
        kwargs["device"] = _build_device_config(kwargs["device"])
    return TrainingConfig(**kwargs)


__all__ = [
    "DataConfig",
    "DeviceConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
]
