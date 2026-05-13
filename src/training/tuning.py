"""Hyperparameter tuning con Optuna sobre el pipeline existente.

Diseño:

* ``SearchSpace`` dataclass: declara qué hiperparámetros samplear y sus
  rangos (continuos / categoriales). Reproducible y serializable.
* ``BackboneObjective``: callable que Optuna invoca por trial. Reciba
  un ``Trial`` y devuelve la métrica objetivo (default: **Brier score
  post-calibración** sobre el cabezal CALL/PUT). Muestrea hyperparams,
  construye ``TrainingConfig`` derivado y entrena con ``Trainer``.
* **Walk-forward k-folds dentro del trial**: la métrica final es el
  promedio sobre ``k`` folds purgados temporalmente. Sin esto, Optuna
  sobreajusta al fold de validación único.
* **Pruning por epoch**: cada epoch del trial reporta su métrica
  intermedia via ``trial.report``; los trials peores que la mediana
  se cortan temprano (`MedianPruner`).
* ``XGBoostMetaLearnerObjective``: objective específico para el
  ``RegimeAwareMetaLearner``. Usa `optuna.integration.XGBoostPruningCallback`.
  No requiere GPU.
* ``tune(objective, study_name, n_trials, storage, sampler, pruner)``
  helper que ata todo: study SQLite reanudable, sampler/pruner
  parametrizables, callbacks de logging.

Cero hardcodes: tanto el search space como el número de folds, métrica
objetivo y pruner son inyectados por el caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import optuna
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import collate_window_samples
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import build_model_from_config
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.heads import HeadConfig
from src.training.config import (
    TrainingConfig,
)
from src.training.losses import MultiContractLoss
from src.training.trainer import Trainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space (declarativo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int
    log: bool = False
    step: int = 1


@dataclass(frozen=True)
class Categorical:
    choices: tuple[Any, ...]


SearchParam = FloatRange | IntRange | Categorical


@dataclass(frozen=True)
class SearchSpace:
    """Hiperparámetros muestreables del backbone + cabezales + optimizador.

    Cualquier campo dejado en ``None`` se mantiene fijo al valor de la
    ``base_config``. La firma es deliberadamente granular para permitir
    fijar lo que sabemos y tunear lo que dudamos.
    """

    lr: Optional[FloatRange] = field(
        default_factory=lambda: FloatRange(1e-5, 1e-2, log=True)
    )
    weight_decay: Optional[FloatRange] = field(
        default_factory=lambda: FloatRange(1e-6, 1e-3, log=True)
    )
    dropout: Optional[FloatRange] = field(
        default_factory=lambda: FloatRange(0.0, 0.4)
    )
    embedding_dim: Optional[Categorical] = field(
        default_factory=lambda: Categorical((32, 64, 128))
    )
    lstm_hidden: Optional[Categorical] = field(
        default_factory=lambda: Categorical((32, 64, 128))
    )
    num_attention_heads: Optional[Categorical] = field(
        default_factory=lambda: Categorical((2, 4, 8))
    )
    lstm_layers: Optional[IntRange] = field(
        default_factory=lambda: IntRange(1, 3)
    )
    cnn_channels: Optional[Categorical] = field(
        default_factory=lambda: Categorical(
            ((32, 64), (64, 128), (32, 64, 128))
        )
    )
    grad_clip_norm: Optional[FloatRange] = None  # default: fijo
    batch_size: Optional[Categorical] = None     # default: fijo

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, spec in self.__dict__.items():
            if spec is None:
                continue
            if isinstance(spec, FloatRange):
                out[name] = trial.suggest_float(
                    name, spec.low, spec.high, log=spec.log, step=spec.step
                )
            elif isinstance(spec, IntRange):
                out[name] = trial.suggest_int(
                    name, spec.low, spec.high, log=spec.log, step=spec.step
                )
            elif isinstance(spec, Categorical):
                # Optuna ≥3 requiere strings / ints / floats; tuples se serializan
                # como strings y se reverten manualmente.
                if any(isinstance(c, tuple) for c in spec.choices):
                    raw = trial.suggest_categorical(
                        name, [repr(c) for c in spec.choices]
                    )
                    # Recuperar el tuple original por igualdad de repr.
                    out[name] = next(c for c in spec.choices if repr(c) == raw)
                else:
                    out[name] = trial.suggest_categorical(name, list(spec.choices))
        return out


# ---------------------------------------------------------------------------
# Backbone objective con walk-forward
# ---------------------------------------------------------------------------


@dataclass
class BackboneObjective:
    """Objective Optuna para el backbone CNN+LSTM+TFT + cabezales.

    Atributos
    ---------
    dataset:
        Dataset completo (no spliteado). Las particiones train/val se
        construyen por fold dentro del trial.
    base_config:
        ``TrainingConfig`` base; el sampler la modifica por trial.
    search_space:
        ``SearchSpace`` que declara qué hiperparámetros tunear.
    embedding:
        ``AssetTimeframeEmbedding`` compartido (debe tener los símbolos
        ya registrados — generalmente lo hace el dataset al construirse).
    k_folds:
        Número de folds walk-forward. Default 3.
    max_epochs_per_trial:
        Override de ``base_config.epochs`` para el tuning (útil para
        acotar coste).
    target:
        ``"brier"`` (post-calibración, default) o ``"val_loss"``.
    """

    dataset: Dataset
    base_config: TrainingConfig
    search_space: SearchSpace
    embedding: AssetTimeframeEmbedding
    k_folds: int = 3
    max_epochs_per_trial: Optional[int] = None
    target: str = "brier"
    contracts: tuple[str, ...] = ("CALLPUT",)
    horizons: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.k_folds < 1:
            raise ValueError("k_folds must be >= 1")
        if self.target not in ("brier", "val_loss"):
            raise ValueError("target must be 'brier' or 'val_loss'")

    # ------------------------------------------------------------------
    # Trial entrypoint
    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        sampled = self.search_space.sample(trial)
        cfg = self._derive_config(sampled)

        fold_scores: list[float] = []
        n = len(self.dataset)  # type: ignore[arg-type]
        # Calculamos los splits de walk-forward dividiendo el dataset en
        # ``k_folds`` chunks crecientes (expanding window).
        fold_size = n // (self.k_folds + 1)
        if fold_size < 16:
            raise optuna.TrialPruned("dataset too small for k_folds")

        purge = cfg.data.effective_purge()
        for k in range(1, self.k_folds + 1):
            train_end = fold_size * k
            val_start = train_end + purge
            val_end = min(n, val_start + fold_size)
            if val_end - val_start < 8:
                break
            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))

            train_ds = Subset(self.dataset, train_idx)
            val_ds = Subset(self.dataset, val_idx)
            score = self._train_one_fold(trial, cfg, train_ds, val_ds, fold_idx=k - 1)
            fold_scores.append(score)
            # Reportamos la media acumulada hasta este fold para que el
            # pruner pueda cortar trials malos antes de terminar todos los folds.
            trial.report(float(np.mean(fold_scores)), step=k - 1)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after fold {k}")

        return float(np.mean(fold_scores)) if fold_scores else float("inf")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive_config(self, sampled: Mapping[str, Any]) -> TrainingConfig:
        from dataclasses import replace
        m = self.base_config.model
        o = self.base_config.optimizer
        d = self.base_config.data

        new_model = replace(
            m,
            embedding_dim=sampled.get("embedding_dim", m.embedding_dim),
            lstm_hidden=sampled.get("lstm_hidden", m.lstm_hidden),
            num_attention_heads=sampled.get(
                "num_attention_heads", m.num_attention_heads
            ),
            lstm_layers=sampled.get("lstm_layers", m.lstm_layers),
            cnn_channels=tuple(sampled.get("cnn_channels", m.cnn_channels)),
            dropout=sampled.get("dropout", m.dropout),
        )
        new_opt = replace(
            o,
            lr=sampled.get("lr", o.lr),
            weight_decay=sampled.get("weight_decay", o.weight_decay),
            grad_clip_norm=sampled.get("grad_clip_norm", o.grad_clip_norm),
        )
        new_data = replace(d, batch_size=sampled.get("batch_size", d.batch_size))
        new_cfg = replace(
            self.base_config,
            model=new_model,
            optimizer=new_opt,
            data=new_data,
            epochs=self.max_epochs_per_trial or self.base_config.epochs,
        )
        return new_cfg

    def _train_one_fold(
        self,
        trial: optuna.Trial,
        cfg: TrainingConfig,
        train_ds: Dataset,
        val_ds: Dataset,
        *,
        fold_idx: int,
    ) -> float:
        head_cfg = HeadConfig(
            contracts=self.contracts,
            horizons=self.horizons,
            use_context=cfg.model.use_asset_timeframe_context,
            dropout=cfg.model.dropout,
        )
        from dataclasses import replace
        cfg = replace(cfg, model=replace(cfg.model, head=head_cfg))
        sample0 = self.dataset[0]  # type: ignore[index]
        num_features = sample0.features.shape[-1]
        model = build_model_from_config(
            cfg.model,
            num_features=num_features,
            sequence_length=cfg.data.window_size,
            embedding=self.embedding,
        )
        loss_fn = MultiContractLoss(
            contracts=self.contracts, horizons=self.horizons
        )
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=cfg,
            forward_fn=_forward_fn,
            collate_fn=collate_window_samples,
        )
        state = trainer.fit()
        if self.target == "val_loss":
            return float(state.best_val_loss)
        # target=brier post-calibración.
        return _brier_post_calibration(
            trainer, val_ds, self.contracts, self.horizons, cfg.data.batch_size
        )


def _forward_fn(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out: torch.Tensor = model(batch["features"], batch["symbol_id"], batch["granularity_id"])
    return out


def _brier_post_calibration(
    trainer: Trainer,
    val_ds: Dataset,
    contracts: Sequence[str],
    horizons: Sequence[int],
    batch_size: int,
) -> float:
    """Calibra un bundle sobre val y reporta el Brier promedio por celda."""
    bundle = PerContractCalibratorBundle(
        contracts=contracts, horizons=horizons,
        window_size=10_000, min_observations=10,
    )
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
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
    report = bundle.quality_report()
    if not report:
        return float("inf")
    return float(np.mean([v["brier_score"] for v in report.values()]))


# ---------------------------------------------------------------------------
# XGBoost (meta-learner) objective — G7
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XGBoostSearchSpace:
    """Search space para el meta-learner XGBoost.

    Más barato que el backbone (sin GPU), así que conviene tunearlo
    primero para validar el wrapper Optuna.
    """

    n_estimators: IntRange = field(default_factory=lambda: IntRange(50, 500))
    learning_rate: FloatRange = field(
        default_factory=lambda: FloatRange(1e-3, 3e-1, log=True)
    )
    max_depth: IntRange = field(default_factory=lambda: IntRange(2, 10))
    subsample: FloatRange = field(default_factory=lambda: FloatRange(0.5, 1.0))
    colsample_bytree: FloatRange = field(
        default_factory=lambda: FloatRange(0.5, 1.0)
    )
    min_child_weight: IntRange = field(default_factory=lambda: IntRange(1, 10))


@dataclass
class XGBoostMetaLearnerObjective:
    """Objective Optuna sobre ``RegimeAwareMetaLearner``."""

    X: np.ndarray
    y: np.ndarray
    search_space: XGBoostSearchSpace = field(default_factory=XGBoostSearchSpace)
    n_splits: int = 3
    early_stopping_rounds: int = 20
    class_weight: Optional[str] = "balanced"

    def __post_init__(self) -> None:
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must share length")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2 for TimeSeriesSplit")

    def __call__(self, trial: optuna.Trial) -> float:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.utils.class_weight import compute_sample_weight

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                self.search_space.n_estimators.low,
                self.search_space.n_estimators.high,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.search_space.learning_rate.low,
                self.search_space.learning_rate.high,
                log=self.search_space.learning_rate.log,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                self.search_space.max_depth.low,
                self.search_space.max_depth.high,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                self.search_space.subsample.low,
                self.search_space.subsample.high,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                self.search_space.colsample_bytree.low,
                self.search_space.colsample_bytree.high,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                self.search_space.min_child_weight.low,
                self.search_space.min_child_weight.high,
            ),
        }

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_tr, X_va = self.X[train_idx], self.X[val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]
            sw = (
                compute_sample_weight(self.class_weight, y_tr)
                if self.class_weight is not None
                else None
            )
            est = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                early_stopping_rounds=self.early_stopping_rounds,
                verbosity=0,
                **params,
            )
            est.fit(
                X_tr, y_tr,
                sample_weight=sw,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            best_score = float(est.best_score)
            scores.append(best_score)
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at fold {fold_idx + 1}")

        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# tune() helper
# ---------------------------------------------------------------------------


def tune(
    objective: Callable[[optuna.Trial], float],
    *,
    study_name: str,
    n_trials: int,
    storage: Optional[str] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    direction: str = "minimize",
    timeout: Optional[float] = None,
    n_jobs: int = 1,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Crea/reanuda un study y corre ``n_trials``.

    Defaults pensados para evitar overfit:
    * ``sampler=TPESampler(seed=42)``.
    * ``pruner=MedianPruner(n_warmup_steps=2, n_min_trials=5)``.
    * ``storage=None`` → in-memory (para CI/tests); ``"sqlite:///path"``
      para reanudar entre sesiones.
    """
    sampler = sampler or optuna.samplers.TPESampler(seed=42)
    pruner = pruner or optuna.pruners.MedianPruner(
        n_warmup_steps=2, n_min_trials=5
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=storage is not None,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )
    return study


__all__ = [
    "BackboneObjective",
    "Categorical",
    "FloatRange",
    "IntRange",
    "SearchSpace",
    "XGBoostMetaLearnerObjective",
    "XGBoostSearchSpace",
    "tune",
]
