"""Orquestador walk-forward para validación honesta del pipeline completo.

Particiona un dataset largo en folds rolling/expanding con purga y
embargo entre train/val/test, entrena el modelo en cada fold, calibra
sobre val, evalúa el backtester sobre test, y reporta métricas por
fold para detectar drift.

Diseño:

* **Expanding** (default): el train comienza en 0 y crece con cada
  fold; el test es siempre el siguiente bloque.
* **Rolling**: train de tamaño fijo que desliza con cada fold (más
  realista cuando el régimen cambia drásticamente).
* Purga + embargo entre train/val y entre val/test (Lopez de Prado).
* Reproducible: cada fold inicializa con un seed derivado del global.
* El orquestador toma como entrada un **factory** del modelo + loss,
  no instancias compartidas — cada fold entrena un modelo fresco.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.data.dataset import collate_window_samples
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.ensemble import SignalPolicy
from src.training.config import TrainingConfig
from src.training.losses import MultiContractLoss
from src.training.trainer import Trainer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuración del walk-forward."""

    n_folds: int = 5
    # Fracción del dataset reservada como warm-up (train mínimo inicial).
    initial_train_fraction: float = 0.4
    # Tamaño relativo de val dentro del bloque val+test que sigue al train.
    val_fraction_of_block: float = 0.5
    # Purga (en muestras) entre train|val y entre val|test. Si 0, se
    # toma ``max(horizons)`` del ``TrainingConfig`` automáticamente.
    purge: int = 0
    embargo: int = 0
    mode: str = "expanding"   # "expanding" | "rolling"
    # Tamaño del train en modo rolling (en muestras). Si 0, usa el del
    # primer fold expanding como anchor.
    rolling_window: int = 0
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if not 0.0 < self.initial_train_fraction < 1.0:
            raise ValueError("initial_train_fraction must be in (0, 1)")
        if not 0.0 < self.val_fraction_of_block < 1.0:
            raise ValueError("val_fraction_of_block must be in (0, 1)")
        if self.mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge and embargo must be >= 0")
        if self.rolling_window < 0:
            raise ValueError("rolling_window must be >= 0")


@dataclass
class FoldResult:
    """Resultado de un único fold."""

    fold_index: int
    train_range: tuple[int, int]
    val_range: tuple[int, int]
    test_range: tuple[int, int]
    train_loss: float
    val_loss: float
    metrics: BacktestMetrics
    backtest: BacktestResult = field(repr=False)


@dataclass
class WalkForwardResult:
    """Resultado agregado del orquestador."""

    folds: list[FoldResult]

    def aggregate_metrics(self) -> BacktestMetrics:
        """Concatena los returns de todos los folds y recalcula."""
        all_returns = np.concatenate(
            [f.backtest.total_returns() for f in self.folds]
        ) if self.folds else np.empty(0, dtype=np.float64)
        per_contract_lists: dict[str, list[np.ndarray]] = {}
        for f in self.folds:
            for contract, arr in f.backtest.returns_by_contract().items():
                per_contract_lists.setdefault(contract, []).append(arr)
        per_contract_concat = {
            k: np.concatenate(v) for k, v in per_contract_lists.items()
        }
        return compute_metrics(
            all_returns, per_contract_returns=per_contract_concat
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class WalkForwardOrchestrator:
    """Corre walk-forward sobre un dataset.

    Parameters
    ----------
    dataset:
        Dataset cronológicamente ordenado.
    model_factory:
        Callable ``() -> nn.Module``. Cada fold crea su propio modelo.
        Recomendado pasar una lambda que cierre sobre el config.
    base_config:
        ``TrainingConfig`` que el ``Trainer`` consume; cada fold puede
        usar `epochs` distinto pasando ``override_epochs``.
    contracts / horizons:
        Para el ``MultiContractLoss``, el calibrador y el ``BacktestEngine``.
    walk_forward_cfg:
        ``WalkForwardConfig``.
    backtest_cfg:
        Económico (payout, commission, base_stake).
    signal_policy:
        Umbrales/sizing.
    """

    def __init__(
        self,
        dataset: Dataset,
        model_factory: Callable[[], torch.nn.Module],
        base_config: TrainingConfig,
        contracts: Sequence[str],
        horizons: Sequence[int],
        *,
        walk_forward_cfg: Optional[WalkForwardConfig] = None,
        backtest_cfg: Optional[BacktestConfig] = None,
        signal_policy: Optional[SignalPolicy] = None,
        forward_fn: Optional[Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor]] = None,
    ) -> None:
        self.dataset = dataset
        self.model_factory = model_factory
        self.base_config = base_config
        self.contracts = tuple(contracts)
        self.horizons = tuple(int(h) for h in horizons)
        self.cfg = walk_forward_cfg or WalkForwardConfig()
        self.backtest_cfg = backtest_cfg or BacktestConfig()
        self.signal_policy = signal_policy or SignalPolicy()
        self.forward_fn = forward_fn

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> WalkForwardResult:
        folds = list(self._iter_folds())
        if not folds:
            return WalkForwardResult(folds=[])
        results: list[FoldResult] = []
        for fold_idx, ranges in enumerate(folds):
            train_range, val_range, test_range = ranges
            log.info(
                "fold %d: train=%s val=%s test=%s",
                fold_idx, train_range, val_range, test_range,
            )
            res = self._run_fold(fold_idx, train_range, val_range, test_range)
            results.append(res)
        return WalkForwardResult(folds=results)

    # ------------------------------------------------------------------
    # Particionamiento
    # ------------------------------------------------------------------

    def _effective_purge(self) -> int:
        if self.cfg.purge > 0:
            return int(self.cfg.purge)
        return int(max(self.horizons))

    def _iter_folds(self) -> Iterator[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        n = len(self.dataset)  # type: ignore[arg-type]
        purge = self._effective_purge()
        embargo = int(self.cfg.embargo)
        margin = purge + embargo

        initial_train = int(n * self.cfg.initial_train_fraction)
        # Lo que queda después del primer train se reparte en n_folds bloques iguales.
        remaining = n - initial_train
        if remaining < 2 * self.cfg.n_folds:
            return  # dataset demasiado chico
        block = remaining // self.cfg.n_folds
        if block < 4:
            return

        rolling = self.cfg.rolling_window if self.cfg.rolling_window > 0 else initial_train

        for fi in range(self.cfg.n_folds):
            block_start = initial_train + fi * block
            block_end = block_start + block
            val_size = max(1, int(block * self.cfg.val_fraction_of_block))
            val_start = block_start
            val_end = val_start + val_size
            test_start = val_end + margin
            test_end = block_end
            # Train: expanding (0..val_start-margin) o rolling (rolling_window).
            train_end = max(0, val_start - margin)
            if self.cfg.mode == "expanding":
                train_start = 0
            else:
                train_start = max(0, train_end - rolling)
            if train_end - train_start < 8:
                continue
            if val_end - val_start < 2:
                continue
            if test_end - test_start < 2:
                continue
            yield (
                (int(train_start), int(train_end)),
                (int(val_start), int(val_end)),
                (int(test_start), int(test_end)),
            )

    # ------------------------------------------------------------------
    # Un fold
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold_idx: int,
        train_range: tuple[int, int],
        val_range: tuple[int, int],
        test_range: tuple[int, int],
    ) -> FoldResult:
        seed = self.base_config.device.seed + fold_idx + self.cfg.seed_offset
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_ds = Subset(self.dataset, range(*train_range))
        val_ds = Subset(self.dataset, range(*val_range))
        test_ds = Subset(self.dataset, range(*test_range))

        model = self.model_factory()
        loss_fn = MultiContractLoss(contracts=self.contracts, horizons=self.horizons)
        forward_fn = self.forward_fn or (
            lambda m, b: m(b["features"], b["symbol_id"], b["granularity_id"])
        )
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=self.base_config,
            forward_fn=forward_fn,
            collate_fn=collate_window_samples,
        )
        state = trainer.fit()

        # Calibrate over val.
        bundle = PerContractCalibratorBundle(
            contracts=self.contracts, horizons=self.horizons,
            min_observations=10,
        )
        self._populate_bundle(trainer, val_ds, bundle)
        bundle.update_all(background=False)

        # Backtest sobre test.
        engine = BacktestEngine(
            model=trainer._inner_model,  # type: ignore[attr-defined]
            calibrator=bundle,
            contracts=self.contracts,
            horizons=self.horizons,
            policy=self.signal_policy,
            config=self.backtest_cfg,
            device=trainer.device,
        )
        bt_result = engine.run(test_ds)
        per_contract = bt_result.returns_by_contract()
        metrics = compute_metrics(
            bt_result.total_returns(),
            per_contract_returns=per_contract if per_contract else None,
        )

        train_loss = float(state.history[-1].get("train_loss", float("nan"))) if state.history else float("nan")
        val_loss = float(state.history[-1].get("val_loss", float("nan"))) if state.history else float("nan")
        return FoldResult(
            fold_index=fold_idx,
            train_range=train_range,
            val_range=val_range,
            test_range=test_range,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            backtest=bt_result,
        )

    def _populate_bundle(
        self,
        trainer: Trainer,
        val_ds: Dataset,
        bundle: PerContractCalibratorBundle,
    ) -> None:
        from torch.utils.data import DataLoader
        loader = DataLoader(
            val_ds, batch_size=self.base_config.data.batch_size,
            shuffle=False, collate_fn=collate_window_samples,
        )
        trainer.model.eval()
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(trainer.device, non_blocking=True)
                sym = batch["symbol_id"].to(trainer.device, non_blocking=True)
                gran = batch["granularity_id"].to(trainer.device, non_blocking=True)
                try:
                    logits = trainer._inner_model(features, sym, gran)  # type: ignore[attr-defined]
                except TypeError:
                    logits = trainer._inner_model(features)  # type: ignore[attr-defined]
                bundle.add_observations(
                    logits, batch["labels"], batch["label_mask"]
                )


__all__ = [
    "FoldResult",
    "WalkForwardConfig",
    "WalkForwardOrchestrator",
    "WalkForwardResult",
]
