"""CLI de hyperparameter tuning con Optuna.

Dos modos:

* ``--target backbone``: tunea el backbone CNN+LSTM+TFT + cabezales
  multi-contract con ``BackboneObjective``. Walk-forward k-folds
  dentro del trial; métrica = Brier post-calibración (default) o
  val_loss cruda. Usa los mismos argumentos de dataset que
  ``scripts/train.py``.
* ``--target xgboost``: tunea el ``RegimeAwareMetaLearner`` con
  ``XGBoostMetaLearnerObjective``. Requiere ``--xgb-X`` y ``--xgb-y``
  apuntando a archivos ``.npz``.

Storage SQLite por defecto: ``./optuna_studies/<study_name>.db``. El
study se reanuda automáticamente si ya existe.

Uso ejemplo (backbone, smoke 3 trials):

  python scripts/tune.py --target backbone \\
    --db ./market.duckdb --symbol R_100 --granularity 60 \\
    --window-size 30 --horizons 1 3 \\
    --study-name backbone_R_100 --n-trials 3 \\
    --max-epochs-per-trial 2 --k-folds 2

Uso ejemplo (XGBoost):

  python scripts/tune.py --target xgboost \\
    --xgb-X ./X_train.npz --xgb-y ./y_train.npz \\
    --study-name meta_xgb --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import optuna

from src.training.tuning import (
    BackboneObjective,
    SearchSpace,
    XGBoostMetaLearnerObjective,
    XGBoostSearchSpace,
    tune,
)

log = logging.getLogger("tune")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--target", choices=("backbone", "xgboost"), required=True,
        help="qué tunear",
    )
    p.add_argument("--study-name", type=str, required=True)
    p.add_argument(
        "--storage", type=str, default=None,
        help="optuna storage URL (default: sqlite local ./optuna_studies/<name>.db)",
    )
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO")

    # Backbone-specific.
    p.add_argument("--db", type=str, default=None, help="DuckDB store (backbone)")
    p.add_argument("--symbol", action="append", default=None)
    p.add_argument("--kind", type=str, default="candles", choices=("ticks", "candles"))
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3])
    p.add_argument("--contracts", type=str, nargs="+", default=["CALLPUT"])
    p.add_argument("--k-folds", type=int, default=3)
    p.add_argument("--max-epochs-per-trial", type=int, default=3)
    p.add_argument(
        "--objective-metric", type=str, choices=("brier", "val_loss"),
        default="brier",
    )

    # XGBoost-specific.
    p.add_argument("--xgb-X", type=str, default=None, help="ruta a .npy/.npz con X")
    p.add_argument("--xgb-y", type=str, default=None, help="ruta a .npy/.npz con y")
    p.add_argument("--xgb-n-splits", type=int, default=3)
    return p


def _default_storage(study_name: str) -> str:
    Path("optuna_studies").mkdir(parents=True, exist_ok=True)
    return f"sqlite:///optuna_studies/{study_name}.db"


# ---------------------------------------------------------------------------
# Backbone driver
# ---------------------------------------------------------------------------


def _run_backbone_tuning(args: argparse.Namespace, storage: Optional[str]) -> optuna.Study:
    if not args.db or not args.symbol:
        raise SystemExit("--db and at least one --symbol are required for backbone target")

    # Importes locales para no cargar duckdb/torch al usar --target xgboost.
    from src.connectors.deriv.storage import DuckDBStore
    from src.data.dataset import (
        LabelSpec,
        MultiSymbolWindowDataset,
        WindowDataset,
        WindowDatasetConfig,
    )
    from src.data.store_adapter import StoreView
    from src.models.conditioning import AssetTimeframeEmbedding
    from src.training.config import (
        DataConfig,
        DeviceConfig,
        ModelConfig,
        OptimizerConfig,
        TrainingConfig,
    )

    store = DuckDBStore(args.db, read_only=True)
    embedding = AssetTimeframeEmbedding(embedding_dim=32)

    label_specs = tuple(LabelSpec(c) for c in args.contracts)
    win_cfg = WindowDatasetConfig(
        window_size=args.window_size,
        horizons=tuple(args.horizons),
        label_specs=label_specs,
    )
    per_symbol = []
    for sym in args.symbol:
        view = StoreView(
            symbol=sym,
            kind=args.kind,
            granularity=args.granularity if args.kind == "candles" else None,
        )
        per_symbol.append(WindowDataset(store, view, win_cfg, embedding))
    dataset = (
        per_symbol[0] if len(per_symbol) == 1
        else MultiSymbolWindowDataset(per_symbol)
    )

    base_cfg = TrainingConfig(
        epochs=args.max_epochs_per_trial,
        model=ModelConfig(),
        data=DataConfig(
            window_size=args.window_size,
            horizons=tuple(args.horizons),
            contracts=tuple(args.contracts),
        ),
        optimizer=OptimizerConfig(),
        device=DeviceConfig(strategy="cpu", seed=args.seed),
    )

    objective = BackboneObjective(
        dataset=dataset,
        base_config=base_cfg,
        search_space=SearchSpace(),
        embedding=embedding,
        k_folds=args.k_folds,
        max_epochs_per_trial=args.max_epochs_per_trial,
        target=args.objective_metric,
        contracts=tuple(args.contracts),
        horizons=tuple(args.horizons),
    )

    return tune(
        objective,
        study_name=args.study_name,
        n_trials=args.n_trials,
        storage=storage,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )


# ---------------------------------------------------------------------------
# XGBoost driver
# ---------------------------------------------------------------------------


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        return np.load(p)["arr_0"]
    return np.load(p)


def _run_xgboost_tuning(args: argparse.Namespace, storage: Optional[str]) -> optuna.Study:
    if not args.xgb_X or not args.xgb_y:
        raise SystemExit("--xgb-X and --xgb-y are required for xgboost target")
    X = _load_array(args.xgb_X)
    y = _load_array(args.xgb_y).astype(int)
    objective = XGBoostMetaLearnerObjective(
        X=X, y=y,
        search_space=XGBoostSearchSpace(),
        n_splits=args.xgb_n_splits,
    )
    return tune(
        objective,
        study_name=args.study_name,
        n_trials=args.n_trials,
        storage=storage,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    storage = args.storage or _default_storage(args.study_name)

    if args.target == "backbone":
        study = _run_backbone_tuning(args, storage)
    else:
        study = _run_xgboost_tuning(args, storage)

    log.info("best trial: %s", study.best_trial.number)
    log.info("best value: %s", study.best_value)
    log.info("best params:\n%s", json.dumps(study.best_params, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
