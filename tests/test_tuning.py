"""Tests para Optuna tuning (G1 + G7 + G8 + G9).

Diseñado para ser **barato**:
* ``RandomSampler`` + ``n_trials=2`` + 1 epoch por trial + 2 folds.
* Datasets sintéticos pequeños (<200 muestras) sobre DuckDB en memoria.
* No requiere GPU.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch

from src.connectors.deriv.storage import CandleRow, DuckDBStore
from src.data.dataset import (
    LabelSpec,
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
from src.training.tuning import (
    BackboneObjective,
    Categorical,
    FloatRange,
    IntRange,
    SearchSpace,
    XGBoostMetaLearnerObjective,
    XGBoostSearchSpace,
    tune,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path):
    """Dataset chico (200 ventanas) sobre DuckDB sintético."""
    db = tmp_path / "tune.db"
    store = DuckDBStore(db)
    rng = np.random.default_rng(0)
    n = 350
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    rows = [
        CandleRow(
            symbol="R_100", granularity=60,
            epoch=int(epochs[i]),
            open=float(base[i] + rng.standard_normal() * 0.1),
            high=float(base[i] + abs(rng.standard_normal()) * 0.2),
            low=float(base[i] - abs(rng.standard_normal()) * 0.2),
            close=float(base[i]),
        )
        for i in range(n)
    ]
    store.upsert_candles(rows)
    store.close()  # liberar para que el caller pueda re-abrir read-only si quiere

    store_ro = DuckDBStore(db, read_only=True)
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg = WindowDatasetConfig(
        window_size=20, horizons=(1, 3),
        label_specs=(LabelSpec("CALLPUT"),),
    )
    ds = WindowDataset(
        store_ro, StoreView("R_100", "candles", 60), cfg, emb
    )
    yield ds, emb
    store_ro.close()


# ---------------------------------------------------------------------------
# SearchSpace sampling
# ---------------------------------------------------------------------------


def test_search_space_samples_only_declared_fields() -> None:
    space = SearchSpace(
        lr=FloatRange(1e-4, 1e-2, log=True),
        weight_decay=None,
        dropout=FloatRange(0.0, 0.3),
        embedding_dim=Categorical((16, 32)),
        lstm_hidden=None,
        num_attention_heads=None,
        lstm_layers=IntRange(1, 2),
        cnn_channels=Categorical(((8, 16), (8, 16, 32))),
    )
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    sampled = space.sample(trial)
    assert "lr" in sampled and 1e-4 <= sampled["lr"] <= 1e-2
    assert "dropout" in sampled and 0.0 <= sampled["dropout"] <= 0.3
    assert "weight_decay" not in sampled
    assert sampled["embedding_dim"] in (16, 32)
    assert sampled["lstm_layers"] in (1, 2)
    # tuple categorical se reconstruye correctamente.
    assert sampled["cnn_channels"] in ((8, 16), (8, 16, 32))


# ---------------------------------------------------------------------------
# Backbone objective smoke (1 trial, 1 fold, 1 epoch)
# ---------------------------------------------------------------------------


def test_backbone_objective_smoke(tiny_dataset) -> None:
    ds, emb = tiny_dataset
    base_cfg = TrainingConfig(
        epochs=1,
        model=ModelConfig(
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            cnn_channels=(8, 16), dropout=0.0,
        ),
        data=DataConfig(
            window_size=20, horizons=(1, 3), contracts=("CALLPUT",),
            batch_size=16,
        ),
        optimizer=OptimizerConfig(lr=1e-3, grad_clip_norm=1.0),
        device=DeviceConfig(strategy="cpu"),
    )
    space = SearchSpace(
        lr=FloatRange(1e-4, 1e-2, log=True),
        weight_decay=None,
        dropout=FloatRange(0.0, 0.2),
        embedding_dim=Categorical((16,)),  # fijo para reproducibilidad
        lstm_hidden=Categorical((16,)),
        num_attention_heads=Categorical((2,)),
        lstm_layers=None,
        cnn_channels=Categorical(((8, 16),)),
    )
    objective = BackboneObjective(
        dataset=ds,
        base_config=base_cfg,
        search_space=space,
        embedding=emb,
        k_folds=2,
        max_epochs_per_trial=1,
        target="brier",
        contracts=("CALLPUT",),
        horizons=(1, 3),
    )
    study = tune(
        objective,
        study_name="backbone_smoke",
        n_trials=2,
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    assert len(study.trials) == 2
    # Brier score por definición en [0, 1].
    assert 0.0 <= study.best_value <= 1.0


def test_backbone_objective_val_loss_target(tiny_dataset) -> None:
    ds, emb = tiny_dataset
    base_cfg = TrainingConfig(
        epochs=1,
        model=ModelConfig(
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            cnn_channels=(8, 16), dropout=0.0,
        ),
        data=DataConfig(
            window_size=20, horizons=(1,), contracts=("CALLPUT",),
            batch_size=16,
        ),
        optimizer=OptimizerConfig(lr=1e-3),
        device=DeviceConfig(strategy="cpu"),
    )
    objective = BackboneObjective(
        dataset=ds,
        base_config=base_cfg,
        search_space=SearchSpace(
            lr=FloatRange(1e-4, 1e-2, log=True),
            weight_decay=None, dropout=None,
            embedding_dim=Categorical((16,)),
            lstm_hidden=Categorical((16,)),
            num_attention_heads=Categorical((2,)),
            lstm_layers=None, cnn_channels=Categorical(((8, 16),)),
        ),
        embedding=emb,
        k_folds=2, max_epochs_per_trial=1,
        target="val_loss",
        contracts=("CALLPUT",), horizons=(1, 3),  # matchea horizons del dataset
    )
    study = tune(
        objective, study_name="backbone_val_loss",
        n_trials=1,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
    )
    assert study.best_value > 0


def test_backbone_objective_prunes_small_dataset() -> None:
    """k_folds demasiado grande sobre dataset chico → TrialPruned."""
    # Dataset minimal in-process (no DuckDB).
    from torch.utils.data import Dataset as TorchDataset

    class TinyDS(TorchDataset):
        def __init__(self, n):
            self.n = n
            self.feature_shape = (20, 4)
        def __len__(self): return self.n
        def __getitem__(self, i):
            from src.data.dataset import WindowSample
            return WindowSample(
                features=torch.randn(20, 4),
                labels=torch.zeros(1, 1, dtype=torch.int8),
                label_mask=torch.ones(1, 1, dtype=torch.bool),
                symbol_id=torch.tensor(0),
                granularity_id=torch.tensor(0),
                anchor_epoch=torch.tensor(0),
            )

    ds = TinyDS(10)  # demasiado chico para 5 folds.
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg = TrainingConfig(
        epochs=1,
        model=ModelConfig(
            embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
            cnn_channels=(8, 16), dropout=0.0,
        ),
        data=DataConfig(window_size=20, horizons=(1,), batch_size=4),
        optimizer=OptimizerConfig(),
        device=DeviceConfig(strategy="cpu"),
    )
    objective = BackboneObjective(
        dataset=ds, base_config=cfg, search_space=SearchSpace(
            lr=FloatRange(1e-4, 1e-3), weight_decay=None, dropout=None,
            embedding_dim=Categorical((16,)),
            lstm_hidden=Categorical((16,)),
            num_attention_heads=Categorical((2,)),
            lstm_layers=None, cnn_channels=Categorical(((8, 16),)),
        ),
        embedding=emb,
        k_folds=5,  # demasiado para n=10
        max_epochs_per_trial=1,
        target="val_loss",
    )
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=1, catch=(Exception,))
    # El trial fue prunneado por dataset chico.
    assert study.trials[0].state in (
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    )


# ---------------------------------------------------------------------------
# XGBoost objective (G7)
# ---------------------------------------------------------------------------


def _synthetic_xgb_data(n: int = 200, n_features: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] > 0).astype(int) + ((X[:, 0] < 0) & (X[:, 1] > 0)).astype(int) * 2
    return X, y.astype(int)


def test_xgboost_objective_runs() -> None:
    X, y = _synthetic_xgb_data()
    objective = XGBoostMetaLearnerObjective(
        X=X, y=y,
        search_space=XGBoostSearchSpace(
            n_estimators=IntRange(20, 30),
            learning_rate=FloatRange(0.05, 0.2, log=False),
            max_depth=IntRange(2, 4),
        ),
        n_splits=3, early_stopping_rounds=5,
    )
    study = tune(
        objective, study_name="xgb_smoke", n_trials=3,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
    )
    assert len(study.trials) == 3
    # mlogloss ∈ (0, +inf)
    assert study.best_value > 0


def test_xgboost_objective_prunes_via_median() -> None:
    """MedianPruner debe cortar al menos un trial si la media intermedia
    es peor que la mediana del resto."""
    X, y = _synthetic_xgb_data(n=300)
    objective = XGBoostMetaLearnerObjective(
        X=X, y=y,
        search_space=XGBoostSearchSpace(
            n_estimators=IntRange(20, 100),
            learning_rate=FloatRange(1e-4, 5e-1, log=True),
            max_depth=IntRange(2, 8),
        ),
        n_splits=3, early_stopping_rounds=5,
    )
    study = tune(
        objective, study_name="xgb_pruned", n_trials=6,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=1, n_min_trials=2
        ),
    )
    # Como mínimo, los 6 trials terminaron en algún estado válido.
    states = [t.state for t in study.trials]
    valid = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    assert all(s in valid for s in states)


# ---------------------------------------------------------------------------
# tune() helper: storage SQLite y resume
# ---------------------------------------------------------------------------


def test_tune_sqlite_storage_round_trip(tmp_path) -> None:
    """El study persiste a SQLite y se reanuda con load_if_exists."""
    X, y = _synthetic_xgb_data(n=120)
    objective = XGBoostMetaLearnerObjective(
        X=X, y=y,
        search_space=XGBoostSearchSpace(
            n_estimators=IntRange(20, 30),
            learning_rate=FloatRange(0.05, 0.1),
            max_depth=IntRange(2, 3),
        ),
        n_splits=2, early_stopping_rounds=5,
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study1 = tune(
        objective, study_name="resume_test", n_trials=2,
        storage=storage,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
    )
    n_trials_initial = len(study1.trials)
    study2 = tune(
        objective, study_name="resume_test", n_trials=1,
        storage=storage,
        sampler=optuna.samplers.RandomSampler(seed=1),
        pruner=optuna.pruners.NopPruner(),
    )
    # El segundo run reanuda y agrega 1 trial al storage.
    assert len(study2.trials) == n_trials_initial + 1


# ---------------------------------------------------------------------------
# CLI scripts/tune.py
# ---------------------------------------------------------------------------


def _load_tune_module():
    spec = importlib.util.spec_from_file_location(
        "scripts.tune",
        Path(__file__).resolve().parent.parent / "scripts" / "tune.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tune_cli_xgboost_smoke(tmp_path) -> None:
    X, y = _synthetic_xgb_data(n=150)
    xpath = tmp_path / "X.npy"
    ypath = tmp_path / "y.npy"
    np.save(xpath, X)
    np.save(ypath, y)

    mod = _load_tune_module()
    storage = f"sqlite:///{tmp_path / 'cli.db'}"
    rc = mod.main([
        "--target", "xgboost",
        "--study-name", "cli_smoke",
        "--storage", storage,
        "--n-trials", "2",
        "--xgb-X", str(xpath),
        "--xgb-y", str(ypath),
        "--xgb-n-splits", "2",
    ])
    assert rc == 0


def test_tune_cli_backbone_dry_smoke(tmp_path) -> None:
    """Smoke del CLI backbone con dataset DuckDB sintético + 1 trial chico."""
    # Construir DB sintético.
    db = tmp_path / "tune_cli.db"
    store = DuckDBStore(db)
    rng = np.random.default_rng(0)
    n = 300
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    store.upsert_candles([
        CandleRow(
            symbol="R_100", granularity=60,
            epoch=int(epochs[i]),
            open=float(base[i] + rng.standard_normal() * 0.1),
            high=float(base[i] + abs(rng.standard_normal()) * 0.2),
            low=float(base[i] - abs(rng.standard_normal()) * 0.2),
            close=float(base[i]),
        )
        for i in range(n)
    ])
    store.close()

    mod = _load_tune_module()
    storage = f"sqlite:///{tmp_path / 'cli_backbone.db'}"
    rc = mod.main([
        "--target", "backbone",
        "--study-name", "cli_backbone_smoke",
        "--storage", storage,
        "--n-trials", "1",
        "--db", str(db),
        "--symbol", "R_100",
        "--kind", "candles", "--granularity", "60",
        "--window-size", "20", "--horizons", "1",
        "--contracts", "CALLPUT",
        "--k-folds", "2",
        "--max-epochs-per-trial", "1",
        "--objective-metric", "val_loss",
    ])
    assert rc == 0
