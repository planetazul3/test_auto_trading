"""Tests de integración: composite model + multi-symbol + calibrator bundle + train CLI."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.connectors.deriv.storage import CandleRow, DuckDBStore
from src.data.dataset import (
    LabelSpec,
    MultiSymbolWindowDataset,
    WindowDataset,
    WindowDatasetConfig,
)
from src.data.store_adapter import StoreView
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import BackboneWithHeads, build_model_from_config
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.heads import HeadConfig
from src.training.config import (
    ModelConfig,
)


# ---------------------------------------------------------------------------
# Fixtures: populated DuckDB with multiple symbols
# ---------------------------------------------------------------------------


def _synthetic_candles(symbol_seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_seed)
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame(
        {
            "epoch": epochs,
            "open": base + rng.standard_normal(n) * 0.1,
            "high": base + np.abs(rng.standard_normal(n)) * 0.2,
            "low": base - np.abs(rng.standard_normal(n)) * 0.2,
            "close": base + rng.standard_normal(n) * 0.1,
        }
    )


@pytest.fixture
def multi_symbol_store(tmp_path):
    store = DuckDBStore(tmp_path / "multi.db")
    for sym, seed in [("R_100", 0), ("R_50", 1)]:
        df = _synthetic_candles(seed)
        store.upsert_candles(
            [
                CandleRow(
                    symbol=sym, granularity=60, epoch=int(r.epoch),
                    open=float(r.open), high=float(r.high),
                    low=float(r.low), close=float(r.close),
                )
                for r in df.itertuples()
            ]
        )
    yield store
    store.close()


# ---------------------------------------------------------------------------
# MultiSymbolWindowDataset
# ---------------------------------------------------------------------------


def test_multi_symbol_dataset_preserves_symbol_ids(multi_symbol_store) -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg = WindowDatasetConfig(window_size=20, horizons=(1,), label_specs=(LabelSpec("CALLPUT"),))
    ds_a = WindowDataset(multi_symbol_store, StoreView("R_100", "candles", 60), cfg, emb)
    ds_b = WindowDataset(multi_symbol_store, StoreView("R_50", "candles", 60), cfg, emb)
    ms = MultiSymbolWindowDataset([ds_a, ds_b])

    # Cada sample debe llevar el symbol_id correcto.
    seen_a = ms[0].symbol_id.item()
    seen_b = ms[len(ds_a) + 1].symbol_id.item()
    assert seen_a == emb.symbol_id("R_100")
    assert seen_b == emb.symbol_id("R_50")
    assert seen_a != seen_b
    assert len(ms) == len(ds_a) + len(ds_b)
    assert ms.num_features == ds_a.num_features


def test_multi_symbol_dataset_rejects_schema_mismatch(multi_symbol_store) -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg_a = WindowDatasetConfig(window_size=20, horizons=(1,), label_specs=(LabelSpec("CALLPUT"),))
    ds_a = WindowDataset(multi_symbol_store, StoreView("R_100", "candles", 60), cfg_a, emb)

    # Importar tick fixture para construir un dataset con schema distinto.
    from src.data.features import FeatureBuilderConfig
    cfg_b = WindowDatasetConfig(
        window_size=20,
        horizons=(1,),
        label_specs=(LabelSpec("CALLPUT"),),
        feature_config=FeatureBuilderConfig(return_windows=(1,)),  # menos features
    )
    ds_b = WindowDataset(multi_symbol_store, StoreView("R_50", "candles", 60), cfg_b, emb)
    with pytest.raises(ValueError, match="num_features|feature_names"):
        MultiSymbolWindowDataset([ds_a, ds_b])


# ---------------------------------------------------------------------------
# BackboneWithHeads
# ---------------------------------------------------------------------------


def test_backbone_with_heads_forward_shape() -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=16)
    emb.register_symbol("R_100")
    emb.register_granularity(60)
    head_cfg = HeadConfig(contracts=("CALLPUT", "TOUCHNOTOUCH"), horizons=(1, 3))
    model = BackboneWithHeads(
        num_features=5, sequence_length=12, embedding=emb,
        head_config=head_cfg, embedding_dim=16, lstm_hidden=16,
        num_attention_heads=2, lstm_layers=1, dropout=0.0,
        cnn_channels=(8, 16),
    )
    x = torch.randn(3, 12, 5)
    sid = torch.tensor([emb.symbol_id("R_100")] * 3)
    gid = torch.tensor([emb.granularity_id(60)] * 3)
    out = model(x, sid, gid)
    assert out.shape == (3, 2, 2)


def test_backbone_with_heads_without_context() -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=16)
    head_cfg = HeadConfig(contracts=("CALLPUT",), horizons=(1,), use_context=False)
    model = BackboneWithHeads(
        num_features=4, sequence_length=8, embedding=emb,
        head_config=head_cfg, embedding_dim=16, lstm_hidden=16,
        num_attention_heads=2, lstm_layers=1, dropout=0.0,
        cnn_channels=(8, 16),
    )
    out = model(torch.randn(2, 8, 4))  # no symbol_id ni granularity_id
    assert out.shape == (2, 1, 1)


def test_backbone_with_heads_requires_context_when_configured() -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=16)
    head_cfg = HeadConfig(contracts=("CALLPUT",), horizons=(1,), use_context=True)
    model = BackboneWithHeads(
        num_features=4, sequence_length=8, embedding=emb,
        head_config=head_cfg, embedding_dim=16, lstm_hidden=16,
        num_attention_heads=2, lstm_layers=1, dropout=0.0,
        cnn_channels=(8, 16),
    )
    with pytest.raises(ValueError, match="symbol_id"):
        model(torch.randn(2, 8, 4))


def test_build_model_from_config() -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=16)
    cfg = ModelConfig(
        embedding_dim=16, lstm_hidden=16, num_attention_heads=2,
        cnn_channels=(8, 16), dropout=0.0,
        head=HeadConfig(contracts=("CALLPUT",), horizons=(1,)),
    )
    model = build_model_from_config(cfg, num_features=4, sequence_length=8, embedding=emb)
    assert isinstance(model, BackboneWithHeads)
    assert model.count_parameters() > 0


# ---------------------------------------------------------------------------
# PerContractCalibratorBundle
# ---------------------------------------------------------------------------


def test_calibrator_bundle_add_observations_and_calibrate() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT", "HIGHERLOWER"), horizons=(1, 3),
        window_size=2000, min_observations=20,
    )
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((500, 2, 2)).astype(np.float32)
    # Labels correlacionados con el logit → la calibración debe converger.
    labels = (logits + rng.standard_normal(logits.shape) * 0.5 > 0).astype(np.int8)
    mask = np.ones_like(labels, dtype=bool)
    bundle.add_observations(logits, labels, mask)
    updated = bundle.update_all(background=False)
    assert updated > 0

    new_logits = rng.standard_normal((4, 2, 2)).astype(np.float32)
    probs = bundle.calibrate(new_logits)
    assert probs.shape == (4, 2, 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_calibrator_bundle_state_dict_round_trip() -> None:
    a = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,), min_observations=10)
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((100, 1, 1)).astype(np.float32)
    labels = (logits > 0).astype(np.int8)
    a.add_observations(logits, labels)
    a.update_all()

    b = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    b.load_state_dict(a.state_dict())
    # Predicciones idénticas tras el load.
    test = np.array([-1.0, 0.0, 1.0]).reshape(3, 1, 1).astype(np.float32)
    np.testing.assert_allclose(a.calibrate(test), b.calibrate(test), atol=1e-8)


def test_calibrator_bundle_quality_report_keys() -> None:
    bundle = PerContractCalibratorBundle(
        contracts=("CALLPUT",), horizons=(1, 3), min_observations=10
    )
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((200, 1, 2)).astype(np.float32)
    labels = (logits > 0).astype(np.int8)
    bundle.add_observations(logits, labels)
    bundle.update_all()
    report = bundle.quality_report()
    assert "CALLPUT__h1" in report and "CALLPUT__h3" in report
    for stats in report.values():
        assert 0 <= stats["brier_score"] <= 1
        assert 0 <= stats["ece"] <= 1


def test_calibrator_bundle_rejects_bad_shape() -> None:
    bundle = PerContractCalibratorBundle(contracts=("CALLPUT",), horizons=(1,))
    with pytest.raises(ValueError):
        bundle.add_observations(np.zeros((2, 2)), np.zeros((2, 2)))  # 2-D


# ---------------------------------------------------------------------------
# train.py CLI smoke (--dry-run)
# ---------------------------------------------------------------------------


def _load_train_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    spec = importlib.util.spec_from_file_location(
        "scripts.train",
        Path(__file__).resolve().parent.parent / "scripts" / "train.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_cli_dry_run(multi_symbol_store) -> None:
    """Asegura que el CLI ensambla datasets y modelo sin entrenar."""
    db_path = multi_symbol_store.path
    # DuckDB no permite mezclar read-only y read-write sobre la misma DB.
    multi_symbol_store.close()
    train_mod = _load_train_module()

    rc = train_mod.main([
        "--db", db_path,
        "--symbol", "R_100", "--symbol", "R_50",
        "--kind", "candles", "--granularity", "60",
        "--window-size", "20", "--horizons", "1", "3",
        "--contracts", "CALLPUT", "HIGHERLOWER",
        "--epochs", "1", "--batch-size", "8",
        "--embedding-dim", "16", "--lstm-hidden", "16",
        "--num-heads", "2", "--cnn-channels", "8", "16",
        "--device-strategy", "cpu", "--dry-run",
    ])
    assert rc == 0


def test_train_cli_runs_one_epoch(multi_symbol_store, tmp_path) -> None:
    """Pasada real: 1 epoch, CPU, comprueba que se crea best.pt + bundle."""
    db_path = multi_symbol_store.path
    multi_symbol_store.close()
    train_mod = _load_train_module()

    ckpt_dir = tmp_path / "ckpts"
    rc = train_mod.main([
        "--db", db_path,
        "--symbol", "R_100",
        "--kind", "candles", "--granularity", "60",
        "--window-size", "20", "--horizons", "1",
        "--contracts", "CALLPUT",
        "--epochs", "1", "--batch-size", "8",
        "--embedding-dim", "16", "--lstm-hidden", "16",
        "--num-heads", "2", "--cnn-channels", "8", "16",
        "--device-strategy", "cpu",
        "--checkpoint-dir", str(ckpt_dir),
    ])
    assert rc == 0
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "calibrator_bundle.json").exists()
    bundle_state = json.loads((ckpt_dir / "calibrator_bundle.json").read_text())
    assert isinstance(bundle_state, dict)
