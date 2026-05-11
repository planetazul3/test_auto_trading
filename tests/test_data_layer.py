"""Tests para ``src/data``: store_adapter, features, labels, dataset, sampler."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.connectors.deriv.storage import CandleRow, DuckDBStore, TickRow
from src.data.dataset import (
    LabelSpec,
    WindowDataset,
    WindowDatasetConfig,
    collate_window_samples,
)
from src.data.features import (
    CandleFeatureBuilder,
    FeatureBuilderConfig,
    TickFeatureBuilder,
    build_feature_builder,
)
from src.data.labels import (
    IGNORE_LABEL,
    callput_labeler,
    digit_even_odd_labeler,
    higherlower_labeler,
    touch_notouch_labeler,
)
from src.data.sampler import DistributedTimeSeriesSampler, purged_split
from src.data.store_adapter import StoreView, list_available_views, load_view
from src.models.conditioning import AssetTimeframeEmbedding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_candles_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    epochs = (np.arange(n, dtype=np.int64) * 60) + 1_700_000_000
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    high = base + np.abs(rng.standard_normal(n)) * 0.2
    low = base - np.abs(rng.standard_normal(n)) * 0.2
    open_ = base + rng.standard_normal(n) * 0.1
    close = base + rng.standard_normal(n) * 0.1
    return pd.DataFrame(
        {"epoch": epochs, "open": open_, "high": high, "low": low, "close": close}
    )


@pytest.fixture
def synthetic_ticks_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 250
    epochs = (np.arange(n, dtype=np.int64)) + 1_700_000_000
    quote = 1.234 + np.cumsum(rng.standard_normal(n) * 0.001)
    spread = 0.0002 + np.abs(rng.standard_normal(n)) * 0.00005
    bid = quote - spread / 2.0
    ask = quote + spread / 2.0
    return pd.DataFrame(
        {
            "epoch": epochs,
            "quote": quote,
            "bid": bid,
            "ask": ask,
            "pip_size": 1e-4,
            "tick_id": [f"t_{i}" for i in range(n)],
        }
    )


@pytest.fixture
def populated_store(synthetic_candles_df, synthetic_ticks_df, tmp_path):
    store = DuckDBStore(tmp_path / "test.db")
    # Tick batches.
    tick_records = [
        TickRow(
            symbol="R_100",
            epoch=int(row.epoch),
            quote=float(row.quote),
            bid=float(row.bid),
            ask=float(row.ask),
            pip_size=float(row.pip_size),
            tick_id=row.tick_id,
        )
        for row in synthetic_ticks_df.itertuples()
    ]
    store.upsert_ticks(tick_records)
    # Candles.
    candle_records = [
        CandleRow(
            symbol="R_100",
            granularity=60,
            epoch=int(row.epoch),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
        )
        for row in synthetic_candles_df.itertuples()
    ]
    store.upsert_candles(candle_records)
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_candle_feature_builder_outputs_finite(synthetic_candles_df) -> None:
    fb = CandleFeatureBuilder()
    feats = fb.fit_transform(synthetic_candles_df)
    assert feats.shape[0] == len(synthetic_candles_df)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    assert fb.num_features == feats.shape[1]
    assert len(fb.feature_names) == feats.shape[1]


def test_tick_feature_builder_includes_spread(synthetic_ticks_df) -> None:
    fb = TickFeatureBuilder()
    feats = fb.fit_transform(synthetic_ticks_df)
    assert "spread_norm" in fb.feature_names
    assert np.isfinite(feats).all()


def test_build_feature_builder_routes_correctly() -> None:
    assert isinstance(build_feature_builder("candles"), CandleFeatureBuilder)
    assert isinstance(build_feature_builder("ticks"), TickFeatureBuilder)
    with pytest.raises(ValueError):
        build_feature_builder("orderbook")  # type: ignore[arg-type]


def test_candle_features_are_causal(synthetic_candles_df) -> None:
    fb1 = CandleFeatureBuilder()
    feats_full = fb1.fit_transform(synthetic_candles_df)
    pivot = 150
    fb2 = CandleFeatureBuilder()
    truncated = synthetic_candles_df.iloc[: pivot + 1].copy()
    # Perturbar el final del truncado no debe alterar lo computado al inicio.
    feats_trunc = fb2.fit_transform(truncated)
    # Comparar sólo donde la ventana ya estaba "calentada".
    warmup = max(fb1.config.return_windows) + fb1.config.zscore_window
    np.testing.assert_allclose(
        feats_full[warmup:pivot, : feats_trunc.shape[1]],
        feats_trunc[warmup:pivot],
        rtol=1e-5, atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_callput_labeler_basic() -> None:
    prices = np.array([1.0, 1.1, 1.2, 1.15, 1.05, 0.95], dtype=np.float64)
    out = callput_labeler(prices, horizons=[1, 2])
    # h=1: diff = [0.1, 0.1, -0.05, -0.1, -0.1, _]; labels => 1,1,0,0,0,IGN
    assert out[1].tolist() == [1, 1, 0, 0, 0, IGNORE_LABEL]


def test_callput_labeler_epsilon_dead_band() -> None:
    prices = np.array([1.0, 1.0001, 0.9999, 1.0], dtype=np.float64)
    out = callput_labeler(prices, [1], epsilon=0.001)
    # Todos los movimientos están dentro del dead-band → IGNORE_LABEL.
    assert (out[1][:3] == IGNORE_LABEL).all()


def test_higherlower_with_barrier() -> None:
    prices = np.array([100.0, 100.5, 99.0, 101.0], dtype=np.float64)
    out = higherlower_labeler(prices, [1], barrier_pct=0.005)
    # 100→100.5 (+0.5%): HIGHER barrier=100.5 → false (no estricto > 100.5)
    # 100.5→99.0 (-1.5%): LOWER → 0.
    assert out[1][1] == 0


def test_touch_notouch_up() -> None:
    prices = np.array([100.0, 100.1, 100.5, 101.5, 99.5], dtype=np.float64)
    out = touch_notouch_labeler(prices, [3], barrier_pct=0.01, direction="up")
    # anchor=100, barrier=101.0; futuro=[100.1, 100.5, 101.5] → toca.
    assert out[3][0] == 1


def test_digit_even_odd_labeler_handles_pip_scale() -> None:
    prices = np.array([1.23, 1.24, 1.25, 1.26], dtype=np.float64)
    out = digit_even_odd_labeler(prices, [1], pip_scale=100.0)
    # Último dígito (entero tras *100): 23→ODD(0), 24→EVEN(1), 25→ODD(0).
    assert out[1][:3].tolist() == [1, 0, 1]


# ---------------------------------------------------------------------------
# Store adapter
# ---------------------------------------------------------------------------


def test_store_adapter_loads_candles_and_ticks(populated_store) -> None:
    candles = load_view(populated_store, StoreView("R_100", "candles", 60))
    ticks = load_view(populated_store, StoreView("R_100", "ticks"))
    assert len(candles) > 0
    assert len(ticks) > 0
    # Orden cronológico estricto.
    assert candles["epoch"].is_monotonic_increasing
    assert ticks["epoch"].is_monotonic_increasing


def test_list_available_views_inventories_store(populated_store) -> None:
    views = list_available_views(populated_store)
    kinds = {v.kind for v in views}
    assert {"ticks", "candles"}.issubset(kinds)


def test_store_view_validates_kind_and_granularity() -> None:
    with pytest.raises(ValueError):
        StoreView("R_100", "orderbook")
    with pytest.raises(ValueError):
        StoreView("R_100", "candles")  # falta granularity
    with pytest.raises(ValueError):
        StoreView("R_100", "ticks", granularity=60)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_window_dataset_candles_endtoend(populated_store) -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg = WindowDatasetConfig(
        window_size=20, stride=2, horizons=(1, 3),
        label_specs=(LabelSpec("CALLPUT"), LabelSpec("HIGHERLOWER", kwargs={"barrier_pct": 0.001})),
    )
    ds = WindowDataset(
        populated_store,
        StoreView("R_100", "candles", 60),
        cfg,
        embedding=emb,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample.features.shape == (20, ds.num_features)
    assert sample.labels.shape == (2, 2)
    assert sample.label_mask.shape == (2, 2)
    assert sample.symbol_id.dtype == torch_long_dtype()
    batch = collate_window_samples([ds[i] for i in range(min(4, len(ds)))])
    assert batch["features"].shape[0] == min(4, len(ds))


def test_window_dataset_ticks(populated_store) -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    cfg = WindowDatasetConfig(
        window_size=10, stride=1, horizons=(1,),
        label_specs=(LabelSpec("CALLPUT"),),
    )
    ds = WindowDataset(
        populated_store,
        StoreView("R_100", "ticks"),
        cfg,
        embedding=emb,
    )
    assert len(ds) > 0
    s = ds[0]
    assert s.features.shape == (10, ds.num_features)


def torch_long_dtype():
    import torch
    return torch.long


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def test_purged_split_no_overlap_and_margin() -> None:
    split = purged_split(1000, val_fraction=0.15, test_fraction=0.15, purge=10, embargo=5)
    train_max = split.train_indices.max() if split.train_indices.size else -1
    val_min = split.val_indices.min() if split.val_indices.size else 1000
    val_max = split.val_indices.max() if split.val_indices.size else -1
    test_min = split.test_indices.min() if split.test_indices.size else 1000
    assert train_max < val_min
    assert val_max < test_min
    assert val_min - train_max - 1 >= 15  # purge + embargo
    assert test_min - val_max - 1 >= 15


def test_distributed_time_series_sampler_partitions_contiguously() -> None:
    indices = list(range(40))
    s0 = DistributedTimeSeriesSampler(indices, num_replicas=4, rank=0)
    s1 = DistributedTimeSeriesSampler(indices, num_replicas=4, rank=1)
    s3 = DistributedTimeSeriesSampler(indices, num_replicas=4, rank=3)
    seen = list(s0) + list(s1) + list(s3)
    assert len(seen) == len(set(seen))   # sin solapamiento
    # Shards contiguos por rank, ordenados temporalmente.
    assert list(s0) == sorted(list(s0))
    assert list(s1) == sorted(list(s1))
    assert max(s0.shard_indices) < min(s1.shard_indices)
    assert max(s1.shard_indices) < min(s3.shard_indices)


def test_distributed_sampler_shuffle_within_shard_only() -> None:
    indices = list(range(40))
    s = DistributedTimeSeriesSampler(
        indices, num_replicas=2, rank=0, shuffle=True, seed=123
    )
    s.set_epoch(0)
    order_e0 = list(s)
    s.set_epoch(1)
    order_e1 = list(s)
    assert sorted(order_e0) == sorted(order_e1)
    assert order_e0 != order_e1
    # Todos los elementos pertenecen al shard del rank 0.
    assert set(order_e0).issubset(set(s.shard_indices.tolist()))
