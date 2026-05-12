"""Tests para los P1+P2 fixes a los modelos.

Cubre:

* Causalidad real del backbone (perturbar futuro no altera pasado).
* Flujo de gradientes a través de todas las ramas.
* Thread-safety del calibrador (no se cuelga ni rompe la curva).
* SHAP API moderna (no falla con SHAP ≥0.42).
* Validaciones explícitas (embedding_dim%num_heads, rnn_type, etc.).
* Cabezales multi-contract / multi-horizon.
* AssetTimeframeEmbedding catálogo dinámico.
* SignalPolicy y HybridSignalEngine end-to-end (CPU).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.calibration import LowLatencyRollingIsotonicCalibrator
from src.models.cnn_extractor import CNN1DExtractor, CausalConv1d
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.ensemble import HybridSignalEngine, SignalPolicy
from src.models.heads import HeadConfig, MultiContractMultiHorizonHead
from src.models.hybrid_tft import HybridCNNLSTMTFT
from src.models.meta_learner import RegimeAwareMetaLearner
from src.models.tft_attention import TFTFusionNode


# ---------------------------------------------------------------------------
# Causalidad
# ---------------------------------------------------------------------------


def test_causal_conv1d_rejects_stride_gt_1() -> None:
    """Un stride > 1 con padding fijo rompe la causalidad: debe rechazarse."""
    # El constructor sólo soporta stride=1 (no expone el parámetro), así que
    # comprobamos que el módulo no acepta uno > 1 por la vía del kwarg.
    with pytest.raises(TypeError):
        CausalConv1d(4, 8, kernel_size=3, stride=2)  # type: ignore[call-arg]


def test_cnn_extractor_is_causal_under_future_perturbation() -> None:
    torch.manual_seed(0)
    model = CNN1DExtractor(
        num_features=4, sequence_length=16, embedding_dim=8, return_sequence=True
    ).eval()
    x = torch.randn(2, 16, 4)
    y_orig = model(x)
    # Perturbamos la mitad futura agresivamente; el pasado no debe cambiar.
    x_mut = x.clone()
    x_mut[:, 8:] = x_mut[:, 8:] + 100.0
    y_mut = model(x_mut)
    torch.testing.assert_close(y_orig[:, :8], y_mut[:, :8], rtol=1e-5, atol=1e-5)


def test_tft_fusion_is_causal_under_future_perturbation() -> None:
    torch.manual_seed(0)
    fusion = TFTFusionNode(
        embedding_dim=16, num_heads=4, num_sources=2, output_dim=16,
        average_attn_weights=True,
    ).eval()
    a = torch.randn(2, 8, 16)
    b = torch.randn(2, 8, 16)
    out_orig, _ = fusion([a, b])
    a_mut = a.clone()
    a_mut[:, 4:] += 50.0
    out_mut, _ = fusion([a_mut, b])
    torch.testing.assert_close(out_orig[:, :4], out_mut[:, :4], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Gradient flow + validaciones
# ---------------------------------------------------------------------------


def test_hybrid_backbone_gradient_flow() -> None:
    torch.manual_seed(0)
    model = HybridCNNLSTMTFT(
        input_features=6, sequence_length=10, cnn_channels=(8, 16),
        lstm_hidden=8, tft_hidden=8, num_attention_heads=2, lstm_layers=1,
    )
    x = torch.randn(3, 10, 6, requires_grad=False)
    logits, attn = model(x)  # usa forward() para incluir el classifier
    loss = logits.sum() + attn.sum() * 0  # attn participa via grafo
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, g in grads if g is None]
    assert not missing, f"missing gradients in: {missing}"
    assert any((g.abs().sum() > 0).item() for _, g in grads)


def test_tft_fusion_rejects_incompatible_embedding_dim() -> None:
    with pytest.raises(ValueError):
        TFTFusionNode(embedding_dim=15, num_heads=4)  # 15 % 4 != 0


def test_bilstm_rejects_invalid_rnn_type() -> None:
    with pytest.raises(ValueError):
        BiLSTMEncoder(input_size=4, rnn_type="conv")


def test_bilstm_gru_path() -> None:
    model = BiLSTMEncoder(
        input_size=4, hidden_size=8, num_layers=1, embedding_dim=8, rnn_type="gru"
    )
    out = model(torch.randn(2, 5, 4))
    assert out.shape == (2, 8)


def test_cnn_extractor_auto_groups_for_odd_channels() -> None:
    # 7 canales no es divisible por 8 → debe ajustar a gcd(7,8)=1, no crashear.
    model = CNN1DExtractor(
        num_features=3, sequence_length=8, embedding_dim=4,
        channels=(7, 14), kernel_sizes=(3, 3), dilations=(1, 1),
    )
    out = model(torch.randn(2, 8, 3))
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Calibrador: thread safety + métricas
# ---------------------------------------------------------------------------


def test_calibrator_update_in_background_is_race_free() -> None:
    """Múltiples llamadas concurrentes a update_in_background no deben
    dispararse en paralelo: la guarda debe permitir sólo un thread activo."""
    cal = LowLatencyRollingIsotonicCalibrator(window_size=500, min_observations=20)
    rng = np.random.default_rng(7)
    for _ in range(200):
        m = float(rng.standard_normal())
        cal.add_observation(m, int(m + 0.2 * rng.standard_normal() > 0))

    spawned = []
    barrier = threading.Barrier(8)

    def runner() -> None:
        barrier.wait()
        spawned.append(cal.update_in_background())

    threads = [threading.Thread(target=runner) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Como máximo uno spawneó el refit; los demás deben devolver False.
    assert sum(1 for s in spawned if s) <= 1
    # Esperar al fin del refit (timeout generoso para entornos lentos / JIT).
    assert cal.wait_update_done(timeout=30.0), "background update did not finish"
    cal.update_calibration_curve()  # asegura un fit completo terminado
    assert cal.is_fitted


def test_calibrator_brier_and_ece_bounded() -> None:
    cal = LowLatencyRollingIsotonicCalibrator(window_size=2000, min_observations=50)
    rng = np.random.default_rng(3)
    for _ in range(1000):
        m = float(rng.standard_normal())
        cal.add_observation(m, int(m + 0.2 * rng.standard_normal() > 0))
    assert cal.update_calibration_curve()
    brier = cal.brier_score()
    ece = cal.expected_calibration_error(n_bins=10)
    assert 0.0 <= brier <= 1.0
    assert 0.0 <= ece <= 1.0


def test_calibrator_reset_clears_state() -> None:
    cal = LowLatencyRollingIsotonicCalibrator(window_size=200, min_observations=10)
    for i in range(50):
        cal.add_observation(float(i % 5), i % 2)
    cal.update_calibration_curve()
    assert cal.is_fitted
    cal.reset()
    assert not cal.is_fitted
    assert cal.n_observations == 0


# ---------------------------------------------------------------------------
# Meta-learner: SHAP, TimeSeriesSplit, validaciones
# ---------------------------------------------------------------------------


def _synthetic_regime_dataset(n: int = 300, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    y = (
        ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
        + ((X[:, 0] < 0) & (X[:, 2] > 0)).astype(int) * 2
    )
    return X, y.astype(int)


def test_meta_learner_rejects_out_of_range_labels() -> None:
    learner = RegimeAwareMetaLearner(n_splits=2, n_estimators=10)
    X, _ = _synthetic_regime_dataset(60)
    with pytest.raises(ValueError):
        learner.fit(X, np.array([0, 1, 3] * (60 // 3))[:60])


def test_meta_learner_cross_val_mlogloss_shape() -> None:
    learner = RegimeAwareMetaLearner(n_splits=3, n_estimators=10)
    X, y = _synthetic_regime_dataset(150)
    losses = learner.cross_val_mlogloss(X, y)
    assert losses.shape == (3,)
    assert np.all(np.isfinite(losses))


def test_meta_learner_shap_modern_api() -> None:
    learner = RegimeAwareMetaLearner(n_splits=2, n_estimators=10).fit(
        *_synthetic_regime_dataset(120)
    )
    X_new, _ = _synthetic_regime_dataset(5, seed=99)
    explanations = learner.get_regime_explanation(X_new, top_n=3)
    assert len(explanations) == 5
    assert all(len(e) == 3 for e in explanations)


def test_meta_learner_feature_importances_exposed() -> None:
    learner = RegimeAwareMetaLearner(n_splits=2, n_estimators=10).fit(
        *_synthetic_regime_dataset(80)
    )
    importances = learner.feature_importances_
    assert importances.shape == (10,)
    assert (importances >= 0).all()


def test_meta_learner_class_weight_balanced() -> None:
    learner = RegimeAwareMetaLearner(
        n_splits=2, n_estimators=10, class_weight="balanced"
    )
    X, y = _synthetic_regime_dataset(120)
    learner.fit(X, y)
    assert learner.is_fitted


# ---------------------------------------------------------------------------
# Heads y conditioning
# ---------------------------------------------------------------------------


def test_multi_contract_head_output_shape() -> None:
    cfg = HeadConfig(
        contracts=("CALLPUT", "HIGHERLOWER"),
        horizons=(1, 5),
        use_context=False,
    )
    head = MultiContractMultiHorizonHead(input_dim=12, config=cfg, context_dim=0)
    out = head(torch.randn(4, 12))
    assert out.shape == (4, 2, 2)
    d = head.as_dict(out)
    assert set(d.keys()) == {"CALLPUT", "HIGHERLOWER"}
    assert set(d["CALLPUT"].keys()) == {1, 5}


def test_multi_contract_head_with_context() -> None:
    cfg = HeadConfig(
        contracts=("CALLPUT",), horizons=(1, 3), use_context=True, dropout=0.0
    )
    head = MultiContractMultiHorizonHead(input_dim=8, config=cfg, context_dim=4)
    emb = torch.randn(3, 8)
    ctx = torch.randn(3, 4)
    out = head(emb, ctx)
    assert out.shape == (3, 1, 2)


def test_asset_timeframe_embedding_dynamic_vocab() -> None:
    emb = AssetTimeframeEmbedding(embedding_dim=8)
    sid = emb.register_symbol("R_100")
    gid = emb.register_granularity(60)
    vec = emb(torch.tensor([sid]), torch.tensor([gid]))
    assert vec.shape == (1, 8)
    # Granularidad None == ticks (id 0 reservado).
    tick_id = emb.register_granularity(None)
    assert tick_id == emb.granularity_id(0)


# ---------------------------------------------------------------------------
# Ensemble end-to-end
# ---------------------------------------------------------------------------


def test_hybrid_signal_engine_extract_features_device_aware() -> None:
    engine = HybridSignalEngine(num_features=4, sequence_length=8, embedding_dim=16)
    x = torch.randn(2, 8, 4)
    feats = engine.extract_features(x, as_numpy=False)
    assert feats["embedding"].shape == (2, 16)
    assert feats["logits"].shape == (2,)
    feats_np = engine.extract_features(x, as_numpy=True)
    assert isinstance(feats_np["embedding"], np.ndarray)


def test_signal_policy_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError):
        SignalPolicy(call_threshold=0.4, put_threshold=0.5)


# ---------------------------------------------------------------------------
# P4.31 — desacoplo embedding_dim ↔ hidden_size
# ---------------------------------------------------------------------------


def test_bilstm_embedding_dim_defaults_to_hidden_size_for_unidirectional() -> None:
    enc = BiLSTMEncoder(input_size=8, hidden_size=32, bidirectional=False)
    out = enc(torch.randn(2, 6, 8))
    # Default desacoplado: embedding_dim = hidden_size para unidireccional.
    assert out.shape == (2, 32)


def test_bilstm_embedding_dim_defaults_to_2x_for_bidirectional() -> None:
    enc = BiLSTMEncoder(input_size=8, hidden_size=16, bidirectional=True)
    out = enc(torch.randn(2, 6, 8))
    # Bidirectional → embedding_dim default = hidden_size * 2.
    assert out.shape == (2, 32)


def test_bilstm_embedding_dim_explicit_override() -> None:
    enc = BiLSTMEncoder(input_size=8, hidden_size=16, embedding_dim=64)
    assert enc(torch.randn(2, 6, 8)).shape == (2, 64)


# ---------------------------------------------------------------------------
# P4.32 — FP16/BF16 inference numerical stability
# ---------------------------------------------------------------------------


def test_hybrid_signal_engine_fp16_inference_matches_fp32_within_tolerance() -> None:
    """``.half()`` debe producir logits cercanos a FP32 dentro de tolerancia."""
    torch.manual_seed(0)
    engine = HybridSignalEngine(
        num_features=4, sequence_length=8, embedding_dim=16
    ).eval()
    x = torch.randn(3, 8, 4)
    out_fp32 = engine.extract_features(x, as_numpy=False)["logits"]

    engine_fp16 = HybridSignalEngine(
        num_features=4, sequence_length=8, embedding_dim=16
    ).eval()
    engine_fp16.load_state_dict(engine.state_dict())
    engine_fp16 = engine_fp16.half()
    out_fp16 = engine_fp16.extract_features(x.half(), as_numpy=False)["logits"]

    torch.testing.assert_close(
        out_fp16.float(), out_fp32, rtol=1e-2, atol=1e-2,
    )


def test_hybrid_signal_engine_bf16_autocast_inference() -> None:
    """``torch.autocast(bf16)`` produce logits cercanos a FP32."""
    torch.manual_seed(0)
    engine = HybridSignalEngine(
        num_features=4, sequence_length=8, embedding_dim=16
    ).eval()
    x = torch.randn(3, 8, 4)
    out_fp32 = engine.extract_features(x, as_numpy=False)["logits"]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with torch.backends.mkldnn.flags(enabled=False):
            out_bf16 = engine.extract_features(x, as_numpy=False)["logits"]
    torch.testing.assert_close(
        out_bf16.float(), out_fp32, rtol=2e-2, atol=2e-2,
    )


# ---------------------------------------------------------------------------
# P4.35 — key_padding_mask con secuencias de longitud variable
# ---------------------------------------------------------------------------


def test_tft_fusion_key_padding_mask_zeros_out_padded_attention() -> None:
    """Con ``key_padding_mask=True`` en la cola, los tokens válidos del head
    no atienden a las posiciones paddeadas (peso = 0) y los logits resultan
    idénticos a una secuencia "limpia" del mismo largo válido."""
    torch.manual_seed(0)
    fusion = TFTFusionNode(
        embedding_dim=16, num_heads=4, num_sources=2, output_dim=16,
        average_attn_weights=False,
    ).eval()

    # Secuencia base de largo 6.
    a_full = torch.randn(1, 6, 16)
    b_full = torch.randn(1, 6, 16)

    # Versión paddeada: alarga a 8 con ruido + máscara que oculta las posiciones 6 y 7.
    a_pad = torch.cat([a_full, torch.randn(1, 2, 16) * 1000.0], dim=1)
    b_pad = torch.cat([b_full, torch.randn(1, 2, 16) * 1000.0], dim=1)
    kpm = torch.zeros(1, 8, dtype=torch.bool)
    kpm[0, 6:] = True

    out_full, attn_full = fusion([a_full, b_full])
    out_pad, attn_pad = fusion([a_pad, b_pad], key_padding_mask=kpm)

    # Los primeros 6 outputs deben coincidir bit-a-bit con la versión sin padding.
    torch.testing.assert_close(out_full, out_pad[:, :6], rtol=1e-4, atol=1e-4)
    # Pesos de atención hacia las posiciones paddeadas deben ser ~0.
    assert attn_pad.shape == (1, 4, 8, 8)
    pad_attention = attn_pad[..., 6:]
    assert pad_attention.abs().max().item() < 1e-5


def test_hybrid_signal_engine_generate_signal_returns_full_payload() -> None:
    engine = HybridSignalEngine(num_features=3, sequence_length=6, embedding_dim=8)
    out = engine.generate_signal(
        torch.randn(6, 3), asset="R_100", timeframe="60s", as_of_epoch=1_700_000_000
    )
    assert out["asset"] == "R_100"
    assert out["timeframe"] == "60s"
    assert out["signal"] in {"CALL", "PUT", "NO_TRADE"}
    assert "1970" not in out["timestamp"] and "2023" in out["timestamp"]
    assert set(out["regime"]["probs"].keys()) == set(engine.regime_labels)
    assert "route" in out["execution"]
