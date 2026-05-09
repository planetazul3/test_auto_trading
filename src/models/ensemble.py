"""
Motor de señales híbrido: CNN + BiLSTM + TFT → Meta-Learner de regímenes
+ calibrador isotónico de baja latencia.

Diseño:
- El cabezal de señal emite **logits** (sin sigmoid) para alimentar
  directamente el calibrador isotónico, evitando el round-trip
  ``sigmoid → logit`` que pierde precisión.
- ``extract_features`` devuelve embeddings y logits crudos; cualquier
  decisión de probabilidad la toma el calibrador.
"""
import datetime
import json

import numpy as np
import torch
import torch.nn as nn

from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.calibration import LowLatencyRollingIsotonicCalibrator
from src.models.cnn_extractor import CNN1DExtractor
from src.models.meta_learner import RegimeAwareMetaLearner
from src.models.tft_attention import GatedResidualNetwork, TFTFusionNode


class HybridSignalEngine(nn.Module):
    """CNN + BiLSTM + TFT → Regime Meta-Learner & calibrador isotónico."""

    def __init__(self, num_features: int, sequence_length: int, embedding_dim: int = 64):
        super().__init__()

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.cnn = CNN1DExtractor(
            num_features, sequence_length, embedding_dim, return_sequence=True
        )
        self.lstm = BiLSTMEncoder(
            num_features,
            hidden_size=64,
            num_layers=2,
            embedding_dim=embedding_dim,
            return_sequence=True,
        )
        self.tft = TFTFusionNode(
            embedding_dim=embedding_dim,
            num_heads=4,
            num_sources=2,
            output_dim=embedding_dim,
        )
        self.output_grn = GatedResidualNetwork(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
        )
        # Cabezal: emite logit (sin sigmoid). La probabilidad la entrega el
        # calibrador isotónico aguas abajo.
        self.signal_head = nn.Linear(embedding_dim, 1)

        self.regime_meta_learner = RegimeAwareMetaLearner()
        self.calibrator = LowLatencyRollingIsotonicCalibrator(window_size=5000)

    def extract_features(self, x: torch.Tensor) -> dict:
        """Extrae embedding y logit crudo (modo eval, sin gradientes)."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                cnn_seq = self.cnn(x)
                lstm_seq = self.lstm(x)
                fused_seq, attn_weights = self.tft([cnn_seq, lstm_seq])
                last_emb = self.output_grn(fused_seq[:, -1, :])
                logits = self.signal_head(last_emb).squeeze(-1).cpu().numpy()
        finally:
            if was_training:
                self.train()

        return {
            "last_embedding": last_emb.cpu().numpy(),
            "logits": logits,
            "attn_weights": attn_weights.cpu().numpy(),
        }

    def generate_signal(
        self,
        x_window: torch.Tensor,
        asset: str,
        timeframe: str,
        feature_names: list,
    ) -> dict:
        """Genera una señal calibrada y ruteada por régimen."""
        if x_window.dim() == 2:
            x_window = x_window.unsqueeze(0)

        features = self.extract_features(x_window)
        emb = features["last_embedding"]
        logit = float(features["logits"][0])

        # 1. Calibración: logit → probabilidad calibrada (sin round-trip).
        p_calibrated = self.calibrator.calibrate_signal(logit)

        # 2. Régimen
        regime_probs = self.regime_meta_learner.predict_regime_probs(emb)[0]
        current_regime = int(np.argmax(regime_probs))
        regime_labels = ["LOW_VOL", "TRENDING", "HIGH_VOL"]

        # 3. Routing y sizing
        base_sizing = 1.5 if (p_calibrated > 0.80 or p_calibrated < 0.20) else 1.0
        regime_multiplier = {0: 0.8, 1: 1.0, 2: 0.4}[current_regime]
        final_sizing = base_sizing * regime_multiplier

        if p_calibrated > 0.70:
            signal = "CALL"
        elif p_calibrated < 0.30:
            signal = "PUT"
        else:
            signal = "NO_TRADE"
            final_sizing = 0.0

        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "asset": asset,
            "signal": signal,
            "p_call_calibrated": round(p_calibrated, 4),
            "regime": {
                "label": regime_labels[current_regime],
                "probs": {
                    regime_labels[i]: round(float(regime_probs[i]), 4)
                    for i in range(3)
                },
            },
            "execution": {
                "sizing_multiplier": round(final_sizing, 2),
                "route": "HFT_EXT" if current_regime == 1 else "LP_INTERNAL",
            },
        }


if __name__ == "__main__":
    print("Inicializando Hybrid Signal Engine...")
    SEQ_LENGTH = 60
    NUM_FEATURES = 125
    EMBEDDING_DIM = 64

    torch.manual_seed(42)
    np.random.seed(42)

    X_train_tensor = torch.randn(100, SEQ_LENGTH, NUM_FEATURES)
    y_regimes = np.random.randint(0, 3, size=(100,))

    engine = HybridSignalEngine(NUM_FEATURES, SEQ_LENGTH, EMBEDDING_DIM)

    with torch.no_grad():
        features = engine.extract_features(X_train_tensor)
        emb = features["last_embedding"]

    print("Entrenando Meta-Learner de Regímenes...")
    engine.regime_meta_learner.fit(emb, y_regimes)

    print("Simulando calibración de señal...")
    rng = np.random.default_rng(42)
    for _ in range(200):
        m = float(rng.standard_normal())
        l = 1 if m > 0.5 else 0
        engine.calibrator.add_observation(m, l)
    engine.calibrator.update_calibration_curve()

    X_live = torch.randn(1, SEQ_LENGTH, NUM_FEATURES)
    feature_names = [f"f_{i}" for i in range(NUM_FEATURES)]
    signal_json = engine.generate_signal(X_live, "BTC/USDT", "5m", feature_names)

    print("\n--- SALIDA FINAL ---")
    print(json.dumps(signal_json, indent=2))
    print("\n[OK] Ensemble engine ejecutado correctamente.")
