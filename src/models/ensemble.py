"""
Motor de señales híbrido que unifica CNN, BiLSTM, TFT y XGBoost.
Versión 2.0: soporta entrenamiento separado de componentes profundos
y ajuste del meta-learner con embeddings precomputados o con tensor crudo.
"""
import sys
import os
import json
import datetime
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cnn_extractor import CNN1DExtractor
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.tft_attention import TFTFusionNode
from src.models.meta_learner import RegimeAwareMetaLearner


from src.models.calibration import LowLatencyRollingIsotonicCalibrator

class HybridSignalEngine(nn.Module):
    """
    Motor de señales híbrido: CNN + BiLSTM + TFT -> Regime Meta-Learner & Calibrated Signal.
    """

    def __init__(self, num_features: int, sequence_length: int, embedding_dim: int = 64):
        super(HybridSignalEngine, self).__init__()

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Modelos base con preservación de secuencia
        self.cnn = CNN1DExtractor(num_features, sequence_length, embedding_dim, return_sequence=True)
        self.lstm = BiLSTMEncoder(num_features, hidden_size=64, num_layers=2, embedding_dim=embedding_dim, return_sequence=True)
        self.tft = TFTFusionNode(embedding_dim=embedding_dim, num_heads=4, num_sources=2, output_dim=embedding_dim)

        # Cabezal de señal (TFT final representation)
        self.signal_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

        # Meta-learner de Regímenes (XGBoost Multi-clase)
        self.regime_meta_learner = RegimeAwareMetaLearner()
        
        # Calibrador de señales de baja latencia (PAVA / Isotonic)
        self.calibrator = LowLatencyRollingIsotonicCalibrator(window_size=5000)

    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Extrae embeddings y probabilidad cruda del modelo profundo.
        """
        self.cnn.eval()
        self.lstm.eval()
        self.tft.eval()

        with torch.no_grad():
            cnn_seq = self.cnn(x)
            lstm_seq = self.lstm(x)
            fused_seq, attn_weights = self.tft([cnn_seq, lstm_seq])

            # Solo nos interesa el último embedding para el meta-learner y el signal head
            last_emb = fused_seq[:, -1, :]
            p_raw = self.signal_head(last_emb).squeeze(-1).cpu().numpy()

        return {
            "last_embedding": last_emb.cpu().numpy(),
            "p_raw_tft": p_raw,
            "attn_weights": attn_weights.cpu().numpy()
        }

    def generate_signal(self, x_window: torch.Tensor, asset: str, timeframe: str,
                        feature_names: list) -> dict:
        """
        Genera señal ruteada y escalada por régimen de mercado.
        """
        if x_window.dim() == 2:
            x_window = x_window.unsqueeze(0)

        features = self.extract_features(x_window)
        emb = features["last_embedding"]
        p_raw = float(features["p_raw_tft"][0])

        # 1. Calibración de la señal (Logit a Prob Calibrada)
        # Convertimos prob cruda a logit para el calibrador
        logit = np.log(p_raw / (1 - p_raw + 1e-9))
        p_calibrated = self.calibrator.calibrate_signal(logit)

        # 2. Predicción de Régimen (Routing)
        regime_probs = self.regime_meta_learner.predict_regime_probs(emb)[0]
        # 0: Low Vol, 1: Trending, 2: High Vol
        current_regime = int(np.argmax(regime_probs))
        regime_labels = ["LOW_VOL", "TRENDING", "HIGH_VOL"]

        # 3. Lógica de Ruteo y Sizing (Capital Allocation)
        # Sizing base basado en confianza de la señal
        base_sizing = 1.0
        if p_calibrated > 0.80 or p_calibrated < 0.20:
            base_sizing = 1.5
        
        # Ajuste por régimen (Blueprint: reducir exposición en High Vol / Crash Risk)
        regime_multiplier = 1.0
        if current_regime == 0: # Low Vol / Mean Reversion
            regime_multiplier = 0.8 # Menor sizing si no hay tendencia clara
        elif current_regime == 2: # High Vol
            regime_multiplier = 0.4 # Sizing defensivo
        
        final_sizing = base_sizing * regime_multiplier

        # Decisión final
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
                "probs": {regime_labels[i]: round(float(regime_probs[i]), 4) for i in range(3)}
            },
            "execution": {
                "sizing_multiplier": round(final_sizing, 2),
                "route": "HFT_EXT" if current_regime == 1 else "LP_INTERNAL"
            }
        }



if __name__ == "__main__":
    print("Inicializando Hybrid Signal Engine (corregido)...")
    SEQ_LENGTH = 60
    NUM_FEATURES = 125
    EMBEDDING_DIM = 64

    torch.manual_seed(42)
    np.random.seed(42)

    X_train_tensor = torch.randn(100, SEQ_LENGTH, NUM_FEATURES)
    y_regimes = np.random.randint(0, 3, size=(100,))

    try:
        engine = HybridSignalEngine(NUM_FEATURES, SEQ_LENGTH, EMBEDDING_DIM)

        # Simular entrenamiento del meta-learner de regímenes
        with torch.no_grad():
            features = engine.extract_features(X_train_tensor)
            emb = features["last_embedding"]
        
        print("Entrenando Meta-Learner de Regímenes...")
        engine.regime_meta_learner.fit(emb, y_regimes)
        
        # Simular calibración inicial
        print("Simulando calibración de señal...")
        for _ in range(200):
            m = np.random.randn()
            l = 1 if m > 0.5 else 0
            engine.calibrator.add_observation(m, l)
        engine.calibrator.update_calibration_curve()

        # Generar una señal de ejemplo
        X_live = torch.randn(1, SEQ_LENGTH, NUM_FEATURES)
        feature_names = [f"f_{i}" for i in range(NUM_FEATURES)]
        signal_json = engine.generate_signal(
            X_live, "BTC/USDT", "5m", feature_names
        )

        print("\n--- SALIDA FINAL ---")
        print(json.dumps(signal_json, indent=2))
        print("\n[OK] Ensemble engine ejecutado correctamente.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()