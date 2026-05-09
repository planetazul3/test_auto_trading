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


class HybridSignalEngine(nn.Module):
    """
    Motor de señales híbrido: CNN + BiLSTM + TFT -> embedding -> XGBoost.
    """

    def __init__(self, num_features: int, sequence_length: int, embedding_dim: int = 64):
        super(HybridSignalEngine, self).__init__()

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Modelos base de deep learning
        self.cnn = CNN1DExtractor(num_features, sequence_length, embedding_dim)
        self.lstm = BiLSTMEncoder(num_features, hidden_size=64, num_layers=2, embedding_dim=embedding_dim)
        self.tft = TFTFusionNode(embedding_dim=embedding_dim, num_heads=4, num_sources=2, output_dim=embedding_dim)

        # Cabezales auxiliares para obtener probabilidades crudas de cada rama
        self.cnn_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.lstm_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.tft_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

        # Meta-learner (XGBoost calibrado)
        self.meta_learner = RegimeAwareMetaLearner(calibration_method='isotonic')

    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Pasa los datos por la red neuronal y devuelve embeddings del TFT
        y probabilidades crudas de cada rama.
        """
        self.cnn.eval()
        self.lstm.eval()
        self.tft.eval()

        with torch.no_grad():
            cnn_emb = self.cnn(x)
            lstm_emb = self.lstm(x)
            tft_emb, attn_weights = self.tft([cnn_emb, lstm_emb])

            p_cnn = self.cnn_head(cnn_emb).squeeze(-1).cpu().numpy()
            p_lstm = self.lstm_head(lstm_emb).squeeze(-1).cpu().numpy()
            p_tft = self.tft_head(tft_emb).squeeze(-1).cpu().numpy()

        return {
            "tft_embeddings": tft_emb.cpu().numpy(),
            "raw_probs": {
                "cnn": p_cnn,
                "lstm": p_lstm,
                "tft": p_tft
            }
        }

    def fit_meta_learner(self, X, y):
        """
        Entrena el meta-learner.
        Acepta:
            - Tensor 3D (batch, seq, features): extraerá embeddings primero.
            - Numpy array 2D (embeddings ya extraídos): los usará directamente.
        """
        if isinstance(X, torch.Tensor):
            # Extraer embeddings con la red actual
            features = self.extract_features(X)
            embeddings = features["tft_embeddings"]
        elif isinstance(X, np.ndarray):
            embeddings = X
        else:
            raise TypeError("X debe ser torch.Tensor 3D o numpy.ndarray 2D")

        print("Entrenando y calibrando XGBoost Meta-Learner (validación temporal)...")
        self.meta_learner.fit(embeddings, y)

    def generate_signal(self, x_window: torch.Tensor, asset: str, timeframe: str,
                        feature_names: list, current_regime: str) -> dict:
        """
        Genera el JSON final de la señal para una ventana de tiempo específica.
        """
        if x_window.dim() == 2:
            x_window = x_window.unsqueeze(0)  # añadir batch

        dl_features = self.extract_features(x_window)
        tft_emb = dl_features["tft_embeddings"]
        raw_probs = dl_features["raw_probs"]

        # Predicción del meta-learner
        xgb_probs = self.meta_learner.predict_proba(tft_emb)
        p_call_calibrated = float(xgb_probs["p_call"][0])

        # Probabilidad cruda del modelo base (sin calibrar)
        p_xgb_raw = float(self.meta_learner.base_model.predict_proba(tft_emb)[0, 1])

        # Lógica de decisión con umbrales estrictos
        if p_call_calibrated > 0.72:
            signal = "CALL"
            confidence = "high" if p_call_calibrated > 0.85 else "medium"
        elif p_call_calibrated < 0.28:
            signal = "PUT"
            confidence = "high" if p_call_calibrated < 0.15 else "medium"
        else:
            signal = "NO_TRADE"
            confidence = "low"

        # Explicabilidad SHAP
        top_features = self.meta_learner.get_shap_explanations(
            tft_emb, feature_names=feature_names, top_n=5
        )[0]

        # Construcción del JSON
        signal_data = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "asset": asset,
            "trigger_timeframe": timeframe,
            "p_call_raw": {
                "cnn": round(float(raw_probs["cnn"][0]), 4),
                "lstm": round(float(raw_probs["lstm"][0]), 4),
                "tft": round(float(raw_probs["tft"][0]), 4),
                "xgb": round(p_xgb_raw, 4)
            },
            "p_call_calibrated": round(p_call_calibrated, 4),
            "signal": signal,
            "confidence": confidence,
            "top_features": top_features,
            "market_regime": current_regime
        }
        return signal_data


if __name__ == "__main__":
    print("Inicializando Hybrid Signal Engine (corregido)...")
    SEQ_LENGTH = 60
    NUM_FEATURES = 125
    EMBEDDING_DIM = 64

    torch.manual_seed(42)
    np.random.seed(42)

    X_train_tensor = torch.randn(500, SEQ_LENGTH, NUM_FEATURES)
    y_train_np = np.random.randint(0, 2, size=(500,))

    try:
        engine = HybridSignalEngine(NUM_FEATURES, SEQ_LENGTH, EMBEDDING_DIM)

        # Ejemplo: entrenar meta-learner pasándole directamente embeddings pre-extraídos
        # En un caso real primero se entrenan los modelos profundos.
        with torch.no_grad():
            features = engine.extract_features(X_train_tensor)
            emb = features["tft_embeddings"]
        engine.fit_meta_learner(emb, y_train_np)

        # Generar una señal de ejemplo
        X_live = torch.randn(1, SEQ_LENGTH, NUM_FEATURES)
        feature_names = [f"tft_dim_{i}" for i in range(EMBEDDING_DIM)]
        signal_json = engine.generate_signal(
            X_live, "BTC/USDT", "5m", feature_names, "volatile"
        )

        print("\n--- SALIDA FINAL ---")
        print(json.dumps(signal_json, indent=2))
        print("\n[OK] Ensemble engine ejecutado correctamente.")

    except Exception as e:
        print(f"\n[ERROR] {e}")