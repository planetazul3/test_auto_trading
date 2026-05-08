import sys
import os
import json
import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Asegurar que Python encuentre el paquete 'src' al ejecutar el script directamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.cnn_extractor import CNN1DExtractor
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.tft_attention import TFTFusionNode
from src.models.meta_learner import XGBoostMetaLearner

class HybridSignalEngine(nn.Module):
    """
    Motor de señales híbrido que unifica CNN, BiLSTM, TFT y XGBoost.
    Genera la señal final calibrada en formato JSON.
    """
    def __init__(self, num_features: int, sequence_length: int, embedding_dim: int = 64):
        super(HybridSignalEngine, self).__init__()
        
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        
        # 1. Inicializar Modelos Base (Deep Learning)
        self.cnn = CNN1DExtractor(num_features, sequence_length, embedding_dim)
        self.lstm = BiLSTMEncoder(num_features, hidden_size=64, num_layers=2, embedding_dim=embedding_dim)
        self.tft = TFTFusionNode(embedding_dim=embedding_dim, num_heads=4, num_sources=2, output_dim=embedding_dim)
        
        # Cabezales auxiliares para obtener p_call_raw de cada componente
        self.cnn_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.lstm_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.tft_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())
        
        # 2. Inicializar Meta-Learner (Machine Learning Clásico)
        self.meta_learner = XGBoostMetaLearner(calibration_method='isotonic')
        
    def extract_features(self, x: torch.Tensor) -> dict:
        """
        Pasa los datos por la red neuronal y extrae embeddings y probabilidades crudas.
        """
        # Modo evaluación para inferencia
        self.cnn.eval()
        self.lstm.eval()
        self.tft.eval()
        
        with torch.no_grad():
            # Extraer representaciones
            cnn_emb = self.cnn(x)
            lstm_emb = self.lstm(x)
            
            # Fusión con atención
            tft_emb, attn_weights = self.tft(cnn_emb, lstm_emb)
            
            # Probabilidades crudas (auxiliares)
            p_cnn = self.cnn_head(cnn_emb).squeeze(-1).numpy()
            p_lstm = self.lstm_head(lstm_emb).squeeze(-1).numpy()
            p_tft = self.tft_head(tft_emb).squeeze(-1).numpy()
            
        return {
            "tft_embeddings": tft_emb.numpy(),
            "raw_probs": {
                "cnn": p_cnn,
                "lstm": p_lstm,
                "tft": p_tft
            }
        }

    def fit_meta_learner(self, x_train: torch.Tensor, y_train: np.ndarray):
        """
        Entrena el meta-learner XGBoost usando las representaciones del TFT.
        En producción, la red neuronal se entrenaría primero. Aquí asumimos
        que la red ya extrae características útiles.
        """
        print("Extrayendo embeddings del TFT para entrenamiento...")
        features = self.extract_features(x_train)
        tft_embeddings = features["tft_embeddings"]
        
        print("Entrenando y calibrando XGBoost Meta-Learner...")
        self.meta_learner.fit(tft_embeddings, y_train)

    def generate_signal(self, x_window: torch.Tensor, asset: str, timeframe: str, 
                        feature_names: list, current_regime: str) -> dict:
        """
        Genera el JSON final de la señal para una ventana de tiempo específica.
        """
        if x_window.dim() == 2:
            x_window = x_window.unsqueeze(0) # Añadir dimensión de batch
            
        # 1. Extracción profunda
        dl_features = self.extract_features(x_window)
        tft_emb = dl_features["tft_embeddings"]
        raw_probs = dl_features["raw_probs"]
        
        # 2. Predicción del Meta-Learner
        xgb_probs = self.meta_learner.predict_proba(tft_emb)
        p_call_calibrated = float(xgb_probs["p_call"][0])
        
        # Probabilidad cruda del XGBoost base (antes de calibrar)
        # Para simplificar en este script, usamos la calibrada como proxy de la cruda del xgb
        # en un entorno real se extraería del self.meta_learner.base_model
        p_xgb_raw = float(self.meta_learner.base_model.predict_proba(tft_emb)[0, 1])
        
        # 3. Lógica de Decisión (Umbrales estrictos)
        if p_call_calibrated > 0.72:
            signal = "CALL"
            confidence = "high" if p_call_calibrated > 0.85 else "medium"
        elif p_call_calibrated < 0.28:
            signal = "PUT"
            # Si p_call < 0.28, entonces p_put > 0.72
            confidence = "high" if p_call_calibrated < 0.15 else "medium"
        else:
            signal = "NO_TRADE"
            confidence = "low"
            
        # 4. Explicabilidad SHAP
        top_features = self.meta_learner.get_shap_explanations(
            tft_emb, feature_names=feature_names, top_n=5
        )[0]
        
        # 5. Construcción del JSON
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
    print("Inicializando Hybrid Signal Engine...")
    
    # Parámetros de la arquitectura
    SEQ_LENGTH = 60
    NUM_FEATURES = 125
    EMBEDDING_DIM = 64
    
    # 1. Generar datos sintéticos para simular el entrenamiento
    torch.manual_seed(42)
    np.random.seed(42)
    
    X_train_tensor = torch.randn(500, SEQ_LENGTH, NUM_FEATURES)
    # Target sintético binario
    y_train_np = np.random.randint(0, 2, size=(500,))
    
    try:
        # Instanciar el motor
        engine = HybridSignalEngine(
            num_features=NUM_FEATURES, 
            sequence_length=SEQ_LENGTH, 
            embedding_dim=EMBEDDING_DIM
        )
        
        # Entrenar el meta-learner
        engine.fit_meta_learner(X_train_tensor, y_train_np)
        
        # 2. Simular una inferencia en tiempo real (1 sola ventana de 60 velas)
        X_live_window = torch.randn(1, SEQ_LENGTH, NUM_FEATURES)
        
        # Nombres de features simulados para el SHAP (en realidad vendrían del TFT)
        tft_feature_names = [f"tft_latent_dim_{i}" for i in range(EMBEDDING_DIM)]
        
        print("\nGenerando señal de trading...")
        signal_json = engine.generate_signal(
            x_window=X_live_window,
            asset="BTC/USDT",
            timeframe="5m",
            feature_names=tft_feature_names,
            current_regime="volatile"
        )
        
        print("\n--- SALIDA FINAL DEL SISTEMA (JSON) ---")
        print(json.dumps(signal_json, indent=2))
        
        # Verificaciones de integridad
        assert "signal" in signal_json, "Falta la clave 'signal'"
        assert signal_json["signal"] in ["CALL", "PUT", "NO_TRADE"], "Señal inválida"
        assert 0.0 <= signal_json["p_call_calibrated"] <= 1.0, "Probabilidad fuera de rango"
        
        print("\n[OK] Pipeline Ensemble ejecutado exitosamente de inicio a fin.")
        
    except Exception as e:
        print(f"\n[ERROR] Fallo en la ejecución del Ensemble: {str(e)}")