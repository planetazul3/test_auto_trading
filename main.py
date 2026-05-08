import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score

# Suprimir warnings para salida limpia en consola
warnings.filterwarnings("ignore")

# Importar módulos propios
from src.features.generator import FeatureGenerator
from src.models.ensemble import HybridSignalEngine

def create_sequences(features_df: pd.DataFrame, target_series: pd.Series, seq_length: int):
    """
    Convierte datos tabulares en tensores 3D (Batch, Seq_Len, Features)
    usando una ventana deslizante.
    """
    X, y = [],[]
    feature_values = features_df.values
    target_values = target_series.values
    
    for i in range(len(features_df) - seq_length):
        X.append(feature_values[i : i + seq_length])
        y.append(target_values[i + seq_length])
        
    return torch.tensor(np.array(X), dtype=torch.float32), np.array(y)

def calculate_max_drawdown(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    Calcula el Máximo Drawdown basado en los retornos generados por las señales.
    Señal: 1 (CALL), -1 (PUT), 0 (NO_TRADE).
    """
    strategy_returns = signals * returns
    cumulative_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    return abs(float(np.min(drawdowns)))

def main():
    print("=== INICIANDO ML-SIGNAL-ENGINE (PIPELINE END-TO-END) ===")
    
    # 1. Configuración de MLflow
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("PUT_CALL_Hybrid_System")
    
    # Parámetros globales
    SEQ_LENGTH = 60
    EMBEDDING_DIM = 64
    DAYS_TRAIN = 60
    DAYS_TEST = 10
    CANDLES_PER_DAY = 288 # Timeframe de 5m (24h * 60m / 5m)
    
    # 2. Simulación de Datos (80 días en total para permitir Walk-Forward)
    print("\n[1/5] Simulando datos de mercado (80 días, TF: 5m)...")
    np.random.seed(42)
    total_candles = 80 * CANDLES_PER_DAY
    dates = pd.date_range(start='2025-01-01', periods=total_candles, freq='5min')
    
    close_prices = 50000 + np.random.randn(total_candles).cumsum() * 10
    df_raw = pd.DataFrame({
        'open': close_prices + np.random.randn(total_candles) * 2,
        'high': close_prices + np.abs(np.random.randn(total_candles) * 5),
        'low': close_prices - np.abs(np.random.randn(total_candles) * 5),
        'close': close_prices,
        'volume': np.abs(np.random.randn(total_candles) * 100),
        'bid': close_prices - 0.5,
        'ask': close_prices + 0.5,
        'bid_vol': np.abs(np.random.randn(total_candles) * 50),
        'ask_vol': np.abs(np.random.randn(total_candles) * 50)
    }, index=dates)
    
    # 3. Generación de Features
    print("[2/5] Generando 120+ features sin Data Leakage...")
    generator = FeatureGenerator()
    df_features = generator.generate_features(df_raw)
    
    # Definir Target: 1 si el precio sube en las próximas 3 velas, 0 si baja
    df_features['target'] = (df_features['close'].shift(-3) > df_features['close']).astype(int)
    df_features['future_return'] = df_features['close'].shift(-3) / df_features['close'] - 1
    df_features = df_features.dropna()
    
    features_only = df_features.drop(columns=['target', 'future_return'])
    target_only = df_features['target']
    returns_only = df_features['future_return']
    
    NUM_FEATURES = len(features_only.columns)
    
    # 4. Walk-Forward Validation
    print(f"[3/5] Iniciando Walk-Forward Validation ({DAYS_TRAIN}d Train / {DAYS_TEST}d Test)...")
    
    train_size = DAYS_TRAIN * CANDLES_PER_DAY
    test_size = DAYS_TEST * CANDLES_PER_DAY
    
    # Tomamos la primera ventana para la demostración
    train_features = features_only.iloc[:train_size]
    train_target = target_only.iloc[:train_size]
    
    test_features = features_only.iloc[train_size : train_size + test_size]
    test_target = target_only.iloc[train_size : train_size + test_size]
    test_returns = returns_only.iloc[train_size : train_size + test_size].values
    
    # Crear tensores 3D
    X_train, y_train = create_sequences(train_features, train_target, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_features, test_target, SEQ_LENGTH)
    
    # Ajustar retornos para que coincidan con la longitud de y_test
    test_returns = test_returns[SEQ_LENGTH:]
    
    # 5. Entrenamiento y MLflow Tracking
    print("[4/5] Entrenando modelo Ensemble y registrando en MLflow...")
    with mlflow.start_run(run_name="Walk_Forward_Window_1"):
        # Loggear parámetros
        mlflow.log_params({
            "seq_length": SEQ_LENGTH,
            "num_features": NUM_FEATURES,
            "embedding_dim": EMBEDDING_DIM,
            "train_days": DAYS_TRAIN,
            "test_days": DAYS_TEST
        })
        
        # Inicializar y entrenar
        engine = HybridSignalEngine(NUM_FEATURES, SEQ_LENGTH, EMBEDDING_DIM)
        engine.fit_meta_learner(X_train, y_train)
        
        # Inferencia en Test
        print("[5/5] Evaluando métricas en ventana de Test...")
        dl_features = engine.extract_features(X_test)
        tft_emb = dl_features["tft_embeddings"]
        
        probs = engine.meta_learner.predict_proba(tft_emb)
        p_call = probs["p_call"]
        
        # Calcular métricas estándar
        y_pred = (p_call > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, p_call)
        
        # Calcular métricas de negocio (Alta Confianza y Drawdown)
        # Umbrales: CALL > 0.72, PUT < 0.28
        high_conf_mask = (p_call > 0.72) | (p_call < 0.28)
        if np.sum(high_conf_mask) > 0:
            y_test_hc = y_test[high_conf_mask]
            y_pred_hc = (p_call[high_conf_mask] > 0.72).astype(int)
            prec_hc = precision_score(y_test_hc, y_pred_hc, zero_division=0)
        else:
            prec_hc = 0.0
            
        # Simular señales para Drawdown
        signals = np.where(p_call > 0.72, 1, np.where(p_call < 0.28, -1, 0))
        mdd = calculate_max_drawdown(signals, test_returns)
        
        # Loggear métricas
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "auc_roc": auc,
            "precision_high_conf": prec_hc,
            "max_drawdown": mdd
        }
        mlflow.log_metrics(metrics)
        
        # Generar una señal de ejemplo (última vela del test)
        sample_window = X_test[-1:]
        feature_names = [f"tft_dim_{i}" for i in range(EMBEDDING_DIM)]
        
        sample_signal = engine.generate_signal(
            x_window=sample_window,
            asset="BTC/USDT",
            timeframe="5m",
            feature_names=feature_names,
            current_regime="trending"
        )
        
        # Guardar señal como artefacto
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/sample_signal.json", "w") as f:
            json.dump(sample_signal, f, indent=2)
        mlflow.log_artifact("artifacts/sample_signal.json")
        
        print("\n=== RESULTADOS DE VALIDACIÓN WALK-FORWARD ===")
        print(f"Accuracy:                {acc:.4f}")
        print(f"F1-Score:                {f1:.4f}")
        print(f"AUC-ROC:                 {auc:.4f}")
        print(f"Precision (Alta Conf.):  {prec_hc:.4f} (Umbrales >0.72 / <0.28)")
        print(f"Max Drawdown:            {mdd:.2%}")
        print("\nEjemplo de Señal Generada (Guardada en MLflow):")
        print(json.dumps(sample_signal, indent=2))
        print("\n[OK] Pipeline ejecutado exitosamente. Revisa la UI de MLflow con 'uv run mlflow ui'")

if __name__ == "__main__":
    main()