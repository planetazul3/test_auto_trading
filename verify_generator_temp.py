import sys
import os
import pandas as pd
import numpy as np

# Añadir src al path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from features.generator import FeatureGenerator

def verify_generator():
    print("Iniciando verificación manual del Feature Generator...")
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='5min')
    close_prices = 100000 + np.random.randn(1000).cumsum() * 10
    df_synthetic = pd.DataFrame({
        'open': close_prices + np.random.randn(1000) * 2,
        'high': close_prices + np.abs(np.random.randn(1000) * 5),
        'low': close_prices - np.abs(np.random.randn(1000) * 5),
        'close': close_prices,
        'volume': np.abs(np.random.randn(1000) * 1000),
        'bid': close_prices - 0.5,
        'ask': close_prices + 0.5,
        'bid_vol': np.abs(np.random.randn(1000) * 500),
        'ask_vol': np.abs(np.random.randn(1000) * 500)
    }, index=dates)

    generator = FeatureGenerator()
    try:
        df_features = generator.generate_features(df_synthetic)
        print(f"Filas originales: {len(df_synthetic)}")
        print(f"Filas tras limpieza: {len(df_features)}")
        print(f"Features generadas: {len(df_features.columns)}")
        print(f"NaNs restantes: {df_features.isna().sum().sum()} (debe ser 0)")
        print(df_features[['close', 'EMA_200', 'RSI_14', 'bid_ask_spread', 'hmm_hidden_state']].tail())
        
        # Verificar específicamente el GMM/HMM state
        unique_states = df_features['hmm_hidden_state'].unique()
        print(f"Estados únicos de hmm_hidden_state: {unique_states}")
        
        if len(df_features) > 0 and df_features.isna().sum().sum() == 0:
            print("[OK] Módulo ejecutado exitosamente.")
        else:
            print("[ERROR] Falló la validación de salida.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    verify_generator()
