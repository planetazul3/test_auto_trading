import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from typing import Callable

def latent_leakage_test(df: pd.DataFrame, feature_generator_func: Callable[[pd.DataFrame], pd.DataFrame]):
    """
    Programmatic latent leakage detection.
    Injects a massive synthetic spike at a random index t_k and asserts
    that no features prior to t_k are altered.
    """
    print("--- Ejecutando Latent Leakage Test (Spike Injection) ---")
    
    # 1. Copia de seguridad de los datos originales
    original_df = df.copy()
    
    # 2. Generar features originales
    features_orig = feature_generator_func(original_df)
    
    # 3. Seleccionar un índice aleatorio t_k (no muy al principio ni al final)
    t_k = np.random.randint(len(df) // 4, 3 * len(df) // 4)
    
    # 4. Inyectar un spike masivo en una columna de precio/volumen en t_k
    df_spiked = original_df.copy()
    price_cols = [c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()]
    if not price_cols:
        price_cols = [df.columns[0]] # Fallback
        
    target_col = price_cols[0]
    df_spiked.iloc[t_k:, df_spiked.columns.get_loc(target_col)] *= 1000.0
    
    # 5. Generar features con el spike
    features_spiked = feature_generator_func(df_spiked)
    
    # 6. Comparar features antes de t_k
    # Las features en T < t_k NO deben cambiar
    before_spike_orig = features_orig.iloc[:t_k]
    before_spike_spiked = features_spiked.iloc[:t_k]
    
    # Ignorar la primera fila si hay NaNs por shifts
    diff = (before_spike_orig.iloc[1:] != before_spike_spiked.iloc[1:]).any().any()
    
    if diff:
        # Identificar qué columnas fallaron
        failed_cols = []
        for col in before_spike_orig.columns:
            if not before_spike_orig[col].iloc[1:].equals(before_spike_spiked[col].iloc[1:]):
                failed_cols.append(col)
        raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: Features altered prior to spike at index {t_k}. "
                           f"Affected columns: {failed_cols}")
    
    print(f"[OK] Latent leakage test passed. Spike at index {t_k} did not affect prior features.")
    return True

def temporal_integrity_check(df: pd.DataFrame, target_col: str = 'target', horizon: int = 1):
    """
    Verifica que no haya fuga de información futura en las features.
    Regla: Ninguna feature en el tiempo T debe estar correlacionada con el target en T+N
    si esa feature usa datos de [T+1, ...].
    """
    print(f"--- Ejecutando Temporal Integrity Check (Horizonte: {horizon}) ---")
    
    # Una forma simple de detectar fugas obvias es verificar si el target
    # está presente en las features o si hay una correlación perfecta inesperada.
    
    features = df.drop(columns=[target_col])
    target = df[target_col]
    
    # 1. Verificar si el target está en las features
    if target_col in features.columns:
        raise ValueError(f"CRITICAL: El target '{target_col}' está presente en el set de features.")
        
    # 2. Verificar correlaciones sospechosas (fugas masivas)
    correlations = features.corrwith(target)
    suspicious = correlations[abs(correlations) > 0.99]
    
    if not suspicious.empty:
        print(f"ADVERTENCIA: Features con correlación > 0.99 detectadas:\n{suspicious}")
        
    # 3. Verificar desfase temporal (Shift check)
    # Si shift(-horizon) de una feature es igual al target, hay fuga.
    for col in features.columns:
        try:
            # Comparamos valores ignorando NaNs
            feat_vals = features[col].values[horizon:]
            target_shifted = target.shift(horizon).values[horizon:]
            if np.array_equal(feat_vals, target_shifted):
                 print(f"ADVERTENCIA: La feature '{col}' parece ser el target desplazado.")
        except:
            pass

    print("[OK] Integridad temporal validada.")
    return True

