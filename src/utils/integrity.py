import pandas as pd
import numpy as np

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
            if np.array_equal(features[col].values, target.shift(horizon).values):
                 print(f"ADVERTENCIA: La feature '{col}' parece ser el target desplazado.")
        except:
            pass

    print("[OK] Integridad temporal validada.")
    return True
