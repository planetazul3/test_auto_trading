import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import shap
import warnings

# Suprimir warnings de SHAP/XGBoost para una salida limpia en consola
warnings.filterwarnings("ignore", message=".*is_sparse.*")

class XGBoostMetaLearner:
    """
    Meta-Learner que toma las representaciones del TFT y emite probabilidades
    calibradas de PUT/CALL. Incluye explicabilidad mediante SHAP.
    """
    def __init__(self, calibration_method: str = 'isotonic', random_state: int = 42):
        if calibration_method not in ['isotonic', 'sigmoid']:
            raise ValueError("calibration_method debe ser 'isotonic' o 'sigmoid'")
            
        self.calibration_method = calibration_method
        self.random_state = random_state
        
        # Modelo base XGBoost configurado para clasificación binaria (0=PUT, 1=CALL)
        self.base_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Wrapper de calibración
        # cv=5 entrena 5 modelos XGBoost internamente y calibra sus salidas
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method=self.calibration_method,
            cv=5
        )
        
        self.is_fitted = False
        self.explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo XGBoost y ajusta la calibración de probabilidades.
        """
        if len(np.unique(y)) != 2:
            raise ValueError("El target 'y' debe ser binario (0 para PUT, 1 para CALL).")
            
        # Entrenar el modelo calibrado (esto entrena el base_model internamente vía CV)
        self.calibrated_model.fit(X, y)
        
        # Para SHAP, necesitamos un modelo base entrenado en todo el dataset
        # ya que CalibratedClassifierCV crea un ensamble de modelos.
        # NOTA DE AUDITORÍA: Los valores SHAP están explicando el modelo base 
        # sin calibrar, no la probabilidad final calibrada. Este es un trade-off 
        # estándar, ya que SHAP no soporta nativamente CalibratedClassifierCV.
        self.base_model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.base_model)
        
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> dict:
        """
        Devuelve las probabilidades calibradas para PUT y CALL.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de predecir.")
            
        # predict_proba devuelve[P(class=0), P(class=1)]
        probs = self.calibrated_model.predict_proba(X)
        
        return {
            "p_put": probs[:, 0],
            "p_call": probs[:, 1]
        }

    def get_shap_explanations(self, X: np.ndarray, feature_names: list = None, top_n: int = 5) -> list:
        """
        Calcula los valores SHAP para explicar las predicciones.
        Devuelve las top_n features más importantes para cada muestra.
        """
        if not self.is_fitted or self.explainer is None:
            raise RuntimeError("El modelo debe ser entrenado antes de calcular SHAP.")
            
        shap_values = self.explainer.shap_values(X)
        
        # Si es clasificación binaria, shap_values puede ser una lista o un array 2D
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Tomamos la clase 1 (CALL)
            
        explanations = []
        for i in range(X.shape[0]):
            # Obtener los índices de las features con mayor impacto absoluto
            sample_shap = np.abs(shap_values[i])
            top_indices = np.argsort(sample_shap)[-top_n:][::-1]
            
            if feature_names is not None:
                top_features = [feature_names[idx] for idx in top_indices]
            else:
                top_features = [f"feature_{idx}" for idx in top_indices]
                
            explanations.append(top_features)
            
        return explanations

if __name__ == "__main__":
    print("Inicializando Meta-Learner XGBoost con Calibración...")
    
    # Simulamos datos de entrada: 1000 muestras, 64 dimensiones (salida del TFT)
    np.random.seed(42)
    NUM_SAMPLES = 1000
    OUTPUT_DIM = 64
    
    X_synthetic = np.random.randn(NUM_SAMPLES, OUTPUT_DIM)
    # Target sintético con algo de relación lineal para que el modelo aprenda algo
    y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] * 0.5 + np.random.randn(NUM_SAMPLES) > 0).astype(int)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)
    
    try:
        # Instanciar y entrenar
        meta_learner = XGBoostMetaLearner(calibration_method='isotonic')
        print(f"Entrenando modelo con {len(X_train)} muestras...")
        meta_learner.fit(X_train, y_train)
        
        # Predecir en test
        print(f"Generando predicciones calibradas para {len(X_test)} muestras...")
        probs = meta_learner.predict_proba(X_test)
        
        # Explicabilidad SHAP
        print("Calculando valores SHAP para interpretabilidad...")
        feature_names =[f"tft_emb_{i}" for i in range(OUTPUT_DIM)]
        top_features = meta_learner.get_shap_explanations(X_test, feature_names=feature_names, top_n=5)
        
        # Mostrar resultados de la primera muestra de test
        print("\n--- REPORTE DE SEÑAL (Muestra 0) ---")
        print(f"Probabilidad PUT (Clase 0):  {probs['p_put'][0]:.4f}")
        print(f"Probabilidad CALL (Clase 1): {probs['p_call'][0]:.4f}")
        print(f"Top 5 Features (SHAP):       {top_features[0]}")
        
        # Verificación de integridad
        assert np.allclose(probs['p_put'] + probs['p_call'], 1.0), "Las probabilidades no suman 1"
        assert np.all((probs['p_call'] >= 0) & (probs['p_call'] <= 1)), "Probabilidades fuera de rango[0,1]"
        
        print("\n[OK] Módulo Meta-Learner ejecutado exitosamente sin errores.")
        
    except Exception as e:
        print(f"\n[ERROR] Fallo en la ejecución del Meta-Learner: {str(e)}")