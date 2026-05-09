"""
Meta-Learner XGBoost con calibración temporal.
Versión 2.0: usa TimeSeriesSplit para evitar fuga de futuro en la calibración.
"""
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import warnings

warnings.filterwarnings("ignore", message=".*is_sparse.*")


class RegimeAwareMetaLearner:
    """
    Meta-Learner que toma las representaciones del TFT y emite probabilidades
    calibradas de PUT/CALL. Incluye explicabilidad mediante SHAP.
    Adaptado para ser sensible al régimen de mercado.
    """

    def __init__(self, calibration_method: str = 'isotonic', random_state: int = 42,
                 n_splits: int = 3):
        """
        Parámetros
        ----------
        calibration_method : 'isotonic' o 'sigmoid'
        random_state : semilla para reproducibilidad
        n_splits : número de divisiones para TimeSeriesSplit
        """
        if calibration_method not in ['isotonic', 'sigmoid']:
            raise ValueError("calibration_method debe ser 'isotonic' o 'sigmoid'")

        self.calibration_method = calibration_method
        self.random_state = random_state
        self.n_splits = n_splits

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

        # Calibración con validación temporal (sin barajar)
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method=self.calibration_method,
            cv=TimeSeriesSplit(n_splits=n_splits)
        )

        self.is_fitted = False
        self.explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo XGBoost y ajusta la calibración de probabilidades
        usando división temporal. Requiere que los datos estén ordenados
        cronológicamente.
        """
        if len(np.unique(y)) != 2:
            raise ValueError("El target 'y' debe ser binario (0 para PUT, 1 para CALL).")

        # Entrenamiento y calibración con TimeSeriesSplit
        self.calibrated_model.fit(X, y)

        # Para SHAP entrenamos una versión sin calibrar sobre todos los datos
        self.base_model.fit(X, y)
        
        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.base_model)
        else:
            self.explainer = None

        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> dict:
        """
        Devuelve probabilidades calibradas para las clases 0 y 1.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado con fit() antes de predecir.")
        probs = self.calibrated_model.predict_proba(X)
        return {
            "p_put": probs[:, 0],
            "p_call": probs[:, 1]
        }

    def get_shap_explanations(self, X: np.ndarray, feature_names: list = None,
                              top_n: int = 5) -> list:
        """
        Calcula los valores SHAP para explicar las predicciones del modelo base.
        Devuelve las top_n features más importantes para cada muestra.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de calcular SHAP.")
        
        if not SHAP_AVAILABLE or self.explainer is None:
            return [["SHAP_NOT_AVAILABLE"] * top_n for _ in range(X.shape[0])]
            
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # clase 1 (CALL)

        explanations = []
        for i in range(X.shape[0]):
            sample_shap = np.abs(shap_values[i])
            top_indices = np.argsort(sample_shap)[-top_n:][::-1]
            if feature_names is not None:
                top_features = [feature_names[idx] for idx in top_indices]
            else:
                top_features = [f"feature_{idx}" for idx in top_indices]
            explanations.append(top_features)
        return explanations


if __name__ == "__main__":
    print("Inicializando Meta-Learner XGBoost con Calibración Temporal...")
    np.random.seed(42)
    NUM_SAMPLES = 1000
    OUTPUT_DIM = 64

    X_synthetic = np.random.randn(NUM_SAMPLES, OUTPUT_DIM)
    y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] * 0.5 + np.random.randn(NUM_SAMPLES) > 0).astype(int)

    # Divisiones temporales manuales para prueba (respetando orden)
    split_idx = int(0.8 * NUM_SAMPLES)
    X_train, X_test = X_synthetic[:split_idx], X_synthetic[split_idx:]
    y_train, y_test = y_synthetic[:split_idx], y_synthetic[split_idx:]

    try:
        meta_learner = XGBoostMetaLearner(calibration_method='isotonic', n_splits=3)
        print(f"Entrenando con {len(X_train)} muestras (orden cronológico)...")
        meta_learner.fit(X_train, y_train)

        print("Generando predicciones calibradas...")
        probs = meta_learner.predict_proba(X_test)

        feature_names = [f"tft_emb_{i}" for i in range(OUTPUT_DIM)]
        top_features = meta_learner.get_shap_explanations(X_test, feature_names=feature_names, top_n=5)

        print("\n--- REPORTE DE SEÑAL (Muestra 0) ---")
        print(f"Probabilidad PUT : {probs['p_put'][0]:.4f}")
        print(f"Probabilidad CALL: {probs['p_call'][0]:.4f}")
        print(f"Top 5 Features (SHAP): {top_features[0]}")

        assert np.allclose(probs['p_put'] + probs['p_call'], 1.0)
        assert np.all((probs['p_call'] >= 0) & (probs['p_call'] <= 1))
        print("\n[OK] Meta-Learner con validación temporal ejecutado correctamente.")

    except Exception as e:
        print(f"\n[ERROR] {e}")