"""
Meta-Learner XGBoost para clasificación de regímenes de mercado.
"""
import numpy as np
import xgboost as xgb
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import warnings

warnings.filterwarnings("ignore", message=".*is_sparse.*")


class RegimeAwareMetaLearner:
    """
    Meta-Learner que actúa como clasificador de regímenes de mercado.
    Usa XGBoost con salida multi-clase (3 regímenes) para rutear señales
    y ajustar el sizing dinámicamente.
    """

    def __init__(self, random_state: int = 42, n_splits: int = 3):
        """
        Configuración multi-clase para 3 regímenes:
        0: Low Volatility / Mean Reversion
        1: Trending / Momentum
        2: High Volatility / Crash-Risk
        """
        self.random_state = random_state
        self.n_splits = n_splits

        self.model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        self.is_fitted = False
        self.explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el clasificador de regímenes. 
        'y' debe contener labels [0, 1, 2].
        """
        if len(np.unique(y)) > 3:
            raise ValueError("El target 'y' debe tener máximo 3 clases de régimen.")

        self.model.fit(X, y)
        
        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = None

        self.is_fitted = True

    def predict_regime_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Devuelve un vector de probabilidades [p0, p1, p2] para cada muestra.
        """
        if not self.is_fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        return self.model.predict_proba(X)

    def get_regime_explanation(self, X: np.ndarray, feature_names: list = None) -> list:
        """
        Explica por qué se eligió un régimen específico.
        """
        if not self.is_fitted or not SHAP_AVAILABLE:
            return []
        
        # Simplificado para brevedad: explicar la clase con mayor prob
        probs = self.predict_regime_probs(X)
        main_regime = np.argmax(probs, axis=1)
        
        shap_values = self.explainer.shap_values(X)
        # shap_values es una lista para multi-clase
        
        explanations = []
        for i in range(X.shape[0]):
            regime_idx = main_regime[i]
            sample_shap = np.abs(shap_values[regime_idx][i])
            top_indices = np.argsort(sample_shap)[-3:][::-1]
            if feature_names:
                explanations.append([feature_names[idx] for idx in top_indices])
            else:
                explanations.append([f"f_{idx}" for idx in top_indices])
        return explanations

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
    print("Inicializando Meta-Learner XGBoost (clasificador de regímenes)...")
    np.random.seed(42)
    NUM_SAMPLES = 1000
    OUTPUT_DIM = 64
    NUM_REGIMES = 3

    X_synthetic = np.random.randn(NUM_SAMPLES, OUTPUT_DIM)
    # Etiquetas multi-clase [0, 1, 2] = (Low Vol, Trending, High Vol)
    y_synthetic = np.random.randint(0, NUM_REGIMES, size=NUM_SAMPLES)

    # Divisiones temporales manuales (respetando orden cronológico)
    split_idx = int(0.8 * NUM_SAMPLES)
    X_train, X_test = X_synthetic[:split_idx], X_synthetic[split_idx:]
    y_train, _y_test = y_synthetic[:split_idx], y_synthetic[split_idx:]

    try:
        meta_learner = RegimeAwareMetaLearner(n_splits=3)
        print(f"Entrenando con {len(X_train)} muestras (orden cronológico)...")
        meta_learner.fit(X_train, y_train)

        print("Generando probabilidades por régimen...")
        regime_probs = meta_learner.predict_regime_probs(X_test)

        feature_names = [f"tft_emb_{i}" for i in range(OUTPUT_DIM)]
        top_features = meta_learner.get_shap_explanations(X_test, feature_names=feature_names, top_n=5)

        regime_labels = ["LOW_VOL", "TRENDING", "HIGH_VOL"]
        print("\n--- REPORTE DE RÉGIMEN (Muestra 0) ---")
        for idx, label in enumerate(regime_labels):
            print(f"  P({label}): {regime_probs[0, idx]:.4f}")
        print(f"Top 5 Features (SHAP): {top_features[0]}")

        assert regime_probs.shape == (len(X_test), NUM_REGIMES)
        assert np.allclose(regime_probs.sum(axis=1), 1.0, atol=1e-6)
        assert np.all((regime_probs >= 0) & (regime_probs <= 1))
        print("\n[OK] Meta-Learner ejecutado correctamente.")

    except Exception as e:
        print(f"\n[ERROR] {e}")