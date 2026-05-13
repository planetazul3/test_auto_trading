"""Meta-Learner XGBoost para clasificación de regímenes de mercado.

Mejoras respecto a la versión previa:

* **API SHAP moderna**: maneja tanto el array 3D ``(n, f, k)`` de SHAP
  ≥0.42 como la lista legacy por clase.
* **CV temporal real**: ``n_splits`` se usa para construir
  ``TimeSeriesSplit`` y reportar mlogloss cross-validado.
* **Validación de etiquetas**: ``y`` debe contener exclusivamente los
  valores ``{0, 1, 2}`` (no se permiten huecos).
* **Soporte de desbalanceo**: ``class_weight='balanced'`` o pesos
  explícitos via ``sample_weight``.
* **Warnings locales**: nada de ``warnings.filterwarnings`` global al
  importar el módulo.
* **Etiquetas de régimen configurables** (no hardcoded en el ensemble).
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependencia opcional
    shap = None
    SHAP_AVAILABLE = False


DEFAULT_REGIME_LABELS: tuple[str, ...] = ("LOW_VOL", "TRENDING", "HIGH_VOL")


def _normalize_shap_values(
    raw: Any, regime_idx: int, sample_idx: int
) -> np.ndarray:
    """Devuelve el vector SHAP ``(num_features,)`` para una muestra y clase.

    Soporta:
    * SHAP <0.42: ``list[ndarray (n,f)]`` indexado por clase.
    * SHAP ≥0.42: ``ndarray (n, f, k)``.
    """
    if isinstance(raw, list):
        out: np.ndarray = np.asarray(raw[regime_idx][sample_idx], dtype=np.float64)
        return out
    arr: np.ndarray = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3:
        slc: np.ndarray = arr[sample_idx, :, regime_idx]
        return slc
    if arr.ndim == 2:
        # Modelo binario: una única matriz (n, f).
        row: np.ndarray = arr[sample_idx]
        return row
    raise ValueError(f"unsupported SHAP value shape: {arr.shape}")


class RegimeAwareMetaLearner:
    """Clasificador XGBoost multi-clase (3 regímenes) con SHAP opcional."""

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_splits: int = 3,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        n_jobs: int = -1,
        regime_labels: Sequence[str] = DEFAULT_REGIME_LABELS,
        class_weight: Optional[str | dict[int, float]] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for TimeSeriesSplit")
        if len(regime_labels) != 3:
            raise ValueError("regime_labels must contain exactly 3 entries")
        self.random_state = random_state
        self.n_splits = n_splits
        self.regime_labels: tuple[str, ...] = tuple(regime_labels)
        self.class_weight = class_weight

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.is_fitted = False
        self.explainer: Optional["shap.TreeExplainer"] = None
        self.cv_mlogloss_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_labels(y: np.ndarray) -> None:
        if y.ndim != 1:
            raise ValueError("y must be 1-D")
        valid = {0, 1, 2}
        actual = set(np.unique(y).tolist())
        if not actual.issubset(valid):
            raise ValueError(
                f"y must contain only labels in {sorted(valid)}; got {sorted(actual)}"
            )

    def _compute_sample_weight(
        self, y: np.ndarray, sample_weight: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if sw.shape != y.shape:
                raise ValueError("sample_weight must match y shape")
            return sw
        if self.class_weight is None:
            return None
        if self.class_weight == "balanced":
            return cast(np.ndarray, compute_sample_weight("balanced", y))
        if isinstance(self.class_weight, dict):
            weights: np.ndarray = np.array([float(self.class_weight[int(c)]) for c in y])
            return weights
        raise ValueError(
            "class_weight must be None, 'balanced' or a dict[int, float]"
        )

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "RegimeAwareMetaLearner":
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self._validate_labels(y)
        sw = self._compute_sample_weight(y, sample_weight)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.model.fit(X, y, sample_weight=sw)
            if SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model)
        self.is_fitted = True
        return self

    def cross_val_mlogloss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calcula mlogloss en ``n_splits`` folds temporales."""
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self._validate_labels(y)
        sw_full = self._compute_sample_weight(y, sample_weight)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        losses: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            sw_tr = None if sw_full is None else sw_full[train_idx]
            est = xgb.XGBClassifier(**self.model.get_params())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                est.fit(X_tr, y_tr, sample_weight=sw_tr)
            proba = est.predict_proba(X_va)
            losses.append(_mlogloss(y_va, proba))
        self.cv_mlogloss_ = np.asarray(losses, dtype=np.float64)
        return self.cv_mlogloss_

    def predict_regime_probs(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted; call .fit() first")
        return cast(np.ndarray, self.model.predict_proba(np.asarray(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted; call .fit() first")
        return cast(np.ndarray, self.model.predict(np.asarray(X)))

    # ------------------------------------------------------------------
    # Importancia / explicabilidad
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return np.asarray(self.model.feature_importances_, dtype=np.float64)

    def _shap_top_features(
        self,
        X: np.ndarray,
        regime_indices: Iterable[int],
        top_n: int,
        feature_names: Optional[Sequence[str]],
    ) -> list[list[str]]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        if not SHAP_AVAILABLE or self.explainer is None:
            return [["SHAP_NOT_AVAILABLE"] * top_n for _ in range(X.shape[0])]
        raw = self.explainer.shap_values(X)
        explanations: list[list[str]] = []
        regime_list = list(regime_indices)
        for i, regime_idx in enumerate(regime_list):
            shap_i = _normalize_shap_values(raw, int(regime_idx), i)
            top_idx = np.argsort(np.abs(shap_i))[-top_n:][::-1]
            if feature_names is not None:
                explanations.append([feature_names[int(j)] for j in top_idx])
            else:
                explanations.append([f"f_{int(j)}" for j in top_idx])
        return explanations

    def get_regime_explanation(
        self,
        X: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        top_n: int = 3,
    ) -> list[list[str]]:
        probs = self.predict_regime_probs(X)
        regimes = np.argmax(probs, axis=1)
        return self._shap_top_features(X, regimes, top_n, feature_names)

    def get_shap_explanations(
        self,
        X: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        top_n: int = 5,
        regime: Optional[int] = None,
    ) -> list[list[str]]:
        """SHAP top-features. Si ``regime`` es ``None``, usa la clase
        modal por muestra; si se especifica, fija la clase a explicar.
        """
        if regime is None:
            probs = self.predict_regime_probs(X)
            regimes = np.argmax(probs, axis=1)
        else:
            if regime not in (0, 1, 2):
                raise ValueError("regime must be 0, 1 or 2")
            regimes = np.full(X.shape[0], regime, dtype=int)
        return self._shap_top_features(X, regimes, top_n, feature_names)


def _mlogloss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    n, k = y_proba.shape
    y_oh = np.zeros_like(y_proba)
    y_oh[np.arange(n), y_true] = 1.0
    clipped = np.clip(y_proba, eps, 1.0 - eps)
    return float(-np.sum(y_oh * np.log(clipped)) / n)


__all__ = ["DEFAULT_REGIME_LABELS", "RegimeAwareMetaLearner"]
