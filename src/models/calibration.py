"""Calibrador isotónico rolling de baja latencia.

Funcionalidad:

* Buffer rodante FIFO de margins/labels (deque).
* Re-fit periódico via ``IsotonicRegression`` (PAVA, O(N)).
* Inferencia O(log N) con búsqueda binaria interpolada en Numba.
* Re-entrenamiento en background **race-free**: la guarda se toma bajo
  lock antes de spawnar el thread, evitando dos refits simultáneos que
  corrompían la curva.
* Snapshot atómico de la curva: leer/asignar ``(x_thresholds, y_values)``
  como una sola tupla previene tuple-tearing entre el thread de update y
  ``calibrate_signal``.
* Métricas de calidad expuestas: Brier score, ECE (Expected Calibration
  Error) y diagnostic ``calibration_curve``.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np
from numba import njit
from sklearn.isotonic import IsotonicRegression


@njit(fastmath=True, cache=True)
def fast_isotonic_inference(
    x_test: float, x_thresholds: np.ndarray, y_values: np.ndarray
) -> float:
    """Inferencia O(log N) con interpolación lineal entre escalones.

    Coincide con ``IsotonicRegression.predict`` cuando ``out_of_bounds='clip'``.
    """
    n = x_thresholds.shape[0]
    if n == 0:
        return 0.5
    if x_test <= x_thresholds[0]:
        return float(y_values[0])
    if x_test >= x_thresholds[-1]:
        return float(y_values[-1])

    low = 0
    high = n - 1
    while high - low > 1:
        mid = (low + high) // 2
        if x_thresholds[mid] <= x_test:
            low = mid
        else:
            high = mid

    x_lo = x_thresholds[low]
    x_hi = x_thresholds[low + 1]
    y_lo = y_values[low]
    y_hi = y_values[low + 1]
    if x_hi == x_lo:
        return float(y_lo)
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    return float(y_lo + slope * (x_test - x_lo))


_NEUTRAL_CURVE: Tuple[np.ndarray, np.ndarray] = (
    np.array([-1e9, 1e9], dtype=np.float64),
    np.array([0.5, 0.5], dtype=np.float64),
)


class LowLatencyRollingIsotonicCalibrator:
    """Calibrador isotónico con ventana rodante y refit en background.

    Parameters
    ----------
    window_size:
        Máximo de observaciones en el buffer FIFO.
    min_observations:
        Umbral mínimo para fittear la primera vez (default 100).
        Parametrizable para entornos con poca historia inicial.
    """

    def __init__(self, window_size: int = 5000, min_observations: int = 100) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if min_observations <= 1:
            raise ValueError("min_observations must be > 1")
        self.window_size = int(window_size)
        self.min_observations = int(min_observations)

        self.margins: deque[float] = deque(maxlen=window_size)
        self.labels: deque[int] = deque(maxlen=window_size)
        self.model: IsotonicRegression = IsotonicRegression(out_of_bounds="clip")

        # Curva como una sola tupla → lecturas atómicas en Python (GIL).
        self._curve: Tuple[np.ndarray, np.ndarray] = _NEUTRAL_CURVE

        self.is_fitted = False
        self._lock = threading.Lock()
        # Guarda race-free: se chequea/setea bajo el mismo lock.
        self._update_in_progress = False

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def add_observation(self, margin: float, label: int) -> None:
        with self._lock:
            self.margins.append(float(margin))
            self.labels.append(int(label))

    def reset(self) -> None:
        """Vacía el buffer y resetea la curva (útil en walk-forward)."""
        with self._lock:
            self.margins.clear()
            self.labels.clear()
            self._curve = _NEUTRAL_CURVE
            self.is_fitted = False

    @property
    def n_observations(self) -> int:
        with self._lock:
            return len(self.margins)

    @property
    def curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Snapshot atómico ``(x_thresholds, y_values)``."""
        return self._curve

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def update_calibration_curve(self) -> bool:
        """Re-fit síncrono. Devuelve ``True`` si la curva se actualizó."""
        with self._lock:
            if len(self.margins) < self.min_observations:
                return False
            x = np.fromiter(self.margins, dtype=np.float64, count=len(self.margins))
            y = np.fromiter(self.labels, dtype=np.float64, count=len(self.labels))

        new_model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        x_th = np.asarray(new_model.X_thresholds_, dtype=np.float64).copy()
        y_th = np.asarray(new_model.y_thresholds_, dtype=np.float64).copy()

        with self._lock:
            self.model = new_model
            # Asignación de tupla atómica respecto a lecturas concurrentes.
            self._curve = (x_th, y_th)
            self.is_fitted = True
        return True

    def update_in_background(self) -> bool:
        """Spawna refit en thread. Race-free: si ya hay uno corriendo, no-op.

        Devuelve ``True`` si se spawnó un nuevo thread.
        """
        with self._lock:
            if self._update_in_progress:
                return False
            self._update_in_progress = True

        def _target() -> None:
            try:
                self.update_calibration_curve()
            finally:
                with self._lock:
                    self._update_in_progress = False

        threading.Thread(target=_target, daemon=True).start()
        return True

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def calibrate_signal(self, margin: float) -> float:
        """Inferencia O(log N) con snapshot atómico de la curva."""
        if not self.is_fitted:
            # Fallback sigmoide hasta el primer fit.
            return float(1.0 / (1.0 + np.exp(-float(margin))))
        x_th, y_th = self._curve  # lectura atómica (Python tuple ref)
        return float(fast_isotonic_inference(float(margin), x_th, y_th))

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    def brier_score(
        self,
        margins: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> float:
        """Brier score sobre la ventana actual (o un set externo)."""
        m, y = self._eval_set(margins, labels)
        if m.size == 0:
            return float("nan")
        p = np.array([self.calibrate_signal(float(v)) for v in m], dtype=np.float64)
        return float(np.mean((p - y) ** 2))

    def expected_calibration_error(
        self,
        margins: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error con binning equi-ancho en [0,1]."""
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        m, y = self._eval_set(margins, labels)
        if m.size == 0:
            return float("nan")
        p = np.array([self.calibrate_signal(float(v)) for v in m], dtype=np.float64)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = p.shape[0]
        for k in range(n_bins):
            lo, hi = bins[k], bins[k + 1]
            mask = (p >= lo) & (p < hi) if k < n_bins - 1 else (p >= lo) & (p <= hi)
            if not mask.any():
                continue
            conf = float(p[mask].mean())
            acc = float(y[mask].mean())
            ece += (mask.sum() / n) * abs(conf - acc)
        return float(ece)

    def _eval_set(
        self,
        margins: Optional[np.ndarray],
        labels: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if margins is None or labels is None:
            with self._lock:
                m = np.fromiter(self.margins, dtype=np.float64, count=len(self.margins))
                y = np.fromiter(self.labels, dtype=np.float64, count=len(self.labels))
        else:
            m = np.asarray(margins, dtype=np.float64)
            y = np.asarray(labels, dtype=np.float64)
        if m.shape != y.shape:
            raise ValueError("margins and labels must share shape")
        return m, y


__all__ = ["LowLatencyRollingIsotonicCalibrator", "fast_isotonic_inference"]
