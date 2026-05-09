import numpy as np
from sklearn.isotonic import IsotonicRegression
from collections import deque

class LowLatencyRollingIsotonicCalibrator:
    """
    Calibrador de probabilidades usando regresión isotónica con ventana rodante.
    Optimizado para baja latencia mediante un buffer circular.
    """
    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.margins = deque(maxlen=window_size)
        self.labels = deque(maxlen=window_size)
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def add_observation(self, margin: float, label: int):
        """
        Añade una nueva observación (logit/margen y etiqueta real) al buffer.
        """
        self.margins.append(float(margin))
        self.labels.append(int(label))
        self.is_fitted = False # Marcar como no entrenado tras nueva data

    def update_calibration_curve(self):
        """
        Entrena el modelo de regresión isotónica con los datos actuales del buffer.
        """
        if len(self.margins) < 100:
            # No hay suficientes datos para una calibración robusta
            return

        x = np.array(self.margins)
        y = np.array(self.labels)
        
        # La regresión isotónica requiere x ordenado o se encarga internamente
        # pero es más estable si hay suficientes valores únicos.
        self.model.fit(x, y)
        self.is_fitted = True

    def calibrate_signal(self, margin: float) -> float:
        """
        Transforma un margen crudo (logit) en una probabilidad calibrada [0, 1].
        """
        if not self.is_fitted:
            # Si no está calibrado, usamos una sigmoide simple como fallback
            return 1.0 / (1.0 + np.exp(-margin))
        
        return float(self.model.predict([margin])[0])

if __name__ == "__main__":
    calibrator = LowLatencyRollingIsotonicCalibrator(window_size=1000)
    
    # Simular datos
    for _ in range(500):
        m = np.random.randn()
        l = 1 if m + np.random.randn() * 0.5 > 0 else 0
        calibrator.add_observation(m, l)
    
    calibrator.update_calibration_curve()
    
    test_margin = 1.5
    prob = calibrator.calibrate_signal(test_margin)
    print(f"Margen: {test_margin} -> Probabilidad Calibrada: {prob:.4f}")
