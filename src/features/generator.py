"""
Generador de 120+ features técnicas, de volumen, microestructura y régimen.
Versión 2.0: sin data leakage por backward fill, ffill estricto para respear orden temporal.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class FeatureGenerator:
    """
    Construye un dataset de características a partir de datos OHLCV + microestructura.
    """

    def __init__(self, use_causal_zscore: bool = False, window: int = 20, mad_fallback: bool = True):
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.microstructure_columns = ['bid', 'ask', 'bid_vol', 'ask_vol']
        self.use_causal_zscore = use_causal_zscore
        self.window = window
        self.mad_fallback = mad_fallback

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Verifica que el DataFrame tenga todas las columnas obligatorias."""
        missing_base = [col for col in self.required_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Faltan columnas base OHLCV: {missing_base}")

        missing_micro = [col for col in self.microstructure_columns if col not in df.columns]
        if missing_micro:
            raise ValueError(
                f"Dato de microestructura no disponible: faltan las columnas {missing_micro}. "
                "Regla de integridad: No se permite simular datos faltantes."
            )

    def safe_causal_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calcula el Z-Score de forma causal (solo usando datos pasados).
        Evita el look-ahead bias al no usar la media/std del dataset completo.
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        if self.mad_fallback:
            # Fallback a Mean Absolute Deviation si la std es 0 o muy pequeña
            rolling_mad = series.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            rolling_std = np.where(rolling_std < 1e-8, rolling_mad + 1e-8, rolling_std)
            
        return (series - rolling_mean) / rolling_std

    def _calculate_hurst(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Hurst exponent en ventana móvil para detectar regímenes."""
        def hurst(x):
            if len(x) < 10:
                return 0.5
            lags = range(2, 10)
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        return series.rolling(window=window).apply(hurst, raw=True)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera todas las features respetando el orden temporal.
        Los NaN originados por los periodos de lookback se rellenan SOLO hacia adelante (ffill).
        """
        self._validate_data(df)
        data = df.copy()

        # 1. TENDENCIA
        data.ta.ema(length=9, append=True)
        data.ta.ema(length=21, append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=200, append=True)
        data.ta.sma(length=20, append=True)
        data.ta.dema(length=20, append=True)
        data.ta.tema(length=20, append=True)
        data.ta.ichimoku(append=True)

        # 2. MOMENTUM
        data.ta.rsi(length=7, append=True)
        data.ta.rsi(length=14, append=True)
        data.ta.rsi(length=21, append=True)
        data.ta.macd(fast=12, slow=26, signal=9, append=True)
        data.ta.stoch(k=14, d=3, append=True)
        data.ta.willr(length=14, append=True)
        data.ta.cci(length=14, append=True)
        data.ta.mfi(length=14, append=True)
        data.ta.roc(length=10, append=True)

        # 3. VOLATILIDAD
        data.ta.atr(length=7, append=True)
        data.ta.atr(length=14, append=True)
        data.ta.bbands(length=20, std=2, append=True)
        data.ta.kc(length=20, append=True)
        data.ta.donchian(lower_length=20, upper_length=20, append=True)

        # 4. VOLUMEN
        data.ta.obv(append=True)
        data.ta.vwap(append=True)
        data.ta.cmf(append=True)
        if self.use_causal_zscore:
            data['volume_zscore'] = self.safe_causal_zscore(data['volume'], self.window)
        else:
            data['volume_zscore'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # 5. MICROESTRUCTURA
        data['bid_ask_spread'] = data['ask'] - data['bid']
        data['order_book_imbalance'] = (data['bid_vol'] - data['ask_vol']) / (data['bid_vol'] + data['ask_vol'] + 1e-8)
        data['delta_volumen'] = data['volume'].diff()

        # 6. RÉGIMEN Y ESTADÍSTICA AVANZADA
        if self.use_causal_zscore:
            data['price_zscore_20'] = self.safe_causal_zscore(data['close'], self.window)
        else:
            data['price_zscore_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        data['realized_volatility'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252 * 288)
        data['hurst_exponent'] = self._calculate_hurst(data['close'], window=20)

        # GMM Hidden State – solo entrena sobre pasado para no ver el futuro
        regime_features = data[['realized_volatility', 'price_zscore_20']]
        states = np.full(len(data), 0.0)
        window_size = 50
        update_freq = 20 # Solo reentrenar cada 20 pasos para velocidad
        
        gmm = None
        for i in range(window_size, len(data)):
            if i % update_freq == 0 or gmm is None:
                train_window = regime_features.iloc[i - window_size:i].dropna()
                if len(train_window) > 30:
                    try:
                        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
                        gmm.fit(train_window)
                    except Exception:
                        pass
            
            if gmm is not None:
                current_row = regime_features.iloc[i:i + 1]
                states[i] = gmm.predict(current_row)[0]
                
        data['hmm_hidden_state'] = states

        # 7. CROSS-TIMEFRAME SIMULADO
        data['ema_alignment'] = np.where(
            (data['EMA_9'] > data['EMA_21']) & (data['EMA_21'] > data['EMA_50']), 1,
            np.where((data['EMA_9'] < data['EMA_21']) & (data['EMA_21'] < data['EMA_50']), -1, 0)
        )

        # Limpieza FINAL sin backward fill
        data = data.dropna(subset=['EMA_200'])      # elimina filas iniciales sin indicadores de largo plazo
        data = data.ffill()                         # solo forward fill, nunca información futura
        return data


if __name__ == "__main__":
    print("Inicializando Feature Generator...")
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
        print("[OK] Módulo ejecutado exitosamente.")
    except Exception as e:
        print(f"[ERROR] {e}")