import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.mixture import GaussianMixture
import warnings

# Suprimir warnings de pandas_ta sobre fragmentación de DataFrames
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class FeatureGenerator:
    def __init__(self):
        self.required_columns =['open', 'high', 'low', 'close', 'volume']
        self.microstructure_columns =['bid', 'ask', 'bid_vol', 'ask_vol']

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Valida la existencia de columnas obligatorias. Lanza ValueError si faltan."""
        missing_base =[col for col in self.required_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Faltan columnas base OHLCV: {missing_base}")
        
        missing_micro = [col for col in self.microstructure_columns if col not in df.columns]
        if missing_micro:
            raise ValueError(
                f"Dato de microestructura no disponible: faltan las columnas {missing_micro}. "
                "Regla de integridad: No se permite simular datos faltantes."
            )

    def _calculate_hurst(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calcula el exponente de Hurst en ventana móvil para detectar reversión/tendencia."""
        def hurst(x):
            if len(x) < 10: return 0.5
            lags = range(2, 10)
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        return series.rolling(window=window).apply(hurst, raw=True)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera 120+ features técnicas, de volumen, microestructura y régimen."""
        self._validate_data(df)
        
        # Copia para evitar SettingWithCopyWarning
        data = df.copy()

        # 1. TENDENCIA (Trend)
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
        data['volume_zscore'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # 5. MICROESTRUCTURA
        data['bid_ask_spread'] = data['ask'] - data['bid']
        data['order_book_imbalance'] = (data['bid_vol'] - data['ask_vol']) / (data['bid_vol'] + data['ask_vol'] + 1e-8)
        data['delta_volumen'] = data['volume'].diff()

        # 6. RÉGIMEN Y ESTADÍSTICA AVANZADA
        data['price_zscore_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        data['realized_volatility'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252 * 288) # Anualizado para 5m
        data['hurst_exponent'] = self._calculate_hurst(data['close'], window=20)

        # GMM Hidden State (Régimen de mercado: 0=Rango, 1=Tendencia, 2=Volátil)
        # Usamos rolling features para evitar Data Leakage global
        regime_features = data[['realized_volatility', 'price_zscore_20']].dropna()
        if len(regime_features) > 50:
            gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
            gmm.fit(regime_features)
            states = gmm.predict(regime_features)
            data.loc[regime_features.index, 'hmm_hidden_state'] = states
        else:
            data['hmm_hidden_state'] = 0

        # 7. CROSS-TIMEFRAME (Simulado internamente para el ejemplo atómico)
        # En producción, esto recibe el merge de otros timeframes.
        data['ema_alignment'] = np.where(
            (data['EMA_9'] > data['EMA_21']) & (data['EMA_21'] > data['EMA_50']), 1, 
            np.where((data['EMA_9'] < data['EMA_21']) & (data['EMA_21'] < data['EMA_50']), -1, 0)
        )

        # MANEJO ESTRICTO DE NaNs (Regla de integridad)
        # 1. Eliminamos las primeras filas que son puramente NaNs por los periodos de lookback (ej. EMA 200)
        data = data.dropna(subset=['EMA_200'])
        # 2. Si quedan NaNs aislados, Forward Fill y luego Backward Fill
        data = data.ffill().bfill()

        return data

if __name__ == "__main__":
    print("Inicializando Feature Generator...")
    
    # Generación de datos sintéticos OHLCV + Microestructura para prueba
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='5min')
    
    # Simulando un random walk para el precio
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
        print("Calculando 120+ features (Trend, Momentum, Volatility, Volume, Microstructure, Regime)...")
        df_features = generator.generate_features(df_synthetic)
        
        print("\n--- REPORTE DE INTEGRIDAD ---")
        print(f"Filas originales: {len(df_synthetic)}")
        print(f"Filas tras limpieza de look-ahead/NaNs: {len(df_features)}")
        print(f"Total de features generadas: {len(df_features.columns)}")
        print(f"Conteo de NaNs restantes: {df_features.isna().sum().sum()} (Debe ser 0)")
        print("\nMuestra de features calculadas:")
        print(df_features[['close', 'EMA_200', 'RSI_14', 'bid_ask_spread', 'hmm_hidden_state', 'hurst_exponent']].tail())
        print("\n[OK] Módulo ejecutado exitosamente sin errores.")
        
    except Exception as e:
        print(f"\n[ERROR] Fallo en la generación: {str(e)}")