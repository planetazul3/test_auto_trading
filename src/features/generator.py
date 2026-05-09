"""
Generador de 120+ features técnicas, de volumen, microestructura y régimen.
Versión 2.0: sin data leakage por backward fill, ffill estricto para respear orden temporal.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import njit
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@njit
def _fast_hurst(x: np.ndarray) -> float:
    """
    Hurst exponent vía método de varianza:

        Var(x[t+k] - x[t]) ∝ k^(2H)

    Equivalentemente ``log(sqrt(std)) = 0.25 * log(Var)`` y la pendiente
    de ``log(sqrt(std))`` vs ``log(k)`` es ``H/2``; por eso se multiplica
    por 2 al final. Se clipa a [0, 1] para conservar la interpretación del
    Blueprint §4.1 (H<0.5 mean-reverting, H>0.5 trending).
    """
    if len(x) < 10:
        return 0.5
    lags = np.arange(2, 10)
    log_lags = np.log(lags)
    log_tau = np.empty(len(lags))

    for i in range(len(lags)):
        lag = lags[i]
        diff = x[lag:] - x[:-lag]
        std_val = np.std(diff)
        if std_val <= 0.0:
            return 0.5
        log_tau[i] = 0.5 * np.log(std_val)

    n = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xx = np.sum(log_lags**2)
    sum_xy = np.sum(log_lags * log_tau)

    denom = (n * sum_xx - sum_x**2)
    if denom == 0:
        return 0.5
    slope = (n * sum_xy - sum_x * sum_y) / denom
    h = slope * 2.0
    if h < 0.0:
        return 0.0
    if h > 1.0:
        return 1.0
    return h


@njit
def _rolling_hurst_numba(values: np.ndarray, window: int) -> np.ndarray:
    """Aplica _fast_hurst en una ventana móvil de forma eficiente."""
    n = len(values)
    res = np.empty(n)
    res[:] = np.nan
    for i in range(window - 1, n):
        res[i] = _fast_hurst(values[i - window + 1 : i + 1])
    return res


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
        Z-Score estrictamente causal (Blueprint §3.1).

        El valor en T se normaliza contra el resumen de [T-window, T-1]: las
        estadísticas se calculan con ``.rolling(window).<stat>().shift(1)`` para
        que T no influya en su propia normalización. Si ``mad_fallback`` está
        activo, ventanas con std degenerada caen a la Median Absolute Deviation
        (mediana de |x - median(x)|) escalada por 1.4826 para mantener
        consistencia con la desviación estándar bajo distribución normal.
        """
        roll = series.rolling(window=window, min_periods=window)
        rolling_mean = roll.mean().shift(1)
        rolling_std = roll.std().shift(1)

        if self.mad_fallback:
            rolling_mad = roll.apply(
                lambda x: np.median(np.abs(x - np.median(x))) * 1.4826,
                raw=True,
            ).shift(1)
            rolling_std = pd.Series(
                np.where(rolling_std < 1e-8, rolling_mad + 1e-8, rolling_std),
                index=series.index,
            )

        return (series - rolling_mean) / (rolling_std + 1e-8)

    def _calculate_hurst(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Hurst exponent en ventana móvil para detectar regímenes mediante Numba."""
        hurst_values = _rolling_hurst_numba(series.values, window)
        return pd.Series(hurst_values, index=series.index)

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera todas las features respetando el orden temporal.
        Los NaN originados por los periodos de lookback se rellenan SOLO hacia adelante (ffill).
        """
        self._validate_data(df)
        data = df.copy()
        
        # Diccionario para coleccionar nuevas columnas y evitar fragmentación
        f = {}

        # 1. TENDENCIA
        f['EMA_9'] = data.ta.ema(length=9)
        f['EMA_21'] = data.ta.ema(length=21)
        f['EMA_50'] = data.ta.ema(length=50)
        f['EMA_200'] = data.ta.ema(length=200)
        f['SMA_20'] = data.ta.sma(length=20)
        f['DEMA_20'] = data.ta.dema(length=20)
        f['TEMA_20'] = data.ta.tema(length=20)
        
        # Ichimoku con lookahead=False: pandas_ta por defecto desplaza Senkou
        # Span A/B 26 períodos hacia adelante, lo que introduce look-ahead.
        ichi = data.ta.ichimoku(lookahead=False)
        if ichi is not None:
            # ichi es una tupla (visible, forward); sólo usamos la parte visible.
            f['ICHIMOKU'] = ichi[0]

        # 2. MOMENTUM
        f['RSI_7'] = data.ta.rsi(length=7)
        f['RSI_14'] = data.ta.rsi(length=14)
        f['RSI_21'] = data.ta.rsi(length=21)
        f['MACD'] = data.ta.macd(fast=12, slow=26, signal=9)
        f['STOCH'] = data.ta.stoch(k=14, d=3)
        f['WILLR'] = data.ta.willr(length=14)
        f['CCI'] = data.ta.cci(length=14)
        f['MFI'] = data.ta.mfi(length=14)
        f['ROC'] = data.ta.roc(length=10)

        # 3. VOLATILIDAD
        f['ATR_7'] = data.ta.atr(length=7)
        f['ATR_14'] = data.ta.atr(length=14)
        atr_long = data.ta.atr(length=100)
        # Blueprint §4.1: ATR Volatility Ratio = ATR_short / ATR_long.
        f['atr_volatility_ratio'] = (f['ATR_7'] / (atr_long + 1e-8)).rename(
            'atr_volatility_ratio'
        )
        f['BBANDS'] = data.ta.bbands(length=20, std=2)
        f['KC'] = data.ta.kc(length=20)
        f['DONCHIAN'] = data.ta.donchian(lower_length=20, upper_length=20)

        # 4. VOLUMEN
        f['OBV'] = data.ta.obv()
        f['VWAP'] = data.ta.vwap()
        f['CMF'] = data.ta.cmf()
        
        if self.use_causal_zscore:
            f['volume_zscore'] = self.safe_causal_zscore(data['volume'], self.window).rename('volume_zscore')
        else:
            vol_roll = data['volume'].rolling(20)
            f['volume_zscore'] = ((data['volume'] - vol_roll.mean()) / vol_roll.std()).rename('volume_zscore')
        
        f['volume_ratio'] = (data['volume'] / data['volume'].rolling(20).mean()).rename('volume_ratio')

        # 5. MICROESTRUCTURA
        f['bid_ask_spread'] = (data['ask'] - data['bid']).rename('bid_ask_spread')
        f['order_book_imbalance'] = ((data['bid_vol'] - data['ask_vol']) / (data['bid_vol'] + data['ask_vol'] + 1e-8)).rename('order_book_imbalance')
        f['delta_volumen'] = data['volume'].diff().rename('delta_volumen')

        # 6. RÉGIMEN Y ESTADÍSTICA AVANZADA
        if self.use_causal_zscore:
            f['price_zscore_20'] = self.safe_causal_zscore(data['close'], self.window).rename('price_zscore_20')
        else:
            close_roll = data['close'].rolling(20)
            f['price_zscore_20'] = ((data['close'] - close_roll.mean()) / close_roll.std()).rename('price_zscore_20')
            
        f['realized_volatility'] = (data['close'].pct_change().rolling(20).std() * np.sqrt(252 * 288)).rename('realized_volatility')
        f['hurst_exponent'] = self._calculate_hurst(data['close'], window=20).rename('hurst_exponent')

        # Blueprint §4.1 #2: Distance from 200-SMA, normalizada.
        sma200 = data['close'].rolling(window=200, min_periods=200).mean()
        f['dist_from_sma200'] = ((data['close'] - sma200) / (sma200 + 1e-8)).rename(
            'dist_from_sma200'
        )

        # GMM Hidden State con etiquetas estables.
        #
        # Problema: GMM(random_state=42) puede asignar permutaciones distintas
        # de cluster_id en cada reentrenamiento sobre ventanas distintas.
        # Solución: tras cada fit, reordenamos los clusters por la media de la
        # `realized_volatility` (col 0) para forzar un mapping determinista
        # 0 → baja vol, 1 → media vol, 2 → alta vol.
        regime_features = pd.DataFrame({
            'realized_volatility': f['realized_volatility'],
            'price_zscore_20': f['price_zscore_20']
        })
        states = np.full(len(data), np.nan, dtype=np.float64)
        window_size = 100
        update_freq = 20

        gmm = None
        label_map = np.array([0, 1, 2], dtype=np.int64)
        for i in range(window_size, len(data), update_freq):
            train_window = regime_features.iloc[i - window_size:i].dropna()
            if len(train_window) > 50:
                try:
                    new_gmm = GaussianMixture(
                        n_components=3, covariance_type='full', random_state=42
                    )
                    new_gmm.fit(train_window)
                    # Orden estable: ranking por media de realized_volatility.
                    vol_means = new_gmm.means_[:, 0]
                    order = np.argsort(vol_means)
                    label_map = np.argsort(order)  # cluster_id -> regime_id
                    gmm = new_gmm
                except Exception:
                    pass

            if gmm is not None:
                end_idx = min(i + update_freq, len(data))
                batch_features = regime_features.iloc[i:end_idx]
                valid_mask = ~batch_features.isna().any(axis=1)
                if valid_mask.any():
                    raw = gmm.predict(batch_features[valid_mask])
                    states[i:end_idx][valid_mask.values] = label_map[raw]

        f['hmm_hidden_state'] = pd.Series(
            states, index=data.index, name='hmm_hidden_state'
        )

        # 7. CROSS-TIMEFRAME SIMULADO
        f['ema_alignment'] = pd.Series(np.where(
            (f['EMA_9'] > f['EMA_21']) & (f['EMA_21'] > f['EMA_50']), 1,
            np.where((f['EMA_9'] < f['EMA_21']) & (f['EMA_21'] < f['EMA_50']), -1, 0)
        ), index=data.index, name='ema_alignment')

        # Combinación FINAL en una sola operación para evitar PerformanceWarning.
        feature_list = [v for v in f.values() if v is not None]
        data = pd.concat([data] + feature_list, axis=1)

        # Algunos indicadores de pandas_ta (p. ej. ichimoku) mutan ``data`` en
        # sitio agregando columnas; combinado con el dict ``f`` se introducen
        # nombres repetidos. Conservamos la primera ocurrencia.
        data = data.loc[:, ~data.columns.duplicated()]

        # Limpieza FINAL sin backward fill.
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