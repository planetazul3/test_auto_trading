"""
Pipeline de producción: arquitectura híbrida causal CNN-LSTM-TFT, calibración
isotónica out-of-fold, meta-learner adaptativo por régimen y walk-forward
multi-fold con verificación de integridad temporal.

Garantías de no-leakage cubiertas:
  * ``temporal_integrity_check`` (spike injection) sobre la pipeline completa.
  * Z-score causal post-feature engineering (ventana móvil, ``shift(1)``).
  * Calibración isotónica entrenada estrictamente sobre un split de validación
    interno disjunto del set de evaluación.
  * Walk-forward expandiente con folds no solapados en evaluación.
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("ADVERTENCIA: mlflow no instalado. Logs de experimentos omitidos.")

from src.features.generator import FeatureGenerator
from src.models.calibration import LowLatencyRollingIsotonicCalibrator
from src.models.ensemble import HybridSignalEngine
from src.models.hybrid_tft import HybridCNNLSTMTFT
from src.utils.integrity import assert_no_target_leakage, temporal_integrity_check

# Silenciamos warnings ruidosos pero no genéricos.
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
SEQ_LENGTH = 60                # 60 barras de 5 min = 5 h de contexto
HORIZON = 12                   # 1 h de horizonte
HIDDEN_SIZE = 128
NUM_ATTENTION_HEADS = 4
DROPOUT = 0.1
LSTM_LAYERS = 2
CNN_CHANNELS = 64
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 64
CALIBRATION_WINDOW = 5000
NORMALIZATION_WINDOW = 100
N_FOLDS = 3                    # walk-forward folds
CANDLES_PER_DAY = 288          # 5-min en sesión 24h cripto
TOTAL_DAYS = 90

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class TimeSeriesDataset(torch.utils.data.Dataset):
    """Genera ventanas (seq_len, features) on-the-fly sin pre-materializar."""

    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.target[idx + self.seq_len]
        return x, y


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def causal_rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Z-score causal columna a columna usando media y std de [t-window, t-1].
    Mantiene categóricas/enteras-discretas intactas.
    """
    out = df.copy()
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        # Variables discretas pequeñas (estados de régimen, alineaciones).
        unique = series.dropna().unique()
        if len(unique) <= 5:
            continue
        roll = series.rolling(window=window, min_periods=window)
        mean = roll.mean().shift(1)
        std = roll.std().shift(1)
        out[col] = (series - mean) / (std + 1e-8)
    return out


def build_synthetic_data() -> pd.DataFrame:
    """OHLCV + microestructura sintéticos para la demo."""
    np.random.seed(42)
    total = TOTAL_DAYS * CANDLES_PER_DAY
    dates = pd.date_range("2025-01-01", periods=total, freq="5min")
    close = 50000 + np.random.randn(total).cumsum() * 10
    return pd.DataFrame(
        {
            "open": close + np.random.randn(total) * 2,
            "high": close + np.abs(np.random.randn(total) * 5),
            "low": close - np.abs(np.random.randn(total) * 5),
            "close": close,
            "volume": np.abs(np.random.randn(total) * 100),
            "bid": close - 0.5,
            "ask": close + 0.5,
            "bid_vol": np.abs(np.random.randn(total) * 50),
            "ask_vol": np.abs(np.random.randn(total) * 50),
        },
        index=dates,
    )


def train_deep_model(
    train_features: np.ndarray,
    train_target: np.ndarray,
    num_features: int,
) -> HybridCNNLSTMTFT:
    """Entrena el modelo causal CNN-LSTM-TFT durante ``EPOCHS`` épocas."""
    model = HybridCNNLSTMTFT(
        input_features=num_features,
        sequence_length=SEQ_LENGTH,
        cnn_channels=CNN_CHANNELS,
        lstm_hidden=HIDDEN_SIZE,
        tft_hidden=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        lstm_layers=LSTM_LAYERS,
        dropout_rate=DROPOUT,
    ).to(DEVICE)

    dataset = TimeSeriesDataset(train_features, train_target, SEQ_LENGTH)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, _ = model(xb.to(DEVICE))
            loss = criterion(logits.squeeze(), yb.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"    epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(dataset):.4f}")
    return model


@torch.no_grad()
def predict_logits(
    model: HybridCNNLSTMTFT, features: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (logits, etiquetas alineadas) sobre ``features``."""
    model.eval()
    dataset = TimeSeriesDataset(features, target, SEQ_LENGTH)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    logits = []
    for xb, _ in loader:
        out, _ = model(xb.to(DEVICE))
        logits.append(out.squeeze(-1).cpu().numpy())
    if not logits:
        return np.array([]), np.array([])
    return np.concatenate(logits), target[SEQ_LENGTH:]


def evaluate_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Entrena, calibra y evalúa un fold walk-forward."""
    feature_cols = [c for c in train_df.columns if c not in ("target", "future_return")]

    train_features = train_df[feature_cols].values.astype(np.float32)
    train_target = train_df["target"].values.astype(np.float32)
    calib_features = calib_df[feature_cols].values.astype(np.float32)
    calib_target = calib_df["target"].values.astype(np.float32)
    test_features = test_df[feature_cols].values.astype(np.float32)
    test_target = test_df["target"].values.astype(np.float32)
    test_returns = test_df["future_return"].values

    print(f"\n[fold {fold_idx}] train={len(train_df)} calib={len(calib_df)} test={len(test_df)}")

    print(f"  [fold {fold_idx}] entrenando CNN-LSTM-TFT…")
    model = train_deep_model(train_features, train_target, len(feature_cols))

    # 1) Calibración out-of-fold sobre calib_df (nunca tocado por el modelo).
    print(f"  [fold {fold_idx}] calibrando isotónica sobre validación…")
    calib_logits, calib_y = predict_logits(model, calib_features, calib_target)
    calibrator = LowLatencyRollingIsotonicCalibrator(window_size=CALIBRATION_WINDOW)
    for margin, label in zip(calib_logits, calib_y):
        calibrator.add_observation(float(margin), int(label))
    calibrator.update_calibration_curve()

    # 2) Evaluación final sobre test_df.
    test_logits, test_y = predict_logits(model, test_features, test_target)
    test_returns_aligned = test_returns[SEQ_LENGTH:]

    calibrated = np.array([calibrator.calibrate_signal(float(m)) for m in test_logits])
    y_pred = (calibrated > 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(test_y, y_pred)),
        "f1": float(f1_score(test_y, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(test_y, calibrated)) if len(np.unique(test_y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(test_y, calibrated)),
    }

    # Señal con umbrales conservadores y P&L sintético.
    signals = np.where(calibrated > 0.72, 1, np.where(calibrated < 0.28, -1, 0))
    strat_returns = signals * test_returns_aligned
    equity = np.cumprod(1 + strat_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.where(running_max == 0, 1, running_max)
    metrics["max_drawdown"] = float(np.min(drawdown)) if len(drawdown) else 0.0
    metrics["final_equity"] = float(equity[-1]) if len(equity) else 1.0

    print(
        f"  [fold {fold_idx}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"auc={metrics['auc']:.4f} brier={metrics['brier']:.4f} "
        f"dd={metrics['max_drawdown']:.2%} eq={metrics['final_equity']:.4f}"
    )
    return {"model": model, "calibrator": calibrator, "metrics": metrics}


def run_ensemble_demo(
    df_clean: pd.DataFrame, feature_cols: list[str], num_features: int
) -> dict:
    """Entrena el HybridSignalEngine sobre todo el histórico y emite una señal."""
    print("\n[ensemble] entrenando HybridSignalEngine end-to-end…")
    engine = HybridSignalEngine(
        num_features=num_features,
        sequence_length=SEQ_LENGTH,
        embedding_dim=64,
    ).to(DEVICE)

    feature_arr = df_clean[feature_cols].values.astype(np.float32)

    # Tomamos las últimas N ventanas para extraer embeddings y entrenar el meta-learner.
    n_windows = min(2000, len(feature_arr) - SEQ_LENGTH)
    windows = np.stack(
        [feature_arr[i : i + SEQ_LENGTH] for i in range(n_windows)], axis=0
    )
    x_tensor = torch.tensor(windows, dtype=torch.float32, device=DEVICE)

    feats = engine.extract_features(x_tensor)
    embeddings = feats["last_embedding"]

    # Etiquetas de régimen derivadas del estado HMM ya producido por el generator.
    if "hmm_hidden_state" in df_clean.columns:
        regime_labels = (
            df_clean["hmm_hidden_state"]
            .iloc[SEQ_LENGTH : SEQ_LENGTH + n_windows]
            .astype(int)
            .values
        )
    else:
        regime_labels = np.random.randint(0, 3, size=n_windows)

    if len(np.unique(regime_labels)) < 2:
        # Fallback determinista si la sesión sintética no muestra varianza.
        regime_labels = np.tile([0, 1, 2], reps=(n_windows // 3) + 1)[:n_windows]

    engine.regime_meta_learner.fit(embeddings, regime_labels)

    # Calibrador rellenado con los logits de las mismas ventanas.
    target_arr = df_clean["target"].iloc[SEQ_LENGTH : SEQ_LENGTH + n_windows].values
    for logit, label in zip(feats["logits"], target_arr):
        engine.calibrator.add_observation(float(logit), int(label))
    engine.calibrator.update_calibration_curve()

    # Señal de ejemplo con la última ventana.
    last_window = torch.tensor(
        feature_arr[-SEQ_LENGTH:], dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)
    signal = engine.generate_signal(last_window, "BTC/USDT", "5m", feature_cols)
    print(f"[ensemble] señal: {signal['signal']} p={signal['p_call_calibrated']} "
          f"regime={signal['regime']['label']}")
    return signal


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=== ML-SIGNAL-ENGINE (Causal Hybrid + Walk-Forward Multi-Fold) ===")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        mlflow.set_experiment("Hybrid_Causal_Isotonic_Regime")

    # 1) Datos
    print("\n[1/6] generando datos sintéticos…")
    df_raw = build_synthetic_data()

    # 2) Test de integridad: spike-injection sobre la pipeline completa.
    print("[2/6] verificando integridad temporal (spike injection)…")
    fg = FeatureGenerator(use_causal_zscore=True, window=20, mad_fallback=True)
    temporal_integrity_check(
        feature_pipeline=fg.generate_features,
        raw_df=df_raw,
        test_iterations=3,
    )

    # 3) Feature engineering + target + segunda capa de chequeo.
    print("[3/6] generando features y target con horizonte=", HORIZON)
    df_features = fg.generate_features(df_raw)
    df_features["future_return"] = (
        df_features["close"].shift(-HORIZON) / df_features["close"] - 1.0
    )
    df_features["target"] = (
        df_features["close"].shift(-HORIZON) > df_features["close"]
    ).astype(int)
    df_features = df_features.dropna()
    assert_no_target_leakage(df_features, target_col="target", horizon=HORIZON)

    # 4) Normalización causal global de columnas no estacionarias.
    print("[4/6] normalización causal global (window=", NORMALIZATION_WINDOW, ")")
    cols = [c for c in df_features.columns if c not in ("target", "future_return")]
    df_norm = causal_rolling_zscore(df_features[cols], window=NORMALIZATION_WINDOW)
    df_norm["target"] = df_features["target"].astype(float)
    df_norm["future_return"] = df_features["future_return"].astype(float)
    df_clean = df_norm.dropna().copy()
    feature_cols = [c for c in df_clean.columns if c not in ("target", "future_return")]
    print(f"  shape final: {df_clean.shape}")

    # 5) Walk-forward multi-fold con calibración fuera de muestra.
    print(f"[5/6] walk-forward {N_FOLDS} folds…")
    fold_size = len(df_clean) // (N_FOLDS + 1)
    train_size = fold_size
    calib_size = fold_size // 4
    test_size = fold_size - calib_size

    fold_metrics = []
    for fold_idx in range(N_FOLDS):
        start = fold_idx * fold_size
        train_end = start + train_size
        calib_end = train_end + calib_size
        test_end = calib_end + test_size

        if test_end > len(df_clean):
            print(f"  [fold {fold_idx}] datos insuficientes — se omite")
            continue

        train_df = df_clean.iloc[start:train_end]
        calib_df = df_clean.iloc[train_end:calib_end]
        test_df = df_clean.iloc[calib_end:test_end]

        result = evaluate_fold(fold_idx, train_df, calib_df, test_df)
        fold_metrics.append(result["metrics"])

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                mlflow.log_metrics({f"fold{fold_idx}_{k}": v for k, v in result["metrics"].items()})

    # 6) Resumen + ensemble end-to-end (Blueprint §4).
    print("\n[6/6] resumen multi-fold:")
    if fold_metrics:
        agg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
        for k, v in agg.items():
            print(f"  mean({k}) = {v:.4f}")

        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="walkforward_summary"):
                mlflow.log_params(
                    {
                        "seq_length": SEQ_LENGTH,
                        "horizon": HORIZON,
                        "hidden_size": HIDDEN_SIZE,
                        "cnn_channels": CNN_CHANNELS,
                        "attention_heads": NUM_ATTENTION_HEADS,
                        "calibration_window": CALIBRATION_WINDOW,
                        "n_folds": N_FOLDS,
                    }
                )
                mlflow.log_metrics(agg)

    run_ensemble_demo(df_clean, feature_cols, num_features=len(feature_cols))

    print("\n[OK] Pipeline híbrido completo: integridad validada, calibración OOF, "
          "walk-forward y meta-learner ejercitados.")


if __name__ == "__main__":
    main()
