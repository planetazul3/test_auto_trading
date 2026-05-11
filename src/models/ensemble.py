"""Motor de señales híbrido: backbone CNN+LSTM+TFT + cabezales + calibrador.

Versión refactorizada:

* Reusa ``HybridCNNLSTMTFT`` como backbone único (antes duplicaba la
  arquitectura). Cualquier mejora al backbone aplica también aquí.
* **Device-aware**: ``generate_signal`` mueve la entrada al device del
  modelo automáticamente; ``extract_features`` opera con tensores en
  device y sólo convierte a NumPy si el caller lo pide.
* No toca ``self.training``: se asume que el caller gestiona ``eval()``.
  Las inferencias usan ``torch.inference_mode``.
* Política de señal (umbrales, sizing, ruta) extraída a
  ``SignalPolicy`` — cero hardcodes en la lógica del modelo.
* ``regime_labels`` y ``contracts`` son configurables (vienen del
  meta-learner y del cabezal). Útil para soportar setups donde alguna
  estrategia define sus propios regímenes/contratos.
* ``as_of_epoch`` permite registrar el timestamp **de mercado** del
  signal en vez del wall-clock del proceso (trazabilidad/backtest).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.models.calibration import LowLatencyRollingIsotonicCalibrator
from src.models.hybrid_tft import HybridCNNLSTMTFT
from src.models.meta_learner import DEFAULT_REGIME_LABELS, RegimeAwareMetaLearner


@dataclass(frozen=True)
class SignalPolicy:
    """Política de decisión sobre la probabilidad calibrada.

    Todos los umbrales son parametrizables; los defaults preservan el
    comportamiento previo del engine.
    """

    call_threshold: float = 0.70
    put_threshold: float = 0.30
    strong_call_threshold: float = 0.80
    strong_put_threshold: float = 0.20
    strong_sizing: float = 1.5
    normal_sizing: float = 1.0
    no_trade_sizing: float = 0.0
    regime_sizing: Mapping[int, float] = field(
        default_factory=lambda: {0: 0.8, 1: 1.0, 2: 0.4}
    )
    regime_routes: Mapping[int, str] = field(
        default_factory=lambda: {1: "HFT_EXT"}
    )
    default_route: str = "LP_INTERNAL"

    def __post_init__(self) -> None:
        if not 0.0 < self.put_threshold < self.call_threshold < 1.0:
            raise ValueError("require 0 < put_threshold < call_threshold < 1")
        if not 0.0 < self.strong_put_threshold <= self.put_threshold:
            raise ValueError("strong_put_threshold must be <= put_threshold and > 0")
        if not self.call_threshold <= self.strong_call_threshold < 1.0:
            raise ValueError("strong_call_threshold must be >= call_threshold and < 1")


class HybridSignalEngine(nn.Module):
    """Engine compuesto: backbone híbrido + meta-learner + calibrador."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        embedding_dim: int = 64,
        *,
        lstm_hidden: Optional[int] = None,
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout_rate: float = 0.1,
        cnn_channels: Sequence[int] | int | None = None,
        policy: Optional[SignalPolicy] = None,
        regime_labels: Sequence[str] = DEFAULT_REGIME_LABELS,
        calibrator_window: int = 5000,
        meta_learner: Optional[RegimeAwareMetaLearner] = None,
    ) -> None:
        super().__init__()
        if num_features <= 0 or sequence_length <= 0 or embedding_dim <= 0:
            raise ValueError("num_features, sequence_length and embedding_dim must be > 0")

        lstm_hidden = lstm_hidden or embedding_dim
        if cnn_channels is None:
            cnn_channels = (embedding_dim, embedding_dim * 2)

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.backbone = HybridCNNLSTMTFT(
            input_features=num_features,
            sequence_length=sequence_length,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            tft_hidden=embedding_dim,
            num_attention_heads=num_attention_heads,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate,
        )
        # Cabezal binario para CALL/PUT — se mantiene como interfaz mínima.
        # Para multi-contract usar ``MultiContractMultiHorizonHead`` aparte.
        self.signal_head = nn.Linear(embedding_dim, 1)

        self.regime_meta_learner = meta_learner or RegimeAwareMetaLearner(
            regime_labels=tuple(regime_labels)
        )
        self.calibrator = LowLatencyRollingIsotonicCalibrator(
            window_size=calibrator_window
        )
        self.policy = policy or SignalPolicy()
        self.regime_labels: tuple[str, ...] = tuple(regime_labels)
        if len(self.regime_labels) != 3:
            raise ValueError("regime_labels must contain exactly 3 entries")

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def extract_features(
        self,
        x: torch.Tensor,
        *,
        as_numpy: bool = False,
    ) -> dict[str, Any]:
        """Devuelve embedding, logit binario y matriz de atención.

        Por defecto los tensores se mantienen en device (latencia mínima).
        ``as_numpy=True`` los traslada a CPU/NumPy — útil para feedear al
        meta-learner XGBoost o al calibrador.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3-D (B,S,F), got {x.dim()}D")
        x = x.to(self._device(), non_blocking=True)
        emb, attn = self.backbone.extract_embedding(x, return_attn=True)
        logits = self.signal_head(emb).squeeze(-1)
        out: dict[str, Any] = {
            "embedding": emb,
            "logits": logits,
            "attn_weights": attn,
        }
        if as_numpy:
            out = {k: v.detach().cpu().numpy() for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    # Inferencia de señal calibrada
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_signal(
        self,
        x_window: torch.Tensor,
        asset: str,
        timeframe: str,
        *,
        as_of_epoch: Optional[int] = None,
    ) -> dict[str, Any]:
        """Produce una señal calibrada + régimen + sizing.

        Parameters
        ----------
        x_window:
            Tensor 2-D ``(S, F)`` o 3-D ``(1, S, F)``.
        asset:
            Etiqueta del símbolo (e.g. ``"R_100"``, ``"frxEURUSD"``).
        timeframe:
            Etiqueta legible del timeframe (e.g. ``"5m"``, ``"ticks"``).
        as_of_epoch:
            Epoch (segundos UTC) del último dato de mercado en la
            ventana. Si es ``None`` se usa el wall-clock actual.
        """
        if x_window.dim() == 2:
            x_window = x_window.unsqueeze(0)
        features = self.extract_features(x_window, as_numpy=False)
        logit = float(features["logits"][0].item())
        emb_np = features["embedding"].detach().cpu().numpy()

        # 1. Calibración: logit → probabilidad calibrada.
        p_calibrated = self.calibrator.calibrate_signal(logit)

        # 2. Régimen.
        if not self.regime_meta_learner.is_fitted:
            # Sin meta-learner entrenado, sólo emitimos signal+sizing sin régimen.
            regime_probs = np.array([1.0 / 3] * 3, dtype=np.float64)
        else:
            regime_probs = self.regime_meta_learner.predict_regime_probs(emb_np)[0]
        current_regime = int(np.argmax(regime_probs))

        # 3. Routing y sizing.
        signal, base_sizing = self._classify_signal(p_calibrated)
        regime_mult = float(self.policy.regime_sizing.get(current_regime, 1.0))
        final_sizing = base_sizing * regime_mult if signal != "NO_TRADE" else 0.0
        route = self.policy.regime_routes.get(current_regime, self.policy.default_route)

        ts = self._format_timestamp(as_of_epoch)
        return {
            "timestamp": ts,
            "asset": asset,
            "timeframe": timeframe,
            "signal": signal,
            "p_call_calibrated": round(p_calibrated, 6),
            "logit": round(logit, 6),
            "regime": {
                "label": self.regime_labels[current_regime],
                "probs": {
                    self.regime_labels[i]: round(float(regime_probs[i]), 6)
                    for i in range(3)
                },
            },
            "execution": {
                "sizing_multiplier": round(final_sizing, 4),
                "route": route,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_signal(self, p: float) -> tuple[str, float]:
        pol = self.policy
        if p >= pol.call_threshold:
            sizing = pol.strong_sizing if p >= pol.strong_call_threshold else pol.normal_sizing
            return "CALL", sizing
        if p <= pol.put_threshold:
            sizing = pol.strong_sizing if p <= pol.strong_put_threshold else pol.normal_sizing
            return "PUT", sizing
        return "NO_TRADE", pol.no_trade_sizing

    @staticmethod
    def _format_timestamp(as_of_epoch: Optional[int]) -> str:
        if as_of_epoch is None:
            return datetime.datetime.now(datetime.timezone.utc).isoformat()
        return datetime.datetime.fromtimestamp(
            int(as_of_epoch), tz=datetime.timezone.utc
        ).isoformat()


__all__ = ["HybridSignalEngine", "SignalPolicy"]
