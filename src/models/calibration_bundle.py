"""Bundle de calibradores isotónicos por (contrato, horizonte).

Cada cabezal de ``MultiContractMultiHorizonHead`` requiere su propia
curva de calibración: la distribución de logits no es comparable entre
contratos (CALL/PUT vs TOUCH/NOTOUCH) ni entre horizontes (h=1 vs h=10
tienen tasas base distintas).

``PerContractCalibratorBundle`` mantiene un
``LowLatencyRollingIsotonicCalibrator`` por celda y expone una API
vectorizada que el motor de inferencia puede llamar con todos los
logits de un step en una sola pasada.

Diseño:

* **Determinístico**: el orden ``(contracts, horizons)`` se fija al
  construir y se respeta en todas las APIs.
* **Thread-safe**: cada calibrador interno ya lo es; el bundle sólo
  delega.
* **Persistencia**: `state_dict` / `load_state_dict` serializan las
  curvas (no los buffers de margins/labels — esos se reconstruyen
  online en producción).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.models.calibration import LowLatencyRollingIsotonicCalibrator


class PerContractCalibratorBundle:
    """Conjunto de calibradores indexado por ``(contract, horizon)``."""

    def __init__(
        self,
        contracts: Sequence[str],
        horizons: Sequence[int],
        *,
        window_size: int = 5000,
        min_observations: int = 100,
    ) -> None:
        if not contracts:
            raise ValueError("contracts must be non-empty")
        if not horizons:
            raise ValueError("horizons must be non-empty")
        self.contracts: tuple[str, ...] = tuple(contracts)
        self.horizons: tuple[int, ...] = tuple(int(h) for h in horizons)
        self._cals: dict[tuple[str, int], LowLatencyRollingIsotonicCalibrator] = {
            (c, h): LowLatencyRollingIsotonicCalibrator(
                window_size=window_size, min_observations=min_observations
            )
            for c in self.contracts
            for h in self.horizons
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, contract: str, horizon: int) -> LowLatencyRollingIsotonicCalibrator:
        key = (contract, int(horizon))
        if key not in self._cals:
            raise KeyError(f"no calibrator for {key!r}")
        return self._cals[key]

    # ------------------------------------------------------------------
    # Observations / fit
    # ------------------------------------------------------------------

    def add_observations(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Recibe ``(B, C, H)`` logits y ``(B, C, H)`` labels.

        Las posiciones con ``mask=False`` se ignoran. Las labels deben
        estar en ``{0, 1}`` para las posiciones válidas.
        """
        logits_np = _to_numpy(logits)
        labels_np = _to_numpy(labels)
        mask_np = _to_numpy(mask) if mask is not None else None
        if logits_np.shape != labels_np.shape:
            raise ValueError("logits and labels must share shape")
        if mask_np is not None and mask_np.shape != logits_np.shape:
            raise ValueError("mask must share shape with logits")
        if logits_np.ndim != 3:
            raise ValueError(f"expected 3-D (B,C,H), got {logits_np.shape}")
        b, c, h = logits_np.shape
        if c != len(self.contracts) or h != len(self.horizons):
            raise ValueError(
                f"shape (C,H)=({c},{h}) doesn't match configured "
                f"({len(self.contracts)},{len(self.horizons)})"
            )
        for ci, contract in enumerate(self.contracts):
            for hi, horizon in enumerate(self.horizons):
                cal = self._cals[(contract, int(horizon))]
                col_logits = logits_np[:, ci, hi]
                col_labels = labels_np[:, ci, hi]
                if mask_np is not None:
                    sel = mask_np[:, ci, hi].astype(bool)
                    col_logits = col_logits[sel]
                    col_labels = col_labels[sel]
                for lg, lb in zip(col_logits.tolist(), col_labels.tolist()):
                    cal.add_observation(float(lg), int(lb))

    def update_all(self, *, background: bool = False) -> int:
        """Refit de todas las curvas con suficiente data. Devuelve cuántas
        se actualizaron."""
        updated = 0
        for cal in self._cals.values():
            if background:
                if cal.update_in_background():
                    updated += 1
            else:
                if cal.update_calibration_curve():
                    updated += 1
        return updated

    # ------------------------------------------------------------------
    # Inferencia vectorizada
    # ------------------------------------------------------------------

    def calibrate(
        self, logits: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """``(B, C, H)`` logits → ``(B, C, H)`` probabilidades calibradas."""
        logits_np = _to_numpy(logits)
        if logits_np.ndim != 3:
            raise ValueError(f"expected 3-D (B,C,H), got {logits_np.shape}")
        b, c, h = logits_np.shape
        if c != len(self.contracts) or h != len(self.horizons):
            raise ValueError("shape (C,H) doesn't match bundle layout")
        out = np.empty_like(logits_np, dtype=np.float64)
        for ci, contract in enumerate(self.contracts):
            for hi, horizon in enumerate(self.horizons):
                cal = self._cals[(contract, int(horizon))]
                for bi in range(b):
                    out[bi, ci, hi] = cal.calibrate_signal(float(logits_np[bi, ci, hi]))
        return out

    # ------------------------------------------------------------------
    # Métricas agregadas
    # ------------------------------------------------------------------

    def quality_report(self) -> dict[str, dict[str, float]]:
        """Brier + ECE por celda. Útil para detectar drift por contrato."""
        report: dict[str, dict[str, float]] = {}
        for (contract, horizon), cal in self._cals.items():
            if not cal.is_fitted or cal.n_observations < cal.min_observations:
                continue
            key = f"{contract}__h{horizon}"
            report[key] = {
                "n_observations": float(cal.n_observations),
                "brier_score": float(cal.brier_score()),
                "ece": float(cal.expected_calibration_error()),
            }
        return report

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, dict[str, np.ndarray]]:
        """Serializa sólo las curvas (no los buffers in-memory)."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for (contract, horizon), cal in self._cals.items():
            if not cal.is_fitted:
                continue
            x, y = cal.curve
            out[f"{contract}__h{horizon}"] = {
                "x_thresholds": np.asarray(x, dtype=np.float64),
                "y_values": np.asarray(y, dtype=np.float64),
            }
        return out

    def load_state_dict(self, state: Mapping[str, Mapping[str, np.ndarray]]) -> None:
        for key, curve in state.items():
            if "__h" not in key:
                raise ValueError(f"malformed key {key!r}")
            contract, h_str = key.rsplit("__h", 1)
            horizon = int(h_str)
            if (contract, horizon) not in self._cals:
                continue  # contrato no presente en el bundle actual
            cal = self._cals[(contract, horizon)]
            x = np.asarray(curve["x_thresholds"], dtype=np.float64).copy()
            y = np.asarray(curve["y_values"], dtype=np.float64).copy()
            cal._curve = (x, y)  # type: ignore[attr-defined]
            cal.is_fitted = True


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        out: np.ndarray = x.detach().cpu().numpy()
        return out
    return np.asarray(x)


__all__ = ["PerContractCalibratorBundle"]
