"""Inductive Conformal Prediction (ICP) para clasificación binaria.

Idea (Vovk et al., 2005; Lei et al., 2018):

Dado un dataset de **calibración** ``{(x_i, y_i)}`` independiente del
train set, definimos un score de no-conformidad sobre las
probabilidades del modelo. Para clasificación binaria con score
clásico ``s_i = 1 - p̂(y_i | x_i)`` (la "no confianza" en la etiqueta
correcta) y para significancia ``α``, el quantile crítico es

    q_α = s_⌈(n+1)(1-α)⌉      (cuantil empírico de los scores)

Para una nueva muestra ``x_test`` y cada clase candidata ``y``, se
incluye ``y`` en el **prediction set** si

    s(x_test, y) ≤ q_α

Garantía marginal de coverage: ``P(y_true ∈ S(x_test)) ≥ 1 - α``
asumiendo intercambiabilidad (no estacionariedad mitigada por
re-calibrar online).

Uso práctico en trading binario:

* ``S = {1}``: el predictor afirma CALL con cobertura ≥ 1-α.
* ``S = {0}``: el predictor afirma PUT con cobertura ≥ 1-α.
* ``S = {0, 1}``: ambivalente → ``NO_TRADE`` (no se cumple la cobertura
  con confianza suficiente para ningún lado solo).
* ``S = ∅``: anomalía / distribution shift → ``NO_TRADE`` + alerta.

``ConformalBundle`` mantiene una instancia por ``(contract, horizon)``
para que cada cabezal tenga su propia ventana de calibración (los
scores no son comparables entre contratos).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ConformalPrediction:
    """Resultado de una predicción conformal binaria."""

    include_zero: bool
    include_one: bool

    @property
    def is_confident(self) -> bool:
        """True si el set predice una sola clase (CALL o PUT, no ambivalente)."""
        return self.include_zero ^ self.include_one  # XOR exclusivo

    @property
    def predicted_class(self) -> int:
        """``1`` si el set = ``{1}``; ``0`` si = ``{0}``; ``-1`` si ambivalente o vacío."""
        if self.include_one and not self.include_zero:
            return 1
        if self.include_zero and not self.include_one:
            return 0
        return -1

    @property
    def is_empty(self) -> bool:
        return not (self.include_zero or self.include_one)


class InductiveConformalPredictor:
    """ICP binario con score ``1 - p̂(y|x)``.

    Parameters
    ----------
    alpha:
        Nivel de significancia (default 0.1 → coverage ≥ 90%).
    window_size:
        Ventana FIFO de scores para que el predictor sea **adaptativo**
        (los datos viejos van saliendo). Default 5000.
    min_observations:
        Mínimo de scores antes de devolver un set distinto al
        "default ambivalente" ``{0, 1}``.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        window_size: int = 5000,
        min_observations: int = 50,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if min_observations <= 1:
            raise ValueError("min_observations must be > 1")
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.min_observations = int(min_observations)

        # Ring buffer NumPy (mismo patrón que el calibrador).
        self._scores_buf = np.zeros(self.window_size, dtype=np.float64)
        self._head: int = 0
        self._count: int = 0
        self._lock = threading.Lock()
        # Cache del quantile actual (invalidado en cada add_observation).
        self._sorted_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def add_observation(self, p_of_class_one: float, label: int) -> None:
        """Agrega un score de calibración para ``(p̂(1|x), y)``."""
        p = float(p_of_class_one)
        lbl = int(label)
        if lbl not in (0, 1):
            raise ValueError("label must be 0 or 1")
        # score = 1 - p̂(y_true|x).
        score = (1.0 - p) if lbl == 1 else p
        with self._lock:
            self._scores_buf[self._head] = score
            self._head = (self._head + 1) % self.window_size
            if self._count < self.window_size:
                self._count += 1
            self._sorted_cache = None  # invalidar cache

    def reset(self) -> None:
        with self._lock:
            self._head = 0
            self._count = 0
            self._sorted_cache = None

    @property
    def n_observations(self) -> int:
        with self._lock:
            return int(self._count)

    # ------------------------------------------------------------------
    # Quantile
    # ------------------------------------------------------------------

    def _ensure_sorted_cache(self) -> np.ndarray | None:
        # Llamar bajo lock.
        if self._sorted_cache is not None:
            return self._sorted_cache
        if self._count == 0:
            return None
        if self._count < self.window_size:
            view = self._scores_buf[: self._count]
        else:
            h = self._head
            view = np.concatenate(
                (self._scores_buf[h:], self._scores_buf[:h])
            )
        self._sorted_cache = np.sort(view)
        return self._sorted_cache

    def quantile(self) -> float:
        """Quantile crítico ``q_α`` actual. ``+inf`` si no hay suficiente data."""
        with self._lock:
            if self._count < self.min_observations:
                return float("inf")
            sorted_scores = self._ensure_sorted_cache()
            assert sorted_scores is not None
            n = sorted_scores.shape[0]
            # Rank = ceil((n+1)(1-α)) en 1-based; clamp a [1, n].
            rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
            rank = min(max(rank, 1), n)
            return float(sorted_scores[rank - 1])

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, p_of_class_one: float) -> ConformalPrediction:
        """Devuelve el conformal prediction set para una nueva probabilidad."""
        p = float(p_of_class_one)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        q = self.quantile()
        if not np.isfinite(q):
            # Sin suficiente calibración → conservador: ambivalente.
            return ConformalPrediction(include_zero=True, include_one=True)
        score_one = 1.0 - p   # asumir clase=1
        score_zero = p        # asumir clase=0
        return ConformalPrediction(
            include_zero=bool(score_zero <= q),
            include_one=bool(score_one <= q),
        )


# ---------------------------------------------------------------------------
# Bundle paralelo al PerContractCalibratorBundle
# ---------------------------------------------------------------------------


class ConformalBundle:
    """Conjunto de ``InductiveConformalPredictor`` por ``(contract, horizon)``.

    API vectorizada para inferencia: dado un tensor de probabilidades
    calibradas ``(B, C, H)``, devuelve:

    * ``predict_sets(probs) → (B, C, H, 2) bool[include_zero, include_one]``
    * ``is_confident(probs) → (B, C, H) bool`` (single-element set)

    Para entrenar/calibrar:

    * ``add_observations(probs, labels, mask)`` con shapes ``(B, C, H)``.
    """

    def __init__(
        self,
        contracts: Sequence[str],
        horizons: Sequence[int],
        *,
        alpha: float = 0.1,
        window_size: int = 5000,
        min_observations: int = 50,
    ) -> None:
        if not contracts:
            raise ValueError("contracts must be non-empty")
        if not horizons:
            raise ValueError("horizons must be non-empty")
        self.contracts: tuple[str, ...] = tuple(contracts)
        self.horizons: tuple[int, ...] = tuple(int(h) for h in horizons)
        self.alpha = float(alpha)
        self._cps: dict[tuple[str, int], InductiveConformalPredictor] = {
            (c, h): InductiveConformalPredictor(
                alpha=alpha,
                window_size=window_size,
                min_observations=min_observations,
            )
            for c in self.contracts
            for h in self.horizons
        }

    def get(self, contract: str, horizon: int) -> InductiveConformalPredictor:
        key = (contract, int(horizon))
        if key not in self._cps:
            raise KeyError(f"no conformal predictor for {key!r}")
        return self._cps[key]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def add_observations(
        self,
        probs: Any,
        labels: Any,
        mask: Any = None,
    ) -> None:
        """``probs`` y ``labels`` con shape ``(B, C, H)``; ``mask`` opcional."""
        probs_np = _to_numpy(probs)
        labels_np = _to_numpy(labels)
        mask_np = _to_numpy(mask) if mask is not None else None
        if probs_np.shape != labels_np.shape:
            raise ValueError("probs and labels must share shape")
        if mask_np is not None and mask_np.shape != probs_np.shape:
            raise ValueError("mask must share shape with probs")
        if probs_np.ndim != 3:
            raise ValueError(f"expected 3-D (B,C,H), got {probs_np.shape}")
        b, c, h = probs_np.shape
        if c != len(self.contracts) or h != len(self.horizons):
            raise ValueError(
                f"shape (C,H)=({c},{h}) doesn't match bundle layout "
                f"({len(self.contracts)},{len(self.horizons)})"
            )
        for ci, contract in enumerate(self.contracts):
            for hi, horizon in enumerate(self.horizons):
                cp = self._cps[(contract, int(horizon))]
                col_probs = probs_np[:, ci, hi]
                col_labels = labels_np[:, ci, hi]
                if mask_np is not None:
                    sel = mask_np[:, ci, hi].astype(bool)
                    col_probs = col_probs[sel]
                    col_labels = col_labels[sel]
                for p_val, lbl in zip(col_probs.tolist(), col_labels.tolist()):
                    if int(lbl) in (0, 1):
                        cp.add_observation(float(p_val), int(lbl))

    # ------------------------------------------------------------------
    # Inferencia vectorizada
    # ------------------------------------------------------------------

    def predict_sets(self, probs: Any) -> np.ndarray:
        """``(B, C, H) → (B, C, H, 2)`` con [include_zero, include_one]."""
        probs_np = _to_numpy(probs)
        if probs_np.ndim != 3:
            raise ValueError(f"expected 3-D (B,C,H), got {probs_np.shape}")
        b, c, h = probs_np.shape
        out = np.zeros((b, c, h, 2), dtype=bool)
        for ci, contract in enumerate(self.contracts):
            for hi, horizon in enumerate(self.horizons):
                cp = self._cps[(contract, int(horizon))]
                q = cp.quantile()
                if not np.isfinite(q):
                    out[:, ci, hi, 0] = True
                    out[:, ci, hi, 1] = True
                    continue
                p_col = probs_np[:, ci, hi]
                out[:, ci, hi, 0] = p_col <= q          # include 0 si score_0 = p <= q
                out[:, ci, hi, 1] = (1.0 - p_col) <= q  # include 1 si score_1 = 1-p <= q
        return out

    def is_confident(self, probs: Any) -> np.ndarray:
        """``(B, C, H) → (B, C, H) bool`` (single-element set)."""
        sets = self.predict_sets(probs)
        only_zero = sets[..., 0] & ~sets[..., 1]
        only_one = sets[..., 1] & ~sets[..., 0]
        return only_zero | only_one

    def predicted_classes(self, probs: Any) -> np.ndarray:
        """``(B, C, H) → (B, C, H) int``: 1=CALL, 0=PUT, -1=ambivalente/vacío."""
        sets = self.predict_sets(probs)
        out = np.full(sets.shape[:-1], -1, dtype=np.int8)
        only_zero = sets[..., 0] & ~sets[..., 1]
        only_one = sets[..., 1] & ~sets[..., 0]
        out[only_zero] = 0
        out[only_one] = 1
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def coverage_report(self) -> dict[str, dict[str, float]]:
        """Coverage empírico estimado (1 - α) por celda, para diagnóstico."""
        out: dict[str, dict[str, float]] = {}
        for (contract, horizon), cp in self._cps.items():
            if cp.n_observations < cp.min_observations:
                continue
            out[f"{contract}__h{horizon}"] = {
                "n_observations": float(cp.n_observations),
                "target_coverage": float(1.0 - cp.alpha),
                "q_alpha": float(cp.quantile()),
            }
        return out


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            out: np.ndarray = x.detach().cpu().numpy()
            return out
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x)


__all__ = [
    "ConformalBundle",
    "ConformalPrediction",
    "InductiveConformalPredictor",
]
