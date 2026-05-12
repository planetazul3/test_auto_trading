"""Monitor de drift de calibración online (B2 del audit tracking).

Idea: el ``PerContractCalibratorBundle`` ya expone ``quality_report()``
con Brier score y ECE por celda ``(contract, horizon)``. ``OnlineCalibrationMonitor``
poll-ea ese reporte y, cuando una celda excede umbrales configurables,
dispara un refit (sincrónico o en background) sobre esa celda.

Diseño:

* **Sin estado global**: cada monitor es independiente, reseteable.
* **Umbrales por métrica**: `max_brier`, `max_ece`, `min_observations`
  (no triggera sin data suficiente).
* **Cooldown opcional**: para evitar refit en cadena, una celda puede
  marcarse como "en cooldown" durante N segundos tras un refit. El reloj
  lo provee el caller (epoch) para mantener determinismo.
* **Hysteresis**: una celda en alerta vuelve a "ok" sólo si la métrica
  baja `recovery_margin` por debajo del umbral, evitando flapping.

API mínima:

    monitor = OnlineCalibrationMonitor(max_brier=0.30, max_ece=0.10)
    decisions = monitor.check(bundle, now_epoch=epoch)
    # decisions: dict[cell, DriftDecision]
    n_refit = monitor.maybe_refit(bundle, decisions, background=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

from src.models.calibration_bundle import PerContractCalibratorBundle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decisión por celda
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftDecision:
    """Decisión por celda ``(contract, horizon)``."""

    cell: str
    needs_refit: bool
    in_cooldown: bool
    reason: Optional[str]  # e.g. "brier>0.30" | "ece>0.10" | "ok" | "cooldown"
    brier_score: Optional[float]
    ece: Optional[float]
    n_observations: int


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


@dataclass
class _CellState:
    """Estado interno por celda."""

    in_alert: bool = False
    last_refit_epoch: Optional[int] = None


class OnlineCalibrationMonitor:
    """Detector + trigger de refit por drift de Brier/ECE.

    Parameters
    ----------
    max_brier:
        Brier score máximo aceptable. Si la celda lo excede → alerta.
    max_ece:
        ECE máximo aceptable.
    min_observations:
        Mínimo de observaciones para considerar el reporte fiable.
    recovery_margin:
        Hysteresis: una celda en alerta sale sólo si su métrica baja
        por debajo de ``threshold - recovery_margin``.
    cooldown_seconds:
        Segundos mínimos entre refits sucesivos sobre la misma celda
        (medidos con el ``epoch`` que pasa el caller).
    """

    def __init__(
        self,
        *,
        max_brier: float = 0.30,
        max_ece: float = 0.10,
        min_observations: int = 100,
        recovery_margin: float = 0.02,
        cooldown_seconds: int = 300,
    ) -> None:
        if not 0.0 < max_brier <= 1.0:
            raise ValueError("max_brier must be in (0, 1]")
        if not 0.0 < max_ece <= 1.0:
            raise ValueError("max_ece must be in (0, 1]")
        if min_observations <= 1:
            raise ValueError("min_observations must be > 1")
        if recovery_margin < 0:
            raise ValueError("recovery_margin must be >= 0")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")

        self.max_brier = float(max_brier)
        self.max_ece = float(max_ece)
        self.min_observations = int(min_observations)
        self.recovery_margin = float(recovery_margin)
        self.cooldown_seconds = int(cooldown_seconds)
        self._states: dict[str, _CellState] = {}

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    def check(
        self,
        bundle: PerContractCalibratorBundle,
        *,
        now_epoch: int,
    ) -> dict[str, DriftDecision]:
        """Evalúa cada celda del bundle y devuelve la decisión por celda."""
        report = bundle.quality_report()
        out: dict[str, DriftDecision] = {}
        for cell_key in self._iter_cells(bundle):
            state = self._states.setdefault(cell_key, _CellState())
            metrics = report.get(cell_key)
            if metrics is None:
                # Celda sin data suficiente todavía → no triggear.
                out[cell_key] = DriftDecision(
                    cell=cell_key, needs_refit=False, in_cooldown=False,
                    reason="insufficient_observations",
                    brier_score=None, ece=None, n_observations=0,
                )
                continue

            brier = float(metrics["brier_score"])
            ece = float(metrics["ece"])
            n_obs = int(metrics["n_observations"])
            in_cooldown = self._in_cooldown(state, now_epoch)

            # ¿Excedemos umbral? Si in_alert=True, exigimos hysteresis
            # estricta para considerarla recuperada.
            if state.in_alert:
                # Volver a OK si BOTH métricas bajan al o por debajo del umbral - margin.
                # ``<=`` (no ``<``) para permitir recuperación cuando ambas
                # métricas valen exactamente cero (caso perfecto).
                recovered = (
                    brier <= (self.max_brier - self.recovery_margin)
                    and ece <= (self.max_ece - self.recovery_margin)
                )
                if recovered:
                    state.in_alert = False
                    out[cell_key] = DriftDecision(
                        cell=cell_key, needs_refit=False, in_cooldown=in_cooldown,
                        reason="recovered",
                        brier_score=brier, ece=ece, n_observations=n_obs,
                    )
                else:
                    # Sigue en alerta; decidir refit si pasó el cooldown.
                    out[cell_key] = DriftDecision(
                        cell=cell_key, needs_refit=not in_cooldown,
                        in_cooldown=in_cooldown,
                        reason="cooldown" if in_cooldown else self._reason_for(brier, ece),
                        brier_score=brier, ece=ece, n_observations=n_obs,
                    )
                continue

            # No estaba en alerta — chequear umbrales.
            if brier > self.max_brier or ece > self.max_ece:
                state.in_alert = True
                out[cell_key] = DriftDecision(
                    cell=cell_key, needs_refit=not in_cooldown,
                    in_cooldown=in_cooldown,
                    reason="cooldown" if in_cooldown else self._reason_for(brier, ece),
                    brier_score=brier, ece=ece, n_observations=n_obs,
                )
            else:
                out[cell_key] = DriftDecision(
                    cell=cell_key, needs_refit=False, in_cooldown=in_cooldown,
                    reason="ok",
                    brier_score=brier, ece=ece, n_observations=n_obs,
                )
        return out

    # ------------------------------------------------------------------
    # Refit trigger
    # ------------------------------------------------------------------

    def maybe_refit(
        self,
        bundle: PerContractCalibratorBundle,
        decisions: Mapping[str, DriftDecision],
        *,
        now_epoch: int,
        background: bool = True,
    ) -> int:
        """Ejecuta refits sobre las celdas marcadas. Devuelve cuántas."""
        triggered = 0
        for cell, decision in decisions.items():
            if not decision.needs_refit:
                continue
            contract, h_str = cell.rsplit("__h", 1)
            horizon = int(h_str)
            cal = bundle.get(contract, horizon)
            if background:
                started = cal.update_in_background()
            else:
                started = cal.update_calibration_curve()
            if started:
                triggered += 1
                self._states[cell].last_refit_epoch = int(now_epoch)
                log.info(
                    "drift-refit cell=%s reason=%s brier=%.4f ece=%.4f",
                    cell, decision.reason, decision.brier_score or 0.0,
                    decision.ece or 0.0,
                )
        return triggered

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._states.clear()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _iter_cells(self, bundle: PerContractCalibratorBundle):
        for c in bundle.contracts:
            for h in bundle.horizons:
                yield f"{c}__h{int(h)}"

    def _in_cooldown(self, state: _CellState, now_epoch: int) -> bool:
        if state.last_refit_epoch is None:
            return False
        return (int(now_epoch) - state.last_refit_epoch) < self.cooldown_seconds

    def _reason_for(self, brier: float, ece: float) -> str:
        parts = []
        if brier > self.max_brier:
            parts.append(f"brier>{self.max_brier}")
        if ece > self.max_ece:
            parts.append(f"ece>{self.max_ece}")
        return ",".join(parts) if parts else "unknown"


__all__ = ["DriftDecision", "OnlineCalibrationMonitor"]
