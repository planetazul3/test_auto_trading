"""Cabezales pluggables para el motor de señales.

Soporta:

* **Multi-contract**: un cabezal por contrato Deriv (CALL/PUT,
  HIGHER/LOWER, ONETOUCH/NOTOUCH, …). Internamente cada cabezal es
  binario; la diferencia entre contratos vive en el etiquetador.
* **Multi-horizon**: por cada contrato se predicen ``H`` horizontes
  simultáneamente (e.g., ``[1, 3, 5, 10]`` pasos).
* **Contexto opcional**: el embedding de entrada puede concatenarse con
  un vector de contexto (``AssetTimeframeEmbedding``).
* **Logits crudos** (sin sigmoid) — la calibración isotónica se aplica
  aguas abajo sobre los logits.

El cabezal **no asume** un conjunto fijo de contratos: el caller pasa
la lista al instanciar y el módulo genera automáticamente los pesos
correspondientes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn


# Contratos comunes ofrecidos por Deriv (binarios). El usuario puede
# extender la lista en runtime sin tocar este archivo.
DERIV_BINARY_CONTRACTS: tuple[str, ...] = (
    "CALLPUT",      # CALL/PUT (sign del retorno a horizonte H)
    "HIGHERLOWER",  # HIGHER/LOWER (cruce de barrier)
    "TOUCHNOTOUCH", # ONETOUCH/NOTOUCH (toque a barrier)
    "ENDSINOUT",    # ENDSIN/ENDSOUT (cierre dentro/fuera de rango)
    "DIGITEVENODD", # DIGITEVEN/DIGITODD
)


@dataclass(frozen=True)
class HeadConfig:
    """Configuración paramétrica del cabezal multi-contract / multi-horizon."""

    contracts: tuple[str, ...] = DERIV_BINARY_CONTRACTS
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    hidden_dim: Optional[int] = None  # default: input_dim * 2
    dropout: float = 0.1
    use_context: bool = True

    def __post_init__(self) -> None:
        if not self.contracts:
            object.__setattr__(self, "contracts", DERIV_BINARY_CONTRACTS)
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MultiContractMultiHorizonHead(nn.Module):
    """Cabezal por (contrato × horizonte) con shared trunk opcional.

    Parameters
    ----------
    input_dim:
        Dimensión del embedding final del backbone.
    config:
        ``HeadConfig`` con la lista de contratos, horizontes y dropout.
    context_dim:
        Si > 0, se concatena un vector de contexto a la entrada
        (e.g. ``AssetTimeframeEmbedding``).
    """

    def __init__(
        self,
        input_dim: int,
        config: HeadConfig = HeadConfig(),
        *,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if context_dim < 0:
            raise ValueError("context_dim must be >= 0")
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.config = config

        trunk_in = input_dim + (context_dim if config.use_context else 0)
        hidden = config.hidden_dim if config.hidden_dim is not None else trunk_in * 2

        # Shared trunk — extracts features comunes para todos los cabezales.
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
        )

        # Un Linear independiente por (contract, horizon).
        self.heads = nn.ModuleDict()
        for c in config.contracts:
            for h in config.horizons:
                self.heads[self._key(c, h)] = nn.Linear(hidden, 1)

    @staticmethod
    def _key(contract: str, horizon: int) -> str:
        return f"{contract}__h{int(horizon)}"

    @property
    def contracts(self) -> tuple[str, ...]:
        return tuple(self.config.contracts)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(self.config.horizons)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embedding: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, input_dim)`` + ``(B, context_dim)`` → ``(B, C, H)`` logits."""
        if embedding.dim() != 2:
            raise ValueError(f"embedding must be 2-D, got {embedding.dim()}D")
        if self.config.use_context and self.context_dim > 0:
            if context is None:
                raise ValueError(
                    "context tensor required (config.use_context=True, "
                    f"context_dim={self.context_dim})"
                )
            if context.shape != (embedding.size(0), self.context_dim):
                raise ValueError(
                    f"context shape {tuple(context.shape)} != "
                    f"({embedding.size(0)}, {self.context_dim})"
                )
            x = torch.cat([embedding, context], dim=-1)
        else:
            x = embedding
        z = self.trunk(x)

        # Stack en orden (contract, horizon) para mantener ordering reproducible.
        outs = []
        for c in self.config.contracts:
            row = []
            for h in self.config.horizons:
                row.append(self.heads[self._key(c, h)](z).squeeze(-1))
            outs.append(torch.stack(row, dim=-1))  # (B, H)
        return torch.stack(outs, dim=1)  # (B, C, H)

    def as_dict(self, logits: torch.Tensor) -> dict[str, dict[int, torch.Tensor]]:
        """Convierte ``(B, C, H)`` a ``{contract: {horizon: (B,) logits}}``."""
        out: dict[str, dict[int, torch.Tensor]] = {}
        for ci, c in enumerate(self.config.contracts):
            out[c] = {}
            for hi, h in enumerate(self.config.horizons):
                out[c][int(h)] = logits[:, ci, hi]
        return out


__all__ = [
    "DERIV_BINARY_CONTRACTS",
    "HeadConfig",
    "MultiContractMultiHorizonHead",
]
