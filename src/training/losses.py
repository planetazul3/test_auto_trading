"""Pérdidas multi-contract / multi-horizon con masking robusto.

* ``MultiContractLoss`` opera sobre el output ``(B, C, H)`` del
  ``MultiContractMultiHorizonHead`` y las labels ``(B, C, H)`` int8
  con valores en ``{0, 1, IGNORE_LABEL}``. La máscara de validez
  ``label_mask`` ``(B, C, H)`` se respeta estrictamente: las posiciones
  inválidas no contribuyen al gradiente.
* Pesos por contrato y horizonte: el caller puede sobreponderar
  contratos con menor cardinal de datos (e.g. NOTOUCH) sin tocar el
  resto. Si no se pasan pesos, todos los cabezales contribuyen por
  igual.
* Pos-weight por contrato (estilo ``BCEWithLogitsLoss``) para corregir
  desbalance de clases. Opcional.
* Devuelve un escalar (loss total) y un dict con la loss desagregada
  por contrato — útil para logging y para detectar cabezales que se
  degradan en aislado.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.labels import IGNORE_LABEL


class MultiContractLoss(nn.Module):
    """BCE-with-logits enmascarada por contrato y horizonte."""

    def __init__(
        self,
        contracts: Sequence[str],
        horizons: Sequence[int],
        *,
        contract_weights: Optional[Mapping[str, float]] = None,
        horizon_weights: Optional[Mapping[int, float]] = None,
        pos_weight: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        if not contracts:
            raise ValueError("contracts must be non-empty")
        if not horizons:
            raise ValueError("horizons must be non-empty")
        self.contracts = tuple(contracts)
        self.horizons = tuple(int(h) for h in horizons)

        c_w = torch.ones(len(self.contracts), dtype=torch.float32)
        if contract_weights:
            for i, c in enumerate(self.contracts):
                if c in contract_weights:
                    c_w[i] = float(contract_weights[c])
        self.register_buffer("contract_weights", c_w)

        h_w = torch.ones(len(self.horizons), dtype=torch.float32)
        if horizon_weights:
            for i, h in enumerate(self.horizons):
                if int(h) in horizon_weights:
                    h_w[i] = float(horizon_weights[int(h)])
        self.register_buffer("horizon_weights", h_w)

        if pos_weight is not None:
            pw = torch.ones(len(self.contracts), dtype=torch.float32)
            for i, c in enumerate(self.contracts):
                if c in pos_weight:
                    pw[i] = float(pos_weight[c])
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.dim() != 3:
            raise ValueError(f"logits must be (B,C,H), got {tuple(logits.shape)}")
        if labels.shape != logits.shape:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} != logits shape "
                f"{tuple(logits.shape)}"
            )
        b, c, h = logits.shape
        if c != len(self.contracts):
            raise ValueError(
                f"logits C={c} doesn't match configured contracts ({len(self.contracts)})"
            )
        if h != len(self.horizons):
            raise ValueError(
                f"logits H={h} doesn't match configured horizons ({len(self.horizons)})"
            )

        if label_mask is None:
            mask = labels != IGNORE_LABEL
        else:
            mask = label_mask.bool()

        labels_f = labels.to(logits.dtype).clamp(min=0.0)

        # BCE elementwise sin reducción.
        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(
                logits, labels_f, reduction="none"
            )
        else:
            # pos_weight broadcastea a (1, C, 1).
            pw = self.pos_weight.view(1, -1, 1)
            bce = F.binary_cross_entropy_with_logits(
                logits, labels_f, reduction="none", pos_weight=pw
            )

        # Aplicar pesos por contrato/horizonte sin allocar tensores extra.
        cw = self.contract_weights.view(1, -1, 1)
        hw = self.horizon_weights.view(1, 1, -1)
        weighted = bce * cw * hw
        # Enmascarar antes de la reducción.
        weighted = weighted * mask.to(weighted.dtype)
        denom = mask.sum().clamp_min(1).to(weighted.dtype)
        loss = weighted.sum() / denom

        # Desglose por contrato (mask-aware) para logging.
        per_contract: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.contracts):
            m_c = mask[:, i, :].to(weighted.dtype)
            denom_c = m_c.sum().clamp_min(1)
            per_contract[name] = (weighted[:, i, :].sum() / denom_c).detach()

        return loss, per_contract


__all__ = ["MultiContractLoss"]
