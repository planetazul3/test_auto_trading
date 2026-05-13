"""Bloques de fusión y atención estilo Temporal Fusion Transformer.

Diseño:

* ``GatedLinearUnit`` y ``GatedResidualNetwork`` son los bloques base
  del TFT (Lim et al., 2020): la GRN ya integra **Add & Norm** internamente,
  así que el caller no debe envolver su salida en otra residual+LayerNorm.
* ``TFTFusionNode`` aplica:
    1. Selección de variables (GLU sobre la concatenación de fuentes).
    2. Multi-Head Self-Attention temporal con máscara causal estricta.
    3. GRN post-atención.
    4. Proyección final.
  La máscara causal se cachea como buffer y se trunca al ``seq_len`` real
  para evitar reconstruirla en cada forward (latencia crítica).
* Soporta ``key_padding_mask`` opcional para secuencias de longitud
  variable.
* Validaciones explícitas: ``embedding_dim`` debe ser divisible por
  ``num_heads`` y todos los tensores fuente deben coincidir en
  ``(batch, seq, embedding_dim)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Máxima longitud de máscara causal pre-allocada en el buffer. Si se
# excede en runtime, se reconstruye on-the-fly (camino frío).
_DEFAULT_MAX_CACHED_SEQ_LEN = 4096


class GatedLinearUnit(nn.Module):
    """GLU del TFT: ``value · sigmoid(gate)`` con proyección 2D."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc(x).chunk(2, dim=-1)
        out: torch.Tensor = value * torch.sigmoid(gate)
        return out


class GatedResidualNetwork(nn.Module):
    """GRN con Add & Norm interno (Lim et al., 2020, eq. 4-5).

    Acepta un contexto estático opcional ``c`` que se suma a la rama
    transformada antes de la activación, útil para inyectar embeddings
    condicionantes (asset, granularidad).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be > 0")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim is not None
            else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)

        self.res_project: nn.Module = (
            nn.Identity()
            if input_dim == self.output_dim
            else nn.Linear(input_dim, self.output_dim)
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = self.res_project(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            # Broadcast del contexto sobre la dimensión temporal si existe.
            ctx = self.context_proj(context)
            if ctx.dim() < h.dim():
                ctx = ctx.unsqueeze(-2)
            h = h + ctx
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        out: torch.Tensor = self.layer_norm(h + residual)
        return out


class TFTFusionNode(nn.Module):
    """Fusión multi-fuente + atención temporal causal.

    Parameters
    ----------
    embedding_dim:
        Dimensión de los embeddings de cada fuente y de la representación
        interna. Debe ser divisible por ``num_heads``.
    num_heads:
        Cabezas de atención multi-head.
    num_sources:
        Número de embeddings que se fusionan al inicio.
    dropout:
        Dropout aplicado en la atención, GRN y selector de variables.
    output_dim:
        Dimensión de salida (por defecto = ``embedding_dim``).
    max_cached_seq_len:
        Tamaño máximo de la máscara causal pre-allocada como buffer. Si la
        secuencia excede, se reconstruye en runtime sin error.
    average_attn_weights:
        Si ``True`` (default) los pesos se promedian sobre cabezas →
        ``(batch, seq, seq)``. Si ``False`` → ``(batch, num_heads, seq, seq)``.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_sources: int = 2,
        dropout: float = 0.2,
        output_dim: Optional[int] = None,
        max_cached_seq_len: int = _DEFAULT_MAX_CACHED_SEQ_LEN,
        average_attn_weights: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if num_sources <= 0:
            raise ValueError("num_sources must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding_dim = embedding_dim
        self.num_sources = num_sources
        self.num_heads = num_heads
        self.output_dim = output_dim or embedding_dim
        self.average_attn_weights = average_attn_weights

        # 1. Variable Selection: concatena fuentes y proyecta con GLU.
        self.source_projector = nn.Sequential(
            nn.Linear(embedding_dim * num_sources, embedding_dim),
            GatedLinearUnit(embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )

        # 2. Multi-Head Attention temporal.
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(embedding_dim)

        # 3. GRN post-atención (ya hace Add&Norm; NO envolver afuera).
        self.grn = GatedResidualNetwork(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            dropout=dropout,
        )

        # 4. Proyección final.
        self.output_projection = nn.Linear(embedding_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)

        # Máscara causal cacheada como buffer. Persistirá en el state_dict
        # pero no participa del backward.
        max_len = max(1, int(max_cached_seq_len))
        mask = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask_cache", mask, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache: torch.Tensor = self._causal_mask_cache  # type: ignore[assignment]
        if seq_len <= cache.size(0):
            return cache[:seq_len, :seq_len].to(device, non_blocking=True)
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        source_embs: List[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args
        ----
        source_embs:
            Lista de ``num_sources`` tensores ``(B, S, E)``.
        key_padding_mask:
            ``(B, S)`` con ``True`` en posiciones a enmascarar.
        """
        if len(source_embs) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} sources, got {len(source_embs)}"
            )
        ref = source_embs[0]
        if ref.dim() != 3:
            raise ValueError(f"Expected 3D tensors, got {ref.dim()}D")
        b, s, _ = ref.shape
        for i, src in enumerate(source_embs):
            if src.shape != ref.shape:
                raise ValueError(
                    f"source_embs[{i}] shape {tuple(src.shape)} != "
                    f"reference {tuple(ref.shape)}"
                )

        # 1. Variable selection.
        x = torch.cat(source_embs, dim=-1)
        x = self.source_projector(x)

        # 2. Multi-head self-attention con máscara causal explícita
        #    (compatibilidad con PyTorch >=2.3 que requiere is_causal=False
        #    cuando se provee attn_mask).
        causal_mask = self._causal_mask(s, x.device)
        attn_output, attn_weights_opt = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=self.average_attn_weights,
            is_causal=False,
        )

        if attn_weights_opt is None:  # need_weights=False u optimizaciones
            if self.average_attn_weights:
                attn_weights = torch.zeros(b, s, s, device=x.device, dtype=x.dtype)
            else:
                attn_weights = torch.zeros(
                    b, self.num_heads, s, s, device=x.device, dtype=x.dtype
                )
        else:
            attn_weights = attn_weights_opt

        # Add & Norm tras la atención.
        x = self.norm_attn(x + attn_output)

        # 3. GRN: ya integra Add&Norm; NO envolver en otra residual.
        x = self.grn(x)

        # 4. Proyección final + LayerNorm.
        out = self.output_norm(self.output_projection(x))
        return out, attn_weights


__all__ = ["GatedLinearUnit", "GatedResidualNetwork", "TFTFusionNode"]
