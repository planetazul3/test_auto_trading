"""Modelo compuesto: backbone + contexto asset/timeframe + cabezales.

Wirea las tres piezas core en una sola ``nn.Module`` cuyo ``forward``
acepta el batch tal como lo produce ``WindowDataset``:

  forward(features, symbol_id, granularity_id) -> (B, C, H) logits.

Diseño:

* El **backbone** (`HybridCNNLSTMTFT`) consume ``features (B, S, F)`` y
  devuelve un embedding ``(B, E)`` por ventana.
* El **contexto** (`AssetTimeframeEmbedding`) mapea
  ``(symbol_id, granularity_id) → (B, C_emb)``.
* El **cabezal** (`MultiContractMultiHorizonHead`) recibe el embedding
  del backbone + el contexto y produce ``(B, C, H)`` logits.

Soporta dos modos de pase de contexto al head:

* ``head.config.use_context=True`` (default): el head concatena
  ``[embedding | context]`` antes del trunk.
* ``False``: el contexto se ignora — útil para ablations o cuando
  se entrena un modelo por símbolo.

El modelo es **completamente paramétrico** desde una `ModelConfig`; un
constructor de conveniencia `from_config(cfg, num_features)` instancia
todo a partir de la dataclass.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from src.models.conditioning import AssetTimeframeEmbedding
from src.models.heads import HeadConfig, MultiContractMultiHorizonHead
from src.models.hybrid_tft import HybridCNNLSTMTFT


class BackboneWithHeads(nn.Module):
    """Composición ``backbone + context_embedding + multi-contract head``."""

    def __init__(
        self,
        *,
        num_features: int,
        sequence_length: int,
        embedding: AssetTimeframeEmbedding,
        head_config: HeadConfig,
        embedding_dim: int = 64,
        lstm_hidden: Optional[int] = None,
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        cnn_channels: tuple[int, ...] = (64, 128),
        cnn_kernel_sizes: tuple[int, ...] = (3, 3),
        cnn_dilations: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        if sequence_length <= 1:
            raise ValueError("sequence_length must be > 1")
        lstm_hidden = lstm_hidden or embedding_dim

        self.backbone = HybridCNNLSTMTFT(
            input_features=num_features,
            sequence_length=sequence_length,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            tft_hidden=embedding_dim,
            num_attention_heads=num_attention_heads,
            lstm_layers=lstm_layers,
            dropout_rate=dropout,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_dilations=cnn_dilations,
        )
        self.context = embedding
        ctx_dim = embedding.embedding_dim if head_config.use_context else 0
        self.head = MultiContractMultiHorizonHead(
            input_dim=embedding_dim,
            config=head_config,
            context_dim=ctx_dim,
        )
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.sequence_length = sequence_length

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self,
        features: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> Any:
        return self.backbone.extract_embedding(features, return_attn=return_attn)

    def forward(
        self,
        features: torch.Tensor,
        symbol_id: Optional[torch.Tensor] = None,
        granularity_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.backbone.extract_embedding(features, return_attn=False)
        if self.head.config.use_context:
            if symbol_id is None or granularity_id is None:
                raise ValueError(
                    "symbol_id and granularity_id required when "
                    "head.config.use_context=True"
                )
            ctx = self.context(symbol_id, granularity_id)
            head_ctx: torch.Tensor = self.head(emb, ctx)
            return head_ctx
        head_noctx: torch.Tensor = self.head(emb)
        return head_noctx

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count_parameters(self, *, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not only_trainable) or p.requires_grad
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"sequence_length={self.sequence_length}, "
            f"embedding_dim={self.embedding_dim}, "
            f"contracts={self.head.contracts}, "
            f"horizons={self.head.horizons}"
        )


def build_model_from_config(
    cfg: Any,                           # training.config.ModelConfig (avoid circular import)
    *,
    num_features: int,
    sequence_length: int,
    embedding: AssetTimeframeEmbedding,
) -> BackboneWithHeads:
    """Instancia un ``BackboneWithHeads`` desde una ``ModelConfig``."""
    return BackboneWithHeads(
        num_features=num_features,
        sequence_length=sequence_length,
        embedding=embedding,
        head_config=cfg.head,
        embedding_dim=cfg.embedding_dim,
        lstm_hidden=cfg.lstm_hidden,
        num_attention_heads=cfg.num_attention_heads,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        cnn_channels=tuple(cfg.cnn_channels),
        cnn_kernel_sizes=tuple(cfg.cnn_kernel_sizes),
        cnn_dilations=tuple(cfg.cnn_dilations),
    )


__all__ = ["BackboneWithHeads", "build_model_from_config"]
