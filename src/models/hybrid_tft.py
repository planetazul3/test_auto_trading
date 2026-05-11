"""Backbone híbrido CNN + LSTM + TFT.

Cambios respecto a la versión anterior:

* ``cnn_channels`` ahora se propaga al ``CNN1DExtractor`` (antes era
  un parámetro muerto). Acepta tanto un escalar (compat) como una
  secuencia ``(c1, c2, …)`` para múltiples bloques.
* Expone ``extract_embedding(x)`` para uso aguas abajo con cabezales
  multi-contract / multi-horizon, sin obligar a pasar por el
  ``classifier`` binario legacy.
* La firma ``forward(x) -> (logits, attn)`` se conserva para no romper
  consumidores existentes (devuelve un logit binario via el classifier
  legacy).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.cnn_extractor import CNN1DExtractor
from src.models.tft_attention import GatedResidualNetwork, TFTFusionNode


def _normalize_channels(value: Union[int, Sequence[int]]) -> tuple[int, ...]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("cnn_channels (int) must be > 0")
        return (value, value * 2)
    seq = tuple(int(v) for v in value)
    if not seq or any(v <= 0 for v in seq):
        raise ValueError("cnn_channels must contain positive ints")
    return seq


class HybridCNNLSTMTFT(nn.Module):
    """Backbone CNN + LSTM ramificadas con fusión TFT y cabezal lineal."""

    def __init__(
        self,
        input_features: int,
        sequence_length: int = 60,
        cnn_channels: Union[int, Sequence[int]] = 64,
        lstm_hidden: int = 128,
        tft_hidden: int = 128,
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout_rate: float = 0.1,
        *,
        cnn_kernel_sizes: Sequence[int] = (3, 3),
        cnn_dilations: Sequence[int] = (1, 2),
    ) -> None:
        super().__init__()

        channels = _normalize_channels(cnn_channels)
        if len(channels) != len(cnn_kernel_sizes) or len(channels) != len(cnn_dilations):
            # Reajuste defensivo: si el caller dio un escalar, completar kernels/dilations.
            cnn_kernel_sizes = tuple(cnn_kernel_sizes[:1]) * len(channels) \
                if len(channels) > len(cnn_kernel_sizes) else tuple(cnn_kernel_sizes)
            cnn_dilations = tuple(cnn_dilations[:1]) * len(channels) \
                if len(channels) > len(cnn_dilations) else tuple(cnn_dilations)
            if len(channels) != len(cnn_kernel_sizes) or len(channels) != len(cnn_dilations):
                raise ValueError(
                    "Length mismatch between cnn_channels, kernel_sizes and dilations"
                )

        self.cnn_extractor = CNN1DExtractor(
            num_features=input_features,
            sequence_length=sequence_length,
            embedding_dim=tft_hidden,
            channels=channels,
            kernel_sizes=cnn_kernel_sizes,
            dilations=cnn_dilations,
            dropout=dropout_rate,
            return_sequence=True,
        )

        self.lstm_encoder = BiLSTMEncoder(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout_rate,
            embedding_dim=tft_hidden,
            bidirectional=False,
            return_sequence=True,
        )

        self.tft_fusion = TFTFusionNode(
            embedding_dim=tft_hidden,
            num_heads=num_attention_heads,
            num_sources=2,
            dropout=dropout_rate,
            output_dim=tft_hidden,
        )

        self.output_grn = GatedResidualNetwork(
            input_dim=tft_hidden,
            hidden_dim=tft_hidden,
            output_dim=tft_hidden,
            dropout=dropout_rate,
        )
        # Cabezal binario legacy (logit único). Mantenido para compat;
        # downstream puede ignorarlo usando ``extract_embedding``.
        self.classifier = nn.Linear(tft_hidden, 1)

        self.tft_hidden = tft_hidden

    # ------------------------------------------------------------------

    def _encode_sequence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cnn_seq = self.cnn_extractor(x)
        lstm_seq = self.lstm_encoder(x)
        return self.tft_fusion([cnn_seq, lstm_seq])

    def extract_embedding(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Devuelve el embedding ``(B, tft_hidden)`` del último paso.

        Si ``return_attn=True`` retorna también la matriz de atención
        del nodo TFT.
        """
        fused_seq, attn = self._encode_sequence(x)
        emb = self.output_grn(fused_seq[:, -1, :])
        if return_attn:
            return emb, attn
        return emb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compat: devuelve ``(logit_binario (B,1), attn)``."""
        emb, attn = self.extract_embedding(x, return_attn=True)
        return self.classifier(emb), attn


__all__ = ["HybridCNNLSTMTFT"]
