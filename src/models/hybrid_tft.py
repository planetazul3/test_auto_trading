"""
Arquitectura Híbrida Causal CNN-LSTM-TFT.

Diferencia respecto al pseudocódigo del Blueprint §1.3 (sequential CNN→LSTM):
aquí CNN y LSTM se ejecutan en paralelo sobre los features crudos y sus
embeddings se fusionan en el `TFTFusionNode`. Esta variante preserva los
patrones locales originales (la CNN no ve la salida ya colapsada del LSTM) y
hace la interpretabilidad por fuente más limpia: las matrices de atención del
TFT muestran qué fuente domina en cada paso.

Para mantener fidelidad al Blueprint en la cabeza de salida, el embedding
fusionado pasa por un Gated Residual Network antes de la proyección final.
"""
from typing import Tuple

import torch
import torch.nn as nn

from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.cnn_extractor import CNN1DExtractor
from src.models.tft_attention import GatedResidualNetwork, TFTFusionNode


class HybridCNNLSTMTFT(nn.Module):
    """CNN + LSTM ramificadas, fusión TFT con máscara causal y cabezal GRN."""

    def __init__(
        self,
        input_features: int,
        sequence_length: int = 60,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        tft_hidden: int = 128,
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.cnn_extractor = CNN1DExtractor(
            num_features=input_features,
            sequence_length=sequence_length,
            embedding_dim=tft_hidden,
            return_sequence=True,
        )

        self.lstm_encoder = BiLSTMEncoder(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout_rate,
            embedding_dim=tft_hidden,
            bidirectional=False,  # Causalidad estricta.
            return_sequence=True,
        )

        self.tft_fusion = TFTFusionNode(
            embedding_dim=tft_hidden,
            num_heads=num_attention_heads,
            num_sources=2,
            dropout=dropout_rate,
            output_dim=tft_hidden,
        )

        # Cabezal: GRN para complejidad adaptativa antes de la proyección final.
        self.output_grn = GatedResidualNetwork(
            input_dim=tft_hidden,
            hidden_dim=tft_hidden,
            output_dim=tft_hidden,
            dropout=dropout_rate,
        )
        self.classifier = nn.Linear(tft_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Devuelve ``(logits[:, 1], attn_weights)`` — logits sin sigmoid."""
        cnn_seq = self.cnn_extractor(x)
        lstm_seq = self.lstm_encoder(x)
        fused_seq, attn_weights = self.tft_fusion([cnn_seq, lstm_seq])

        last_step = fused_seq[:, -1, :]
        gated = self.output_grn(last_step)
        logits = self.classifier(gated)

        return logits, attn_weights
