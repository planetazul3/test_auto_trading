import torch
import torch.nn as nn
from typing import List, Tuple

from src.models.cnn_extractor import CNN1DExtractor
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.tft_attention import TFTFusionNode

class HybridCNNLSTMTFT(nn.Module):
    """
    Arquitectura Híbrida Causal: CNN (patrones locales) + LSTM (dependencias largas) 
    fusionadas mediante un nodo de atención tipo TFT con máscara causal temporal.
    """
    def __init__(self, 
                 input_features: int, 
                 sequence_length: int = 60,
                 cnn_channels: int = 64, 
                 lstm_hidden: int = 128, 
                 tft_hidden: int = 128, 
                 num_attention_heads: int = 4, 
                 dropout_rate: float = 0.1):
        super(HybridCNNLSTMTFT, self).__init__()
        
        # 1. Extractor de patrones locales (CNN) - Preserve sequence
        self.cnn_extractor = CNN1DExtractor(
            num_features=input_features,
            sequence_length=sequence_length,
            embedding_dim=tft_hidden,
            return_sequence=True
        )
        
        # 2. Codificador de dependencias temporales (LSTM) - Preserve sequence
        self.lstm_encoder = BiLSTMEncoder(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout_rate,
            embedding_dim=tft_hidden,
            bidirectional=False,
            return_sequence=True
        )
        
        # 3. Nodo de Fusión Temporal (TFT) - Sequence to Sequence
        self.tft_fusion = TFTFusionNode(
            embedding_dim=tft_hidden,
            num_heads=num_attention_heads,
            num_sources=2,
            dropout=dropout_rate,
            output_dim=tft_hidden
        )
        
        # 4. Cabezal de salida
        self.classifier = nn.Sequential(
            nn.Linear(tft_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            logits: (batch, 1) - Predicción para el último paso
            attention_weights: Pesos de atención temporal
        """
        # Extraer secuencias de representaciones
        cnn_seq = self.cnn_extractor(x)   # (batch, seq, tft_hidden)
        lstm_seq = self.lstm_encoder(x) # (batch, seq, tft_hidden)
        
        # Fusión mediante atención causal
        fused_seq, attn_weights = self.tft_fusion([cnn_seq, lstm_seq]) # (batch, seq, tft_hidden)
        
        # Usamos solo el último paso para la clasificación de la señal actual
        last_step = fused_seq[:, -1, :]
        logits = self.classifier(last_step)
        
        return logits, attn_weights

