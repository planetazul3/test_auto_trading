import torch
import torch.nn as nn
from typing import List, Tuple

from src.models.cnn_extractor import CNN1DExtractor
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.tft_attention import TFTFusionNode

class HybridCNNLSTMTFT(nn.Module):
    """
    Arquitectura Híbrida Causal: CNN (patrones locales) + LSTM (dependencias largas) 
    fusionadas mediante un nodo de atención tipo TFT.
    """
    def __init__(self, 
                 input_features: int, 
                 cnn_channels: int = 64, 
                 lstm_hidden: int = 128, 
                 tft_hidden: int = 128, 
                 num_attention_heads: int = 4, 
                 dropout_rate: float = 0.1):
        super(HybridCNNLSTMTFT, self).__init__()
        
        # 1. Extractor de patrones locales (CNN)
        # Nota: embedding_dim se ajusta para que coincida con tft_hidden
        self.cnn_extractor = CNN1DExtractor(
            num_features=input_features,
            sequence_length=60, # valor por defecto
            embedding_dim=tft_hidden
        )
        
        # 2. Codificador de dependencias temporales (LSTM)
        self.lstm_encoder = BiLSTMEncoder(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout_rate,
            embedding_dim=tft_hidden,
            bidirectional=False # Causalidad estricta
        )
        
        # 3. Nodo de Fusión Temporal (TFT)
        self.tft_fusion = TFTFusionNode(
            embedding_dim=tft_hidden,
            num_heads=num_attention_heads,
            num_sources=2,
            dropout=dropout_rate,
            output_dim=tft_hidden
        )
        
        # 4. Cabezal de salida (Sigmoide para clasificación binaria)
        self.classifier = nn.Sequential(
            nn.Linear(tft_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (batch, seq_len, features)
        Returns:
            logits: (batch, 1)
            attention_weights: Pesos de atención para interpretabilidad
        """
        # Extraer representaciones de cada rama
        cnn_rep = self.cnn_extractor(x)   # (batch, tft_hidden)
        lstm_rep = self.lstm_encoder(x) # (batch, tft_hidden)
        
        # Fusión mediante atención
        fused_rep, attn_weights = self.tft_fusion([cnn_rep, lstm_rep])
        
        # Clasificación
        logits = self.classifier(fused_rep)
        
        return logits, attn_weights
