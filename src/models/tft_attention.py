import torch
import torch.nn as nn

class TFTFusionNode(nn.Module):
    """
    Nodo central de fusión basado en atención (inspirado en TFT).
    Recibe los embeddings de los extractores base (CNN, BiLSTM, etc.),
    aplica Multi-Head Attention para ponderar su importancia relativa,
    y devuelve una representación enriquecida.
    """
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4, 
                 num_sources: int = 2, dropout: float = 0.2, output_dim: int = 64):
        super(TFTFusionNode, self).__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim debe ser divisible por num_heads.")
            
        self.embedding_dim = embedding_dim
        self.num_sources = num_sources
        
        # Multi-Head Attention (batch_first=True para manejar (Batch, Seq, Feature))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Capas de normalización (Add & Norm)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Proyección final tras aplanar las fuentes atendidas
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim * num_sources, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, cnn_emb: torch.Tensor, lstm_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            cnn_emb: Tensor de forma (batch_size, embedding_dim)
            lstm_emb: Tensor de forma (batch_size, embedding_dim)
        Returns:
            enriched_representation: Tensor (batch_size, output_dim)
            attn_weights: Pesos de atención para interpretabilidad (batch_size, num_sources, num_sources)
        """
        # Verificación de dimensiones
        if cnn_emb.shape != lstm_emb.shape:
            raise ValueError(f"Las dimensiones de CNN {cnn_emb.shape} y LSTM {lstm_emb.shape} deben coincidir.")
            
        # Apilamos los embeddings para formar una "secuencia" de fuentes
        # Shape resultante: (batch_size, num_sources, embedding_dim)
        # Donde num_sources = 2 (índice 0: CNN, índice 1: LSTM)
        x = torch.stack([cnn_emb, lstm_emb], dim=1)
        
        # 1. Multi-Head Attention (Self-Attention sobre las fuentes)
        # Query, Key y Value son el mismo tensor 'x'
        attn_output, attn_weights = self.multihead_attn(query=x, key=x, value=x)
        
        # 2. Add & Norm 1
        x = self.norm1(x + attn_output)
        
        # 3. Feed-Forward Network
        ffn_output = self.ffn(x)
        
        # 4. Add & Norm 2
        x = self.norm2(x + ffn_output)
        
        # 5. Aplanar la secuencia de fuentes atendidas
        # Shape: (batch_size, num_sources * embedding_dim)
        x_flat = x.flatten(start_dim=1)
        
        # 6. Proyección final
        enriched_representation = self.output_projection(x_flat)
        
        return enriched_representation, attn_weights

if __name__ == "__main__":
    print("Inicializando TFT Fusion Node...")
    
    # Parámetros simulados
    BATCH_SIZE = 32
    EMBEDDING_DIM = 64
    NUM_HEADS = 4
    OUTPUT_DIM = 64
    
    # Simulamos las salidas previas de la CNN y el BiLSTM
    torch.manual_seed(42)
    mock_cnn_emb = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    mock_lstm_emb = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    
    try:
        # Instanciar modelo
        model = TFTFusionNode(
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_sources=2,
            output_dim=OUTPUT_DIM
        )
        
        model.eval()
        
        with torch.no_grad():
            print(f"Entrada CNN shape:  {mock_cnn_emb.shape}")
            print(f"Entrada LSTM shape: {mock_lstm_emb.shape}")
            
            # Ejecutar forward pass
            enriched_out, attention_weights = model(mock_cnn_emb, mock_lstm_emb)
            
            print(f"\nSalida Enriquecida shape: {enriched_out.shape} -> (Batch, Output_Dim)")
            print(f"Pesos de Atención shape:  {attention_weights.shape} -> (Batch, Target_Seq, Source_Seq)")
            
            print("\nMuestra de pesos de atención (primer elemento del batch):")
            # La matriz 2x2 muestra cómo la CNN y el LSTM se prestan atención mutuamente
            print(attention_weights[0].numpy())
            
            # Verificación de integridad
            assert enriched_out.shape == (BATCH_SIZE, OUTPUT_DIM), "Error en dimensiones de salida"
            assert not torch.isnan(enriched_out).any(), "La salida contiene NaNs"
            
            print("\n[OK] Módulo TFT Attention ejecutado exitosamente sin errores.")
            
    except Exception as e:
        print(f"\n[ERROR] Fallo en la ejecución del TFT Fusion Node: {str(e)}")