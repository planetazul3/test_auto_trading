import torch
import torch.nn as nn
import numpy as np

class CNN1DExtractor(nn.Module):
    """
    Extractor de patrones locales usando CNN-1D.
    Transforma una ventana temporal de features OHLCV+ en un embedding denso.
    """
    def __init__(self, num_features: int, sequence_length: int, embedding_dim: int = 64):
        super(CNN1DExtractor, self).__init__()
        
        if num_features <= 0 or sequence_length <= 0:
            raise ValueError("num_features y sequence_length deben ser mayores a 0.")

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Bloque Convolucional 1: Detección de micro-patrones (kernel=3, ej. 3 velas)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([64, sequence_length])
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Bloque Convolucional 2: Agrupación de patrones (kernel=3)
        seq_len_after_pool1 = sequence_length // 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([128, seq_len_after_pool1])
        self.act2 = nn.GELU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Aplanado y proyección al espacio de embedding deseado
        seq_len_after_pool2 = seq_len_after_pool1 // 2
        flattened_size = 128 * seq_len_after_pool2
        
        if flattened_size <= 0:
            raise ValueError("La longitud de la secuencia es demasiado corta para las capas de pooling.")

        self.fc_projection = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor de forma (batch_size, sequence_length, num_features)
        Returns:
            Tensor de forma (batch_size, embedding_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Se esperaba un tensor 3D (batch, seq, features), se recibió {x.dim()}D")
            
        # PyTorch Conv1d espera (batch_size, channels, sequence_length)
        # Por lo tanto, transponemos las dimensiones 1 y 2
        x = x.transpose(1, 2)
        
        # Bloque 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # Bloque 2
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        # Aplanar (Flatten)
        x = x.flatten(start_dim=1)
        
        # Proyección final
        embedding = self.fc_projection(x)
        
        return embedding

if __name__ == "__main__":
    print("Inicializando CNN-1D Extractor...")
    
    # Parámetros simulados basados en nuestro Feature Generator
    BATCH_SIZE = 32
    SEQ_LENGTH = 60      # Ventana de 60 velas (ej. 5 horas en TF de 5m)
    NUM_FEATURES = 125   # ~120+ features generadas previamente
    EMBEDDING_DIM = 64   # Dimensión de salida para el TFT
    
    # Generar datos sintéticos (Batch, Sequence, Features)
    # Simulamos un batch de datos normalizados (media 0, std 1)
    torch.manual_seed(42)
    synthetic_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES)
    
    try:
        # Instanciar modelo
        model = CNN1DExtractor(
            num_features=NUM_FEATURES, 
            sequence_length=SEQ_LENGTH, 
            embedding_dim=EMBEDDING_DIM
        )
        
        # Modo evaluación para la prueba
        model.eval()
        
        with torch.no_grad():
            print(f"Tensor de entrada shape: {synthetic_input.shape} -> (Batch, Seq_Len, Features)")
            
            # Ejecutar forward pass
            output_embedding = model(synthetic_input)
            
            print(f"Tensor de salida shape:  {output_embedding.shape} -> (Batch, Embedding_Dim)")
            print("\nMuestra del embedding generado (primer elemento del batch, primeros 5 valores):")
            print(output_embedding[0, :5].numpy())
            
            # Verificación de integridad
            assert output_embedding.shape == (BATCH_SIZE, EMBEDDING_DIM), "Error en las dimensiones de salida"
            assert not torch.isnan(output_embedding).any(), "El embedding contiene NaNs"
            
            print("\n[OK] Módulo CNN-1D ejecutado exitosamente sin errores ni NaNs.")
            
    except Exception as e:
        print(f"\n[ERROR] Fallo en la ejecución de la CNN: {str(e)}")