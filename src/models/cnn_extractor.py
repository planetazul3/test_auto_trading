import torch
import torch.nn as nn
import numpy as np

class CausalConv1d(nn.Conv1d):
    """
    Convolución 1D causal para evitar look-ahead bias en series temporales.
    Aplica padding solo al inicio de la secuencia.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
            padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Aplicamos padding manual a la izquierda
        x = nn.functional.pad(x, (self.__padding, 0))
        return super(CausalConv1d, self).forward(x)

class CNN1DExtractor(nn.Module):
    """
    Extractor de patrones locales usando CNN-1D Causal.
    Transforma una ventana temporal de features OHLCV+ en un embedding denso.
    Diseñado para ser robusto a diferentes longitudes de secuencia y evitar fugas de información.
    """
    def __init__(self, num_features: int, sequence_length: int, embedding_dim: int = 64, return_sequence: bool = False):
        super(CNN1DExtractor, self).__init__()
        
        if num_features <= 0 or sequence_length <= 0:
            raise ValueError("num_features y sequence_length deben ser mayores a 0.")

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.return_sequence = return_sequence

        # Bloque Convolucional 1: Detección de micro-patrones
        self.conv1 = CausalConv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.act1 = nn.GELU()
        
        # Bloque Convolucional 2: Agrupación de patrones
        self.conv2 = CausalConv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=128)
        self.act2 = nn.GELU()

        if not self.return_sequence:
            # Adaptive Pooling: Colapsa a vector fijo
            self.global_pool = nn.AdaptiveMaxPool1d(output_size=4)
            flattened_size = 128 * 4
            self.fc_projection = nn.Sequential(
                nn.Linear(flattened_size, 256),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            # Proyección por cada paso de tiempo
            self.step_projection = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        
        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos Kaiming."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor (batch, seq, features)
        Returns:
            Si return_sequence=False: (batch, embedding_dim)
            Si return_sequence=True:  (batch, seq, embedding_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Se esperaba 3D, se recibió {x.dim()}D")
            
        x = x.transpose(1, 2) # (batch, features, seq)
        
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        
        if not self.return_sequence:
            x = self.global_pool(x)
            x = x.flatten(start_dim=1)
            return self.fc_projection(x)
        else:
            # x es (batch, 128, seq) -> volver a (batch, seq, 128)
            x = x.transpose(1, 2)
            return self.step_projection(x)


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