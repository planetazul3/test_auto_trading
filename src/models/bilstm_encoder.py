import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    """
    Codificador secuencial usando BiLSTM.
    Captura dependencias temporales a largo plazo en la ventana de features
    y las comprime en un embedding denso.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, embedding_dim: int = 64):
        super(BiLSTMEncoder, self).__init__()
        
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size y hidden_size deben ser mayores a 0.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Capa BiLSTM
        # batch_first=True significa que la entrada debe ser (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Proyección final: hidden_size * 2 porque es bidireccional
        self.fc_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor de forma (batch_size, sequence_length, input_size)
        Returns:
            Tensor de forma (batch_size, embedding_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Se esperaba un tensor 3D (batch, seq, features), se recibió {x.dim()}D")

        # Salida del LSTM:
        # out: (batch_size, seq_length, hidden_size * 2) -> Todos los estados ocultos
        # h_n: (num_layers * 2, batch_size, hidden_size) -> Último estado oculto
        # c_n: (num_layers * 2, batch_size, hidden_size) -> Último estado de la celda
        out, (h_n, c_n) = self.lstm(x)

        # Extraemos el último estado oculto de la última capa para ambas direcciones
        # h_n[-2, :, :] es la dirección forward de la última capa
        # h_n[-1, :, :] es la dirección backward de la última capa
        hidden_forward = h_n[-2, :, :]
        hidden_backward = h_n[-1, :, :]
        
        # Concatenamos ambas direcciones: shape (batch_size, hidden_size * 2)
        last_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Proyectamos al espacio de embedding deseado
        embedding = self.fc_projection(last_hidden)

        return embedding

if __name__ == "__main__":
    print("Inicializando BiLSTM Encoder...")
    
    # Parámetros simulados (deben coincidir con la entrada de la CNN para paralelismo)
    BATCH_SIZE = 32
    SEQ_LENGTH = 60      # Ventana de 60 velas
    NUM_FEATURES = 125   # ~120+ features generadas
    HIDDEN_SIZE = 64     # Tamaño oculto interno del LSTM
    EMBEDDING_DIM = 64   # Dimensión de salida para el TFT
    
    # Generar datos sintéticos (Batch, Sequence, Features)
    torch.manual_seed(42)
    synthetic_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES)
    
    try:
        # Instanciar modelo
        model = BiLSTMEncoder(
            input_size=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            dropout=0.3,
            embedding_dim=EMBEDDING_DIM
        )
        
        # Modo evaluación
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
            
            print("\n[OK] Módulo BiLSTM ejecutado exitosamente sin errores ni NaNs.")
            
    except Exception as e:
        print(f"\n[ERROR] Fallo en la ejecución del BiLSTM: {str(e)}")