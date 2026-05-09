import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AttentionPooling(nn.Module):
    """
    Mecanismo de atención para colapsar la dimensión temporal.
    Permite al modelo aprender qué pasos de tiempo son más relevantes
    para la señal final.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden)
        weights = self.attn(x)  # (batch, seq, 1)
        weights = F.softmax(weights, dim=1)
        # Suma ponderada: (batch, hidden)
        return torch.sum(x * weights, dim=1)

class BiLSTMEncoder(nn.Module):
    """
    Codificador Temporal Robusto (LSTM/GRU).
    Captura dependencias temporales en la ventana de features y las comprime
    en un embedding denso calibrado para el motor de señales.
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 64, 
                 num_layers: int = 2, 
                 dropout: float = 0.3, 
                 embedding_dim: int = 64,
                 bidirectional: bool = False,
                 rnn_type: str = "lstm",
                 return_sequence: bool = False):
        super(BiLSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence

        # Selección de celda recurrente
        rnn_class = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Dimensión de salida del RNN
        rnn_out_size = hidden_size * (2 if bidirectional else 1)
        self.ln_rnn = nn.LayerNorm(rnn_out_size)
        
        if not self.return_sequence:
            # Mecanismo de Atención Temporal para colapsar
            self.attention = AttentionPooling(rnn_out_size)
            self.fc_projection = nn.Sequential(
                nn.Linear(rnn_out_size, rnn_out_size * 2),
                nn.GELU(),
                nn.LayerNorm(rnn_out_size * 2),
                nn.Dropout(p=dropout),
                nn.Linear(rnn_out_size * 2, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            # Proyección por paso de tiempo
            self.step_projection = nn.Sequential(
                nn.Linear(rnn_out_size, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        
        self._init_weights()

    def _init_weights(self):
        """Inicialización robusta."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                if 'bias_ih' in name and isinstance(self.rnn, nn.LSTM):
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del codificador.
        Args:
            x: Tensor (batch, seq, features)
        Returns:
            embedding: (batch, dim) o (batch, seq, dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Se esperaba 3D, se recibió {x.dim()}D")

        out, _ = self.rnn(x)
        out = self.ln_rnn(out)
        
        if not self.return_sequence:
            context_vector = self.attention(out)
            return self.fc_projection(context_vector)
        else:
            return self.step_projection(out)


if __name__ == "__main__":
    print("Inicializando Advanced Temporal Encoder...")
    
    # Parámetros de prueba
    BATCH_SIZE = 32
    SEQ_LENGTH = 60
    NUM_FEATURES = 125
    HIDDEN_SIZE = 64
    EMBEDDING_DIM = 64
    
    torch.manual_seed(42)
    synthetic_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES)
    
    try:
        # Instanciar modelo (Causal por defecto)
        model = BiLSTMEncoder(
            input_size=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            dropout=0.3,
            embedding_dim=EMBEDDING_DIM,
            bidirectional=False # Strict causality
        )
        
        model.eval()
        
        with torch.no_grad():
            print(f"Configuración: Causal=True, RNN=LSTM, Layers=2")
            print(f"Input shape: {synthetic_input.shape}")
            
            output_embedding = model(synthetic_input)
            
            print(f"Output shape: {output_embedding.shape}")
            
            # Verificaciones
            assert output_embedding.shape == (BATCH_SIZE, EMBEDDING_DIM)
            assert not torch.isnan(output_embedding).any()
            
            print("\n[OK] Codificador avanzado ejecutado exitosamente.")
            
            # Prueba de JIT (TorchScript)
            print("Verificando compatibilidad con TorchScript...")
            scripted_model = torch.jit.script(model)
            scripted_output = scripted_model(synthetic_input)
            assert torch.allclose(output_embedding, scripted_output)
            print("[OK] Modelo compatible con TorchScript para producción.")
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()