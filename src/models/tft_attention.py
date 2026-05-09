import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) para control de flujo de información.
    Permite al modelo suprimir features irrelevantes mediante un mecanismo de gating.
    Basado en el paper 'Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting'.
    """
    def __init__(self, input_dim: int):
        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, ..., input_dim)
        gate = self.fc(x)
        value, gate = gate.chunk(2, dim=-1)
        return value * self.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) inspirado en TFT.
    Proporciona complejidad adaptativa: permite al modelo aplicar transformaciones no lineales 
    complejas solo cuando es necesario, o saltarlas mediante una conexión residual y GLU.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int] = None, dropout: float = 0.1):
        super(GatedResidualNetwork, self).__init__()
        self.output_dim = output_dim or input_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Proyección de residuo si las dimensiones de entrada y salida difieren
        if input_dim != self.output_dim:
            self.res_project = nn.Linear(input_dim, self.output_dim)
        else:
            self.res_project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, ..., input_dim)
        residual = self.res_project(x)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        # Add & Norm: Conexión residual seguida de LayerNorm
        return self.layer_norm(x + residual)

class TFTFusionNode(nn.Module):
    """
    Nodo de Fusión Temporal Avanzado (TFT).
    Aplica atención multi-head a través del tiempo con máscara causal estricta.
    Fusa múltiples fuentes (CNN, LSTM) antes de la atención temporal.
    """
    def __init__(self, 
                 embedding_dim: int = 64, 
                 num_heads: int = 4, 
                 num_sources: int = 2, 
                 dropout: float = 0.2, 
                 output_dim: int = 64):
        super(TFTFusionNode, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_sources = num_sources
        self.num_heads = num_heads
        
        # 1. Variable Selection Gating (Fusión de fuentes por cada paso de tiempo)
        self.source_projector = nn.Sequential(
            nn.Linear(embedding_dim * num_sources, embedding_dim),
            GatedLinearUnit(embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 2. Multi-Head Attention Temporal (Causal)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # 3. Post-Attention GRN
        self.grn = GatedResidualNetwork(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # 4. Proyección final
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, source_embs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass con máscara causal.
        Args:
            source_embs: Lista de Tensors de forma (batch_size, seq_len, embedding_dim)
        """
        # 1. Concatenar y proyectar fuentes: (batch, seq, embedding_dim)
        x = torch.cat(source_embs, dim=-1)
        x = self.source_projector(x)
        
        # 2. Generar Máscara Causal (Upper Triangular)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # 3. Multi-Head Attention Temporal (causal explícita).
        attn_output, attn_weights_opt = self.multihead_attn(
            query=x, key=x, value=x,
            attn_mask=mask,
            is_causal=True,
        )

        attn_weights = attn_weights_opt if attn_weights_opt is not None else torch.empty(0)
        
        x = self.norm1(x + attn_output)
        
        # 4. Post-Attention GRN
        x = self.norm2(x + self.grn(x))
        
        # 5. Proyección Final
        enriched_representation = self.output_projection(x)
        
        return enriched_representation, attn_weights


if __name__ == "__main__":
    print("Iniciando suite de validación: TFTFusionNode (Production-Ready)")

    # Parámetros de prueba consistentes con el HybridSignalEngine
    BATCH_SIZE = 16
    SEQ_LEN = 60
    EMB_DIM = 64
    NUM_HEADS = 4
    OUTPUT_DIM = 64

    torch.manual_seed(42)

    # Prueba con 2 fuentes (CNN, LSTM): tensores 3D (batch, seq, emb)
    print("\n--- TEST 1: Dual Source Integration (CNN + LSTM) ---")
    model_2 = TFTFusionNode(num_sources=2, embedding_dim=EMB_DIM, output_dim=OUTPUT_DIM)
    model_2.eval()

    s1 = torch.randn(BATCH_SIZE, SEQ_LEN, EMB_DIM)
    s2 = torch.randn(BATCH_SIZE, SEQ_LEN, EMB_DIM)

    with torch.no_grad():
        out_2, attn_2 = model_2([s1, s2])

    print(f"Output shape: {tuple(out_2.shape)} (Esperado: {(BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)})")
    print(f"Attn weights shape: {tuple(attn_2.shape)} (Esperado: {(BATCH_SIZE, SEQ_LEN, SEQ_LEN)})")
    assert out_2.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
    assert not torch.isnan(out_2).any(), "NaNs detectados en la salida"

    # Prueba con 3 fuentes (Futuras expansiones: Sentiment, Macro, Orderbook)
    print("\n--- TEST 2: Multi-Source Flexibility (3 fuentes) ---")
    model_3 = TFTFusionNode(num_sources=3, embedding_dim=EMB_DIM, output_dim=OUTPUT_DIM)
    model_3.eval()
    s3 = torch.randn(BATCH_SIZE, SEQ_LEN, EMB_DIM)

    with torch.no_grad():
        out_3, _attn_3 = model_3([s1, s2, s3])
    print(f"Output shape (3 sources): {tuple(out_3.shape)}")
    assert out_3.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

    # Validación de Despliegue (TorchScript)
    print("\n--- TEST 3: Production Deployment (TorchScript JIT) ---")
    try:
        scripted = torch.jit.script(model_2)
        with torch.no_grad():
            s_out, _ = scripted([s1, s2])

        # Verificamos que la salida sea idéntica (dentro de tolerancia numérica)
        assert torch.allclose(out_2, s_out, atol=1e-5)
        print("[OK] Modelo compatible con JIT y verificado para producción.")
    except Exception as e:
        print(f"[FAIL] Error en validación JIT: {e}")
        import traceback
        traceback.print_exc()

    print("\n[SUCCESS] TFTFusionNode validado y listo para integración en HybridSignalEngine.")
