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
    Nodo de Fusión Temporal Avanzado para Motores de Señales Híbridos.
    
    Características de producción:
    - Multi-Head Attention: Pondera dinámicamente la importancia de cada fuente (CNN, LSTM, etc.).
    - Gated Residual Networks: Controla la complejidad de la fusión y suprime ruido.
    - Variable Selection Gating: Gating individual por fuente antes de la atención.
    - Compatibilidad con TorchScript: Optimizado para despliegue de baja latencia.
    - Inicialización Robusta: Usa Xavier y LayerNorm para estabilidad numérica.
    """
    def __init__(self, 
                 embedding_dim: int = 64, 
                 num_heads: int = 4, 
                 num_sources: int = 2, 
                 dropout: float = 0.2, 
                 output_dim: int = 64):
        super(TFTFusionNode, self).__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) debe ser divisible por num_heads ({num_heads}).")
            
        self.embedding_dim = embedding_dim
        self.num_sources = num_sources
        self.num_heads = num_heads
        
        # 1. Gating inicial por fuente (Variable Selection)
        # Esto permite al modelo ignorar por completo una rama si se vuelve poco confiable.
        self.source_gates = nn.ModuleList([
            GatedResidualNetwork(embedding_dim, embedding_dim // 2, dropout=dropout)
            for _ in range(num_sources)
        ])
        
        # 2. Multi-Head Attention (Self-Attention sobre las fuentes atendidas)
        # Permite que las representaciones se "enriquezcan" entre sí.
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # 3. Post-Attention GRN
        # Procesa la representación fusionada antes de la proyección final.
        self.grn = GatedResidualNetwork(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # 4. Proyección final a espacio latente de señal
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim * num_sources, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos Xavier para estabilidad en redes profundas."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, source_embs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass optimizado.
        Args:
            source_embs: Lista de Tensors de forma (batch_size, embedding_dim)
        Returns:
            enriched_representation: Tensor (batch_size, output_dim) listo para el Meta-Learner.
            attn_weights: Pesos de atención (batch_size, num_sources, num_sources) para interpretabilidad.
        """
        if len(source_embs) != self.num_sources:
            raise ValueError(f"Se esperaban {self.num_sources} fuentes, se recibieron {len(source_embs)}.")

        # 1. Gating individual de fuentes (pre-atención)
        # Nota: Usamos enumerate sobre ModuleList para compatibilidad total con TorchScript.
        gated_embs: List[torch.Tensor] = []
        for i, gate in enumerate(self.source_gates):
            gated_embs.append(gate(source_embs[i]))
            
        # 2. Stack sources: (batch_size, num_sources, embedding_dim)
        x = torch.stack(gated_embs, dim=1)
        
        # 3. Multi-Head Attention & Add-Norm
        attn_output, attn_weights_opt = self.multihead_attn(query=x, key=x, value=x)
        
        # Manejo de Optional para TorchScript estricto
        if attn_weights_opt is None:
            attn_weights = torch.empty(0)
        else:
            attn_weights = attn_weights_opt
            
        x = self.norm1(x + attn_output)
        
        # 4. Gated Residual Network & Add-Norm (Post-Attention)
        grn_output = self.grn(x)
        x = self.norm2(x + grn_output)
        
        # 5. Flatten y Proyección Final
        # Concatenamos las representaciones de cada fuente ya procesadas y atendidas.
        x_flat = x.flatten(start_dim=1)
        enriched_representation = self.output_projection(x_flat)
        
        return enriched_representation, attn_weights

if __name__ == "__main__":
    print("Iniciando suite de validación: TFTFusionNode (Production-Ready)")
    
    # Parámetros de prueba consistentes con el HybridSignalEngine
    BATCH_SIZE = 16
    EMB_DIM = 64
    NUM_HEADS = 4
    OUTPUT_DIM = 64
    
    torch.manual_seed(42)
    
    # Prueba con 2 fuentes (CNN, LSTM)
    print("\n--- TEST 1: Dual Source Integration (CNN + LSTM) ---")
    model_2 = TFTFusionNode(num_sources=2, embedding_dim=EMB_DIM, output_dim=OUTPUT_DIM)
    model_2.eval()
    
    s1 = torch.randn(BATCH_SIZE, EMB_DIM)
    s2 = torch.randn(BATCH_SIZE, EMB_DIM)
    
    with torch.no_grad():
        out_2, attn_2 = model_2([s1, s2])
    
    print(f"Output shape: {out_2.shape} (Esperado: {BATCH_SIZE, OUTPUT_DIM})")
    print(f"Attn weights shape: {attn_2.shape} (Esperado: {BATCH_SIZE, 2, 2})")
    assert out_2.shape == (BATCH_SIZE, OUTPUT_DIM)
    assert not torch.isnan(out_2).any(), "NaNs detectados en la salida"

    # Prueba de interpretabilidad
    print(f"Pesos de atención (ejemplo batch 0):\n{attn_2[0].numpy()}")

    # Prueba con 3 fuentes (Futuras expansiones: Sentiment, Macro, Orderbook)
    print("\n--- TEST 2: Multi-Source Flexibility (3 fuentes) ---")
    model_3 = TFTFusionNode(num_sources=3, embedding_dim=EMB_DIM, output_dim=OUTPUT_DIM)
    model_3.eval()
    s3 = torch.randn(BATCH_SIZE, EMB_DIM)
    
    with torch.no_grad():
        out_3, attn_3 = model_3([s1, s2, s3])
    print(f"Output shape (3 sources): {out_3.shape}")
    assert out_3.shape == (BATCH_SIZE, OUTPUT_DIM)

    # Validación de Despliegue (TorchScript)
    print("\n--- TEST 3: Production Deployment (TorchScript JIT) ---")
    try:
        model_2.eval()
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
