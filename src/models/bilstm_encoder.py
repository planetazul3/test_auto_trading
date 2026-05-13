"""Codificador temporal LSTM/GRU con pooling de atención opcional.

Diseño:

* Nombre conservado por compatibilidad histórica; el default es
  ``bidirectional=False`` para mantener causalidad estricta sobre la
  ventana de entrada (mirar hacia atrás sería look-ahead).
* ``rnn_type`` validado explícitamente — falla rápido si recibe valores
  no soportados.
* Inicialización de pesos:
    - Xavier uniform en ``weight_ih`` (input-hidden)
    - Orthogonal en ``weight_hh`` (hidden-hidden) — recomendado para
      estabilidad del backprop en RNNs.
    - Forget-gate bias inicializado a 1.0 sólo en LSTM (la ranura
      ``[n//4:n//2]`` corresponde a ``f`` en el orden i,f,g,o de PyTorch).
* Dropout aplicado **consistentemente** en ambas ramas
  (``return_sequence=True/False``).
* Soporta ``lengths`` opcional para secuencias de longitud variable
  via ``pack_padded_sequence``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_RNN_TYPES = frozenset({"lstm", "gru"})


class AttentionPooling(nn.Module):
    """Pooling temporal por atención softmax (sin look-ahead respecto al
    instante de inferencia: toda la ventana es ``[t-w, t]``)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, S, H), key_padding_mask: (B, S) con True en posiciones a enmascarar
        scores = self.attn(x)  # (B, S, 1)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(-1), float("-inf")
            )
        weights = F.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


class BiLSTMEncoder(nn.Module):
    """Codificador LSTM/GRU configurable."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: Optional[int] = None,
        bidirectional: bool = False,
        rnn_type: str = "lstm",
        return_sequence: bool = False,
    ) -> None:
        """Codificador LSTM/GRU.

        ``embedding_dim`` controla **únicamente** la dimensión de la
        proyección final de salida — es independiente del estado
        interno del RNN (``hidden_size``). Si se deja ``None`` (default
        recomendado), se deriva como ``hidden_size * (2 if bidirectional
        else 1)``, evitando la colisión accidental ``embedding_dim ==
        hidden_size = 64`` que antes confundía al usuario.
        """
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be > 0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if embedding_dim is None:
            embedding_dim = hidden_size * (2 if bidirectional else 1)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        rnn_lower = rnn_type.lower()
        if rnn_lower not in _VALID_RNN_TYPES:
            raise ValueError(
                f"rnn_type must be one of {sorted(_VALID_RNN_TYPES)}, "
                f"got {rnn_type!r}"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence
        self.rnn_type = rnn_lower

        rnn_class = nn.LSTM if rnn_lower == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_size = hidden_size * (2 if bidirectional else 1)
        self.ln_rnn = nn.LayerNorm(rnn_out_size)

        if not return_sequence:
            self.attention = AttentionPooling(rnn_out_size)
            self.head = nn.Sequential(
                nn.Linear(rnn_out_size, rnn_out_size * 2),
                nn.GELU(),
                nn.LayerNorm(rnn_out_size * 2),
                nn.Dropout(dropout),
                nn.Linear(rnn_out_size * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            self.step_head: Optional[nn.Module] = None
        else:
            self.attention = None  # type: ignore[assignment]
            self.head = None  # type: ignore[assignment]
            self.step_head = nn.Sequential(
                nn.Linear(rnn_out_size, embedding_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(embedding_dim),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)
                # Forget gate bias = 1.0 sólo en LSTM (orden i,f,g,o).
                if self.rnn_type == "lstm" and "bias_ih" in name:
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        lengths: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B,S,F), got {x.dim()}D")

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out_packed, batch_first=True, total_length=x.size(1)
            )
        else:
            out, _ = self.rnn(x)
        out = self.ln_rnn(out)

        if self.return_sequence:
            assert self.step_head is not None
            seq_out: torch.Tensor = self.step_head(out)
            return seq_out

        assert self.attention is not None and self.head is not None
        context = self.attention(out, key_padding_mask=key_padding_mask)
        head_out: torch.Tensor = self.head(context)
        return head_out


__all__ = ["AttentionPooling", "BiLSTMEncoder"]
