"""Extractor de patrones locales mediante convoluciones 1D causales.

Diseño:

* ``CausalConv1d`` valida que ``stride == 1`` — con ``stride>1`` el
  padding ``(k-1)*d`` rompe la causalidad silenciosamente y se filtra
  información futura.
* La normalización es ``ChannelLayerNorm`` (LayerNorm aplicado por
  timestep sobre el eje de canales) — **no** ``GroupNorm``, porque ese
  último mezcla estadísticas a lo largo del eje temporal, rompiendo la
  causalidad estricta en inferencia (el valor en t depende del futuro
  via la media móvil de canales). ``ChannelLayerNorm`` normaliza
  ``(B, C, t)`` independientemente para cada ``t``.
* ``CNN1DExtractor`` es **totalmente paramétrico**: canales, kernels y
  dilataciones se exponen al caller. ``group_norm_groups`` se mantiene
  como parámetro deprecado (ignorado) para compat con código existente.
* Dropout consistente entre las dos ramas (``return_sequence=True/False``).
* Inicialización Xavier para activaciones GELU.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _auto_groups(num_channels: int, requested: int) -> int:
    if num_channels <= 0:
        raise ValueError("num_channels must be > 0")
    if requested <= 0:
        return 1
    g = math.gcd(num_channels, requested)
    return max(1, g)


class ChannelLayerNorm(nn.Module):
    """LayerNorm causal sobre canales: normaliza ``(B, C, t)`` por ``t``."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L). LayerNorm espera el eje de features al final.
        out: torch.Tensor = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return out


class CausalConv1d(nn.Conv1d):
    """Convolución 1D causal: padding solo a la izquierda.

    Sólo se soporta ``stride=1``; con stride>1 la salida[t] depende de
    inputs en t+stride-1, lo cual rompe la causalidad.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        self._left_pad = (kernel_size - 1) * dilation
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._left_pad, 0))
        return super().forward(x)


class CNN1DExtractor(nn.Module):
    """Extrae embeddings locales de una ventana ``(B, S, F)``.

    Parameters
    ----------
    num_features:
        Número de features de entrada por paso de tiempo.
    sequence_length:
        Longitud máxima de la ventana (informativo; el módulo es
        agnóstico al ``S`` exacto en runtime).
    embedding_dim:
        Dimensión del embedding de salida.
    channels:
        Lista de canales por bloque convolucional. Default
        ``(64, 128)`` para mantener compatibilidad con la versión previa.
    kernel_sizes:
        Kernel por bloque (mismo largo que ``channels``).
    dilations:
        Dilatación por bloque (mismo largo que ``channels``).
    group_norm_groups:
        Grupos solicitados para ``GroupNorm``. Se ajusta al gcd con el
        número de canales para evitar fallos en runtime.
    pool_output_size:
        Tamaño de salida del ``AdaptiveMaxPool1d`` cuando
        ``return_sequence=False``.
    dropout:
        Dropout aplicado consistentemente en ambas ramas (collapsada y
        secuencial).
    return_sequence:
        Si ``True``, devuelve ``(B, S, embedding_dim)``; si ``False``,
        ``(B, embedding_dim)``.
    """

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        embedding_dim: int = 64,
        *,
        channels: Sequence[int] = (64, 128),
        kernel_sizes: Sequence[int] = (3, 3),
        dilations: Sequence[int] = (1, 2),
        group_norm_groups: int = 8,
        pool_output_size: int = 4,
        dropout: float = 0.3,
        return_sequence: bool = False,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if pool_output_size <= 0:
            raise ValueError("pool_output_size must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        channels = tuple(channels)
        kernel_sizes = tuple(kernel_sizes)
        dilations = tuple(dilations)
        if not channels:
            raise ValueError("channels must be non-empty")
        if not (len(channels) == len(kernel_sizes) == len(dilations)):
            raise ValueError(
                "channels, kernel_sizes and dilations must have the same length"
            )

        self.num_features = num_features
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.return_sequence = return_sequence
        self.channels = channels

        blocks: list[nn.Module] = []
        in_ch = num_features
        for out_ch, k, d in zip(channels, kernel_sizes, dilations):
            # group_norm_groups se mantiene en la firma para compat pero ya no
            # se usa: ChannelLayerNorm es causal-safe.
            _ = _auto_groups(out_ch, group_norm_groups)
            blocks.append(
                nn.Sequential(
                    CausalConv1d(in_ch, out_ch, kernel_size=k, dilation=d),
                    ChannelLayerNorm(out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        last_ch = channels[-1]
        if not return_sequence:
            self.global_pool = nn.AdaptiveMaxPool1d(output_size=pool_output_size)
            flattened = last_ch * pool_output_size
            self.head = nn.Sequential(
                nn.Linear(flattened, max(embedding_dim * 4, embedding_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(embedding_dim * 4, embedding_dim), embedding_dim),
                nn.LayerNorm(embedding_dim),
            )
            self.step_head: Optional[nn.Module] = None
        else:
            self.global_pool = None  # type: ignore[assignment]
            self.head = None  # type: ignore[assignment]
            self.step_head = nn.Sequential(
                nn.Linear(last_ch, embedding_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(embedding_dim),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B,S,F), got {x.dim()}D")
        h = x.transpose(1, 2)  # (B, F, S)
        for block in self.blocks:
            h = block(h)

        if self.return_sequence:
            assert self.step_head is not None
            seq_out: torch.Tensor = self.step_head(h.transpose(1, 2))
            return seq_out
        assert self.global_pool is not None and self.head is not None
        pooled = self.global_pool(h).flatten(start_dim=1)
        head_out: torch.Tensor = self.head(pooled)
        return head_out


__all__ = ["CNN1DExtractor", "CausalConv1d", "ChannelLayerNorm"]
