"""Embeddings condicionantes para soporte cross-asset y cross-timeframe.

El motor de señales debe funcionar para cualquier símbolo (synthetic
indices, forex, commodities…) y para cualquier granularidad ofrecida
por Deriv (ticks + candles ``{60s … 86400s}``). Esta clase encapsula
el catálogo dinámico de símbolos y granularidades y proyecta cada
``(symbol_id, granularity_id)`` a un vector compacto que el backbone
puede consumir como contexto (e.g. en ``GatedResidualNetwork``).

Todo es dinámico: los IDs se asignan al vuelo via ``register_symbol``
/``register_granularity`` y se persisten en el ``state_dict``. Las
granularidades soportadas se inicializan a partir de las constantes del
conector Deriv si están disponibles, y se aceptan extras (``None`` →
ticks, granularidad=0 por convención).
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


# Granularidades documentadas por Deriv API v2 (segundos).
DERIV_GRANULARITIES_SECONDS: tuple[int, ...] = (
    0,  # convención interna: 0 = ticks
    60, 120, 180, 300, 600, 900, 1800,
    3600, 7200, 14400, 28800, 86400,
)


class _DynamicVocab:
    """Vocabulario id↔token con asignación on-the-fly."""

    def __init__(self, initial: Iterable[str] = ()) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        for tok in initial:
            self.register(tok)

    def register(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        idx = len(self._id_to_token)
        self._token_to_id[token] = idx
        self._id_to_token.append(token)
        return idx

    def __len__(self) -> int:
        return len(self._id_to_token)

    def to_id(self, token: str) -> int:
        if token not in self._token_to_id:
            raise KeyError(f"unknown token {token!r}; call .register first")
        return self._token_to_id[token]

    def to_token(self, idx: int) -> str:
        return self._id_to_token[idx]


class AssetTimeframeEmbedding(nn.Module):
    """Combina símbolo y granularidad en un único vector contextual.

    Parameters
    ----------
    embedding_dim:
        Dimensión total del vector de contexto resultante. Se reparte a
        partes iguales entre símbolo y granularidad (la última columna
        absorbe el resto si ``embedding_dim`` es impar).
    max_symbols / max_granularities:
        Tamaños del vocabulario (override-able si se conoce de antemano).
        Por defecto se reservan slots generosos para evitar resizing.
    initial_symbols / initial_granularities:
        Pre-poblan los vocabularios. Las granularidades por defecto son
        las soportadas por la Deriv API v2.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        *,
        max_symbols: int = 1024,
        max_granularities: int = 64,
        initial_symbols: Iterable[str] = (),
        initial_granularities: Iterable[int] = DERIV_GRANULARITIES_SECONDS,
    ) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2")
        if max_symbols <= 0 or max_granularities <= 0:
            raise ValueError("vocab sizes must be > 0")

        self.embedding_dim = embedding_dim
        sym_dim = embedding_dim // 2
        gran_dim = embedding_dim - sym_dim
        self.symbol_dim = sym_dim
        self.granularity_dim = gran_dim

        self.symbol_emb = nn.Embedding(max_symbols, sym_dim)
        self.granularity_emb = nn.Embedding(max_granularities, gran_dim)

        nn.init.normal_(self.symbol_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.granularity_emb.weight, mean=0.0, std=0.02)

        self.symbol_vocab = _DynamicVocab(initial_symbols)
        self.granularity_vocab = _DynamicVocab(
            str(int(g)) for g in initial_granularities
        )
        self.max_symbols = max_symbols
        self.max_granularities = max_granularities

    # ------------------------------------------------------------------
    # Catálogo dinámico
    # ------------------------------------------------------------------

    def register_symbol(self, symbol: str) -> int:
        idx = self.symbol_vocab.register(symbol)
        if idx >= self.max_symbols:
            raise ValueError(
                f"symbol vocab exceeded max_symbols={self.max_symbols}; "
                "re-instantiate with a larger budget"
            )
        return idx

    def register_granularity(self, granularity_seconds: Optional[int]) -> int:
        # ``None`` → ticks (granularidad 0 por convención interna).
        key = str(int(0 if granularity_seconds is None else granularity_seconds))
        idx = self.granularity_vocab.register(key)
        if idx >= self.max_granularities:
            raise ValueError(
                f"granularity vocab exceeded max_granularities={self.max_granularities}"
            )
        return idx

    def symbol_id(self, symbol: str) -> int:
        return self.symbol_vocab.to_id(symbol)

    def granularity_id(self, granularity_seconds: Optional[int]) -> int:
        key = str(int(0 if granularity_seconds is None else granularity_seconds))
        return self.granularity_vocab.to_id(key)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        symbol_ids: torch.Tensor,
        granularity_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``(B,)`` IDs → ``(B, embedding_dim)``."""
        if symbol_ids.shape != granularity_ids.shape:
            raise ValueError("symbol_ids and granularity_ids must share shape")
        if symbol_ids.dim() != 1:
            raise ValueError("expected 1-D batch of IDs")
        s = self.symbol_emb(symbol_ids)
        g = self.granularity_emb(granularity_ids)
        return torch.cat([s, g], dim=-1)


__all__ = [
    "AssetTimeframeEmbedding",
    "DERIV_GRANULARITIES_SECONDS",
]
