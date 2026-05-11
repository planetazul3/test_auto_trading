"""Token bucket asíncrono para respetar el límite de 100 req/s por conexión."""

from __future__ import annotations

import asyncio
import time


class AsyncTokenBucket:
    """Cubeta de tokens con relleno continuo, segura entre tareas asyncio."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = float(rate)
        self.capacity = float(capacity if capacity is not None else rate)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        if amount <= 0:
            raise ValueError("amount must be > 0")
        if amount > self.capacity:
            raise ValueError("amount exceeds bucket capacity")
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                wait = (amount - self._tokens) / self.rate
                await asyncio.sleep(wait)

    @property
    def available_tokens(self) -> float:
        now = time.monotonic()
        return min(self.capacity, self._tokens + (now - self._last) * self.rate)
