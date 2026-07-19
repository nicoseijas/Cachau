"""Cache policy layer: bounded-memory bookkeeping.

The policy decides *which* keys must go; it never touches storage itself —
the decorator applies the returned evictions to the backend, keeping policy
and storage responsibilities separate (GUIDELINES.md §13).
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class LRUBudget:
    """Tracks one function's entry sizes against a byte budget, in LRU order.

    Thread-safe: single-flight only serializes callers of the *same* key, so
    different-key flights mutate this bookkeeping concurrently. The internal
    mutex guards only the fast dict operations, never a computation.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._sizes: OrderedDict[str, int] = OrderedDict()
        self._total = 0
        self._mutex = threading.Lock()

    def fits(self, size: int) -> bool:
        return size <= self.max_bytes

    def touch(self, key: str) -> None:
        """Refresh a key's recency on a cache hit."""
        with self._mutex:
            if key in self._sizes:
                self._sizes.move_to_end(key)

    def forget(self, key: str) -> None:
        with self._mutex:
            self._forget_locked(key)

    def _forget_locked(self, key: str) -> None:
        size = self._sizes.pop(key, None)
        if size is not None:
            self._total -= size

    def admit(self, key: str, size: int) -> tuple[str, ...]:
        """Register ``key`` and return the LRU keys that must be evicted first."""
        with self._mutex:
            self._forget_locked(key)
            evicted: list[str] = []
            while self._total + size > self.max_bytes and self._sizes:
                oldest_key, oldest_size = self._sizes.popitem(last=False)
                self._total -= oldest_size
                evicted.append(oldest_key)
            self._sizes[key] = size
            self._total += size
            return tuple(evicted)

    def reset(self) -> None:
        with self._mutex:
            self._sizes = OrderedDict()
            self._total = 0
