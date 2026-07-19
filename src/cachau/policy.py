"""Cache policy layer: bounded-memory bookkeeping.

The policy decides *which* keys must go; it never touches storage itself —
the decorator applies the returned evictions to the backend, keeping policy
and storage responsibilities separate (GUIDELINES.md §13).
"""

from __future__ import annotations

from collections import OrderedDict


class LRUBudget:
    """Tracks one function's entry sizes against a byte budget, in LRU order."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._sizes: OrderedDict[str, int] = OrderedDict()
        self._total = 0

    def fits(self, size: int) -> bool:
        return size <= self.max_bytes

    def touch(self, key: str) -> None:
        """Refresh a key's recency on a cache hit."""
        if key in self._sizes:
            self._sizes.move_to_end(key)

    def forget(self, key: str) -> None:
        size = self._sizes.pop(key, None)
        if size is not None:
            self._total -= size

    def admit(self, key: str, size: int) -> tuple[str, ...]:
        """Register ``key`` and return the LRU keys that must be evicted first."""
        self.forget(key)
        evicted: list[str] = []
        while self._total + size > self.max_bytes and self._sizes:
            oldest_key, oldest_size = self._sizes.popitem(last=False)
            self._total -= oldest_size
            evicted.append(oldest_key)
        self._sizes[key] = size
        self._total += size
        return tuple(evicted)

    def reset(self) -> None:
        self._sizes = OrderedDict()
        self._total = 0
