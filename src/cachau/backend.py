"""The minimal storage contract every backend implements.

Backends store and retrieve entries; they never decide function semantics
(keys, invalidation, TTL policy) — that belongs to the layers above
(GUIDELINES.md §13).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class CacheEntry:
    """An immutable stored result plus the metadata the policy layers need."""

    value: Any
    namespace: str
    created_at: float = field(default_factory=time.time)


class CacheBackend(Protocol):
    def get(self, key: str) -> CacheEntry | None: ...

    def set(self, key: str, entry: CacheEntry) -> None: ...

    def delete(self, key: str) -> None: ...

    def clear(self, namespace: str | None = None) -> None: ...

    def iter_entries(self) -> Iterator[tuple[str, CacheEntry]]: ...
