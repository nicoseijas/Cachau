"""The minimal storage contract every backend implements.

Backends store and retrieve entries; they never decide function semantics
(keys, invalidation, TTL policy) — that belongs to the layers above
(GUIDELINES.md §13).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, NamedTuple, Protocol


class EntryMetadata(NamedTuple):
    """Lightweight per-entry facts readable without deserializing the value."""

    key: str
    namespace: str
    size: int | None
    # Creation instant, when the backend records one. The only ordering a
    # store preserves across restarts: real LRU recency is process-local and
    # dies with the process, so rehydration falls back to age.
    created_at: float | None = None
    # Expiry instant and declared-dependency fingerprints, both readable from
    # the entry header without deserializing the payload — so inspect() can show
    # TTL and dependency state cheaply over a persistent cache. Optional so a
    # third-party backend that only records key/namespace/size still conforms.
    expires_at: float | None = None
    dependency_fingerprints: dict[str, str] | None = None


@dataclass(frozen=True)
class CacheEntry:
    """An immutable stored result plus the metadata the policy layers need."""

    value: Any
    namespace: str
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size: int | None = None
    # Fingerprints of the external dependencies declared via ``depends_on=`` at
    # the moment this result was computed, keyed by dependency label. ``None``
    # means the function declared no dependencies. Compared on read: a mismatch
    # is a ``dependency_changed`` miss (GUIDELINES.md §7, §8).
    dependency_fingerprints: dict[str, str] | None = None

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class CacheBackend(Protocol):
    def get(self, key: str) -> CacheEntry | None: ...

    def peek(self, key: str) -> CacheEntry | None:
        """Like ``get`` but guaranteed side-effect free (no corrupt-file
        cleanup, no bookkeeping). Used by pure observers such as explain()."""
        ...

    def set(self, key: str, entry: CacheEntry) -> None: ...

    def delete(self, key: str) -> None: ...

    def clear(self, namespace: str | None = None) -> None: ...

    def iter_entries(self) -> Iterator[tuple[str, CacheEntry]]: ...

    def iter_metadata(self) -> Iterator[EntryMetadata]: ...
