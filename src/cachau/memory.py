"""In-memory backend: a plain process-local store."""

from __future__ import annotations

from typing import Iterator

from cachau.backend import CacheEntry, EntryMetadata


class MemoryBackend:
    """Dictionary-backed storage. Unbounded in Phase 0; limits arrive with LRU."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        return self._entries.get(key)

    def set(self, key: str, entry: CacheEntry) -> None:
        self._entries[key] = entry

    def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._entries = {}
        else:
            self._entries = {
                key: entry
                for key, entry in self._entries.items()
                if entry.namespace != namespace
            }

    def iter_entries(self) -> Iterator[tuple[str, CacheEntry]]:
        yield from list(self._entries.items())

    def iter_metadata(self) -> Iterator[EntryMetadata]:
        for key, entry in list(self._entries.items()):
            yield EntryMetadata(key, entry.namespace, entry.size)
