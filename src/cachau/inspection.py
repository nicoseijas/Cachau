"""``func.cache.inspect()`` — browse what a function currently has cached.

A cache you cannot look inside is hard to trust (GUIDELINES.md §8, §12). Where
``stats()`` aggregates and ``explain()`` answers one call, ``inspect()`` lists the
entries themselves: when each was computed, how big it is, how much TTL it has
left, and which dependency fingerprints it carries — all read from entry headers
without deserializing a single payload, so it stays cheap over a large persistent
cache and, like ``explain()``, never mutates anything.

Returns an :class:`Inspection`: a plain read-only sequence of
:class:`CacheEntryView`, with a table ``__str__`` that reads naturally in a
notebook cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from cachau.explanation import _format_bytes, _format_duration


@dataclass(frozen=True)
class CacheEntryView:
    """One cached entry, observed without deserializing its value."""

    key: str
    namespace: str
    checked_at: float
    created_at: float | None = None
    expires_at: float | None = None
    size_bytes: int | None = None
    dependency_fingerprints: dict[str, str] | None = None

    @property
    def digest(self) -> str:
        """The argument digest — the part of the key that identifies the call."""
        return self.key.rsplit(":", 1)[-1]

    @property
    def age_seconds(self) -> float | None:
        if self.created_at is None:
            return None
        return self.checked_at - self.created_at

    @property
    def ttl_remaining_seconds(self) -> float | None:
        if self.expires_at is None or self.checked_at >= self.expires_at:
            return None
        return self.expires_at - self.checked_at

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.checked_at >= self.expires_at

    def __str__(self) -> str:
        parts = [self.digest[:16]]
        if self.size_bytes is not None:
            parts.append(_format_bytes(self.size_bytes))
        if self.created_at is not None:
            parts.append(f"age {_format_duration(self.age_seconds or 0.0)}")
        if self.is_expired:
            parts.append("EXPIRED")
        elif self.ttl_remaining_seconds is not None:
            parts.append(f"ttl {_format_duration(self.ttl_remaining_seconds)}")
        if self.dependency_fingerprints:
            parts.append(f"deps {','.join(sorted(self.dependency_fingerprints))}")
        return "  ".join(parts)


@dataclass(frozen=True)
class Inspection(Sequence[CacheEntryView]):
    """A read-only, notebook-friendly listing of a function's cached entries."""

    namespace: str
    entries: tuple[CacheEntryView, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CacheEntryView]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> CacheEntryView:  # type: ignore[override]
        return self.entries[index]

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries if e.size_bytes is not None)

    def __str__(self) -> str:
        if not self.entries:
            return f"No cached entries for {self.namespace}"
        count = len(self.entries)
        header = (
            f"{count} cached {'entry' if count == 1 else 'entries'} for "
            f"{self.namespace} ({_format_bytes(self.total_bytes)})"
        )
        lines = [header, ""]
        for entry in self.entries:
            lines.append(f"  {entry}")
        return "\n".join(lines)
