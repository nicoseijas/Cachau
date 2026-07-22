"""The answer returned by ``func.cache.explain(...)``.

Explaining is pure observation: it never executes the cached function, never
mutates cache state, never touches counters or LRU recency. The one exception
is a declared ``token(callable)`` dependency, which is evaluated to check for a
change — the caveat the user owns (see ``CacheControl.explain``). The system
should be invisible when it works and fully transparent when you need to
understand it.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class Explanation:
    outcome: str  # "HIT" | "MISS"
    # "found" | "not_found" | "expired" | "invalidated" | "dependency_changed"
    # | "evicted"
    reason: str
    key: str
    namespace: str
    fingerprint: str
    checked_at: float
    created_at: float | None = None
    expires_at: float | None = None
    size_bytes: int | None = None
    # Which declared dependencies changed, when reason == "dependency_changed".
    changed_dependencies: tuple[str, ...] | None = None
    # Per-changed-dependency fingerprint diff: {label: (stored, current)}, where
    # a ``None`` side means the dependency was absent then or now. Lets explain()
    # answer not just WHICH dependency changed but HOW.
    dependency_diff: dict[str, tuple[str | None, str | None]] | None = None
    # Set on a "not_found" miss when this function has already failed cache
    # writes. A failed write leaves nothing behind to find, so without this a
    # broken store is indistinguishable from a cold cache. Function-wide tally,
    # not proof that THIS key's write failed.
    write_errors: int | None = None

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
    def expired_seconds_ago(self) -> float | None:
        if self.reason != "expired" or self.expires_at is None:
            return None
        return self.checked_at - self.expires_at

    def __str__(self) -> str:
        lines = [self.outcome, f"Reason:      {self.reason}"]
        if self.write_errors:
            plural = "s" if self.write_errors != 1 else ""
            lines.append(
                f"Warning:     {self.write_errors} cache write{plural} failed; "
                "this may be a broken store, not a cold cache"
            )
        if self.dependency_diff:
            for label, (stored, current) in sorted(self.dependency_diff.items()):
                before = stored if stored is not None else "<none>"
                after = current if current is not None else "<none>"
                lines.append(f"Changed dep: {label} ({before} -> {after})")
        elif self.changed_dependencies:
            lines.append(f"Changed dep: {', '.join(self.changed_dependencies)}")
        lines.append(f"Namespace:   {self.namespace}")
        if self.created_at is not None:
            lines.append(f"Created:     {_format_timestamp(self.created_at)}")
        age = self.age_seconds
        if age is not None and self.outcome == "HIT":
            lines.append(f"Age:         {_format_duration(age)}")
        remaining = self.ttl_remaining_seconds
        if remaining is not None:
            lines.append(f"Expires in:  {_format_duration(remaining)}")
        expired_ago = self.expired_seconds_ago
        if expired_ago is not None:
            lines.append(
                f"Expired:     {_format_duration(expired_ago)} ago "
                f"(at {_format_timestamp(self.expires_at)})"
            )
        if self.size_bytes is not None:
            lines.append(f"Size:        {_format_bytes(self.size_bytes)}")
        return "\n".join(lines)


def _format_timestamp(timestamp: float | None) -> str:
    # UTC with an explicit label: persisted entries travel across machines
    # and timezones, so an unlabeled local rendering would be ambiguous.
    if timestamp is None:
        return "unknown"
    try:
        moment = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return "unknown"
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _format_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"
