"""The immutable statistics snapshot returned by ``func.cache.stats()``.

Metrics observe without changing behavior (GUIDELINES.md §8, §13). Misses
always carry a reason — a cache that can't say *why* something was a miss
can't be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    hit_rate: float
    miss_not_found: int
    miss_expired: int
    miss_invalidated: int
    expirations: int
    writes: int
    skipped_writes: int
    skipped_oversized: int
    size_estimate_failures: int
    write_errors: int
    delete_errors: int
    evictions: int
    invalidations: int
    code_change_invalidations: int
    entries: int
    current_bytes: int
    total_compute_seconds: float
    estimated_saved_seconds: float
