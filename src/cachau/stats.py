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
    coalesced_hits: int
    misses: int
    hit_rate: float
    miss_not_found: int
    miss_expired: int
    miss_invalidated: int
    miss_dependency_changed: int
    # Sampled verification (verify=): the cached value did not match a fresh
    # recompute — served as the fresh value, counted as a miss with this reason.
    miss_verification_failed: int
    expirations: int
    writes: int
    skipped_writes: int
    skipped_oversized: int
    # Results whose declared dependencies changed while the function was running:
    # returned to the caller but never cached, since no stable fingerprint
    # describes them (would otherwise risk a false HIT). Counted in skipped_writes.
    dependency_race_skips: int
    size_estimate_failures: int
    write_errors: int
    delete_errors: int
    evictions: int
    invalidations: int
    code_change_invalidations: int
    entries: int
    current_bytes: int
    # verify= sampling: how many HITs were recomputed for comparison, and how
    # many of those comparisons failed (each failure is also a
    # miss_verification_failed).
    verifications: int
    verification_failures: int
    # coalesce="processes": hits served by waiting on another process's
    # compute, bounded waits that expired (the caller computed anyway), and
    # stale lock files broken by age.
    process_coalesced_hits: int
    process_flight_timeouts: int
    stale_locks_broken: int
    total_compute_seconds: float
    estimated_saved_seconds: float
    # First computation of a JIT dispatcher includes one-time compilation;
    # it is reported separately and never folded into savings estimates.
    cold_compute_seconds: float
