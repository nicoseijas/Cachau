"""The public ``@cache`` decorator and the per-function control surface."""

from __future__ import annotations

import functools
import os
import pathlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, TypeVar, overload

from cachau.backend import CacheBackend, CacheEntry, EntryMetadata
from cachau.disk import DiskBackend
from cachau.durations import parse_ttl
from cachau.errors import ConfigurationError
from cachau.explanation import Explanation
from cachau.fingerprint import function_fingerprint, function_namespace
from cachau.flight import KeyedLocks
from cachau.keys import digest_arguments
from cachau.memory import MemoryBackend
from cachau.policy import LRUBudget
from cachau.sizes import estimate_size, parse_size
from cachau.stats import CacheStats

F = TypeVar("F", bound=Callable[..., Any])
Clock = Callable[[], float]
SizeOf = Callable[[Any], int]
KeyBuilder = Callable[..., str]

_default_backend = MemoryBackend()
_DEFAULT_PERSIST_DIR = ".cachau"
_perf_counter = time.perf_counter
# Bounded cap for miss-reason attribution markers (label-only: dropping one
# can mislabel a future miss as not_found, never affect correctness).
_INVALIDATION_MARKER_CAP = 4096
_MARKER_MISSING = object()
Persist = bool | str | os.PathLike


class CacheControl:
    """Attached to every cached function as ``func.cache``."""

    def __init__(
        self,
        *,
        namespace: str,
        fingerprint: str,
        backend: CacheBackend,
        ttl_seconds: float | None,
        max_memory_bytes: int | None,
        budget: LRUBudget | None,
        key_builder: KeyBuilder,
        now: Clock,
        flights: KeyedLocks,
        code_change_invalidations: int = 0,
    ) -> None:
        self.namespace = namespace
        self.fingerprint = fingerprint
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_bytes
        self.hits = 0
        self.coalesced_hits = 0
        self.miss_not_found = 0
        self.miss_expired = 0
        self.miss_invalidated = 0
        self.writes = 0
        self.evictions = 0
        self.skipped_oversized = 0
        self.size_estimate_failures = 0
        self.write_errors = 0
        self.delete_errors = 0
        self.invalidations = 0
        self.code_change_invalidations = code_change_invalidations
        self.total_compute_seconds = 0.0
        self.compute_count = 0
        self.estimated_saved_seconds = 0.0
        # Reason markers: the delete succeeded; remembering the key only
        # attributes the next miss to miss_invalidated. Bounded LRU-style.
        self._invalidation_markers: OrderedDict[str, None] = OrderedDict()
        # Pending invalidations: the physical delete FAILED. Authoritative —
        # the wrapper must never serve these keys from the backend, or an
        # explicitly invalidated value would come back as a false HIT.
        self._pending_invalidations: set[str] = set()
        # Savings average uses only computations whose result was actually
        # cached; skipped/oversized computes can never produce a future hit.
        self._saved_basis_seconds = 0.0
        self._saved_basis_count = 0
        self._backend = backend
        self._budget = budget
        self._key_builder = key_builder
        self._now = now
        self._flights = flights
        # Guards compound mutations of the invalidation bookkeeping, which is
        # function-wide state reachable from concurrent flights of different
        # keys. Never held across backend I/O or a computation.
        self._mutation_guard = threading.Lock()

    def explain(self, *args: Any, **kwargs: Any) -> Explanation:
        """Explain what a call with these arguments would do, and why.

        Pure observation: never executes the function, never mutates cache
        state, counters, invalidation bookkeeping, or LRU recency.
        """
        key = self._key_builder(*args, **kwargs)
        checked_at = self._now()
        common = {
            "key": key,
            "namespace": self.namespace,
            "fingerprint": self.fingerprint,
            "checked_at": checked_at,
        }
        if key in self._pending_invalidations:
            return Explanation(outcome="MISS", reason="invalidated", **common)
        # peek() is the non-destructive read (DiskBackend.get removes corrupt
        # files as read-side maintenance; observation must not).
        peek = getattr(self._backend, "peek", self._backend.get)
        entry = peek(key)
        if entry is None:
            reason = (
                "invalidated" if key in self._invalidation_markers else "not_found"
            )
            return Explanation(outcome="MISS", reason=reason, **common)
        facts = {
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "size_bytes": entry.size,
        }
        if entry.is_expired(checked_at):
            return Explanation(outcome="MISS", reason="expired", **common, **facts)
        return Explanation(outcome="HIT", reason="found", **common, **facts)

    def clear(self) -> None:
        """Forget every stored result for this function."""
        self._backend.clear(namespace=self.namespace)
        if self._budget is not None:
            self._budget.reset()
        with self._mutation_guard:
            self._invalidation_markers.clear()
            self._pending_invalidations.clear()

    def invalidate(self, *args: Any, **kwargs: Any) -> None:
        """Forget the stored result for one specific invocation.

        Serialized with any in-flight computation of the same key (per-key
        lock), so it can never interleave with a commit and leave the budget
        or bookkeeping inconsistent with the backend.
        """
        key = self._key_builder(*args, **kwargs)
        with self._flights.holding(key):
            try:
                self._backend.delete(key)
            except Exception:  # noqa: BLE001 - a failed delete never raises
                self.delete_errors += 1
                with self._mutation_guard:
                    self._pending_invalidations.add(key)
            else:
                with self._mutation_guard:
                    self._invalidation_markers[key] = None
                    self._invalidation_markers.move_to_end(key)
                    while len(self._invalidation_markers) > _INVALIDATION_MARKER_CAP:
                        self._invalidation_markers.popitem(last=False)
            if self._budget is not None:
                self._budget.forget(key)
            self.invalidations += 1

    def stats(self) -> CacheStats:
        """An immutable snapshot of this function's cache activity."""
        entries = 0
        current_bytes = 0
        for row in self._iter_metadata():
            if row.namespace == self.namespace:
                entries += 1
                if row.size is not None:
                    current_bytes += row.size
        misses = self.miss_not_found + self.miss_expired + self.miss_invalidated
        total_calls = self.hits + misses
        return CacheStats(
            hits=self.hits,
            coalesced_hits=self.coalesced_hits,
            misses=misses,
            hit_rate=self.hits / total_calls if total_calls else 0.0,
            miss_not_found=self.miss_not_found,
            miss_expired=self.miss_expired,
            miss_invalidated=self.miss_invalidated,
            expirations=self.miss_expired,
            writes=self.writes,
            skipped_writes=(
                self.skipped_oversized
                + self.size_estimate_failures
                + self.write_errors
            ),
            skipped_oversized=self.skipped_oversized,
            size_estimate_failures=self.size_estimate_failures,
            write_errors=self.write_errors,
            delete_errors=self.delete_errors,
            evictions=self.evictions,
            invalidations=self.invalidations,
            code_change_invalidations=self.code_change_invalidations,
            entries=entries,
            current_bytes=current_bytes,
            total_compute_seconds=self.total_compute_seconds,
            estimated_saved_seconds=self.estimated_saved_seconds,
        )

    def _iter_metadata(self) -> Iterable[EntryMetadata]:
        iter_metadata = getattr(self._backend, "iter_metadata", None)
        if iter_metadata is not None:
            return iter_metadata()
        return (
            EntryMetadata(key, entry.namespace, entry.size)
            for key, entry in self._backend.iter_entries()
        )


@overload
def cache(func: F) -> F: ...


@overload
def cache(
    *,
    ttl: int | float | str | None = None,
    max_memory: int | str | None = None,
    persist: Persist | None = None,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
    clock: Clock = ...,
    size_of: SizeOf = ...,
) -> Callable[[F], F]: ...


def cache(
    func: Callable[..., Any] | None = None,
    *,
    ttl: int | float | str | None = None,
    max_memory: int | str | None = None,
    persist: Persist | None = None,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
    clock: Clock = time.time,
    size_of: SizeOf = estimate_size,
) -> Any:
    """Cache a function's results, keyed by its normalized arguments.

    Usable bare (``@cache``) or configured (``@cache(ttl="1h",
    max_memory="2GB", persist=True)``). TTL accepts seconds or readable
    strings and starts when the result is committed. ``max_memory`` accepts
    bytes or size strings (``"512MB"``, ``"2GB"``; binary units) and bounds
    this function's entries with LRU eviction — an entry larger than the whole
    budget is computed and returned but never cached. ``persist=True`` stores
    entries under ``./.cachau`` (or pass a directory) and survives process
    restarts; a failed write never loses the computed result. Exceptions are
    never cached; unhashable arguments fail loudly. ``func.cache`` exposes
    ``stats()``, ``clear()`` and ``invalidate(...)``.
    """
    if func is not None:
        return _wrap(func, ttl, max_memory, persist, namespace, backend, clock, size_of)
    return lambda f: _wrap(
        f, ttl, max_memory, persist, namespace, backend, clock, size_of
    )


def _resolve_backend(
    persist: Persist | None, backend: CacheBackend | None
) -> CacheBackend:
    if persist:
        if backend is not None:
            raise ConfigurationError(
                "persist= and backend= are mutually exclusive: persist creates "
                "a DiskBackend; pass a custom backend without persist instead"
            )
        directory = _DEFAULT_PERSIST_DIR if persist is True else persist
        return DiskBackend(pathlib.Path(directory))
    return backend if backend is not None else _default_backend


def _purge_stale_fingerprints(
    store: CacheBackend, namespace: str, fingerprint: str
) -> int:
    """Delete this namespace's entries written under a different fingerprint.

    Redefining a function (notebook cell re-run, hot reload) changes its
    fingerprint, so the old entries can never be read again — but on a shared
    long-lived backend they would keep consuming memory outside any budget's
    view. Code-change invalidation therefore reclaims the storage too.
    Returns how many entries were invalidated.
    """
    current_prefix = f"{namespace}:{fingerprint}:"
    iter_metadata = getattr(store, "iter_metadata", None)
    rows: Iterable[EntryMetadata] = (
        iter_metadata()
        if iter_metadata is not None
        else (
            EntryMetadata(key, entry.namespace, entry.size)
            for key, entry in store.iter_entries()
        )
    )
    purged = 0
    for row in rows:
        if row.namespace == namespace and not row.key.startswith(current_prefix):
            try:
                store.delete(row.key)
                purged += 1
            except Exception:  # noqa: BLE001 - best-effort cleanup at decoration
                pass
    return purged


def _wrap(
    func: Callable[..., Any],
    ttl: int | float | str | None,
    max_memory: int | str | None,
    persist: Persist | None,
    namespace: str | None,
    backend: CacheBackend | None,
    clock: Clock,
    size_of: SizeOf,
) -> Callable[..., Any]:
    # Fail fast: bad configuration breaks at decoration time, not on first call.
    ttl_seconds = parse_ttl(ttl)
    max_memory_bytes = parse_size(max_memory)
    resolved_namespace = namespace if namespace is not None else function_namespace(func)
    fingerprint = function_fingerprint(func)
    store: CacheBackend = _resolve_backend(persist, backend)
    budget = LRUBudget(max_memory_bytes) if max_memory_bytes is not None else None
    flights = KeyedLocks()
    purged = _purge_stale_fingerprints(store, resolved_namespace, fingerprint)
    last_observed = float("-inf")
    clock_guard = threading.Lock()

    def now() -> float:
        # TTL uses wall-clock time because expires_at must survive process
        # restarts once persistence lands. Wall clocks can step backward (NTP,
        # VM resume); clamping to the last observed reading keeps time monotone
        # for this function so an entry can never appear to grow younger. The
        # guard prevents concurrent flights from racing the read-modify-write
        # and sliding the ratchet backward.
        nonlocal last_observed
        with clock_guard:
            last_observed = max(last_observed, clock())
            return last_observed

    def peek_now() -> float:
        # Read-only view of the same clamped clock, for pure observers
        # (explain, and profile later): sees what the wrapper would see
        # without advancing the ratchet, so a transient clock spike observed
        # only through an observation call can never poison future writes.
        return max(last_observed, clock())

    def build_key(*args: Any, **kwargs: Any) -> str:
        return (
            f"{resolved_namespace}:{fingerprint}:"
            f"{digest_arguments(func, args, kwargs)}"
        )

    def safe_delete(key: str) -> None:
        # A backend delete can fail on disk (locked file, permissions). The
        # cache is an optimization: never let that block returning a value.
        # A lingering entry is still-correct data (a budget overrun at worst),
        # never a false HIT.
        try:
            store.delete(key)
        except Exception:
            control.delete_errors += 1

    _NOT_SERVED = object()

    def serve_if_fresh(key: str, *, coalesced: bool) -> Any:
        """Serve a fresh entry with hit bookkeeping, or return _NOT_SERVED.

        Never classifies misses and never mutates entries — that belongs to
        the locked flight, so the fast path and the post-lock re-check can
        share this without double counting.
        """
        if key in control._pending_invalidations:
            return _NOT_SERVED
        entry = store.get(key)
        if entry is None or entry.is_expired(now()):
            return _NOT_SERVED
        if key in control._pending_invalidations:
            # Re-check after the read: a concurrent invalidate whose physical
            # delete failed may have quarantined the key while we were reading
            # the backend. Never serve a condemned entry.
            return _NOT_SERVED
        if budget is not None:
            budget.touch(key)
        control.hits += 1
        if coalesced:
            control.coalesced_hits += 1
        if control._saved_basis_count:
            control.estimated_saved_seconds += (
                control._saved_basis_seconds / control._saved_basis_count
            )
        return entry.value

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = build_key(*args, **kwargs)
        served = serve_if_fresh(key, coalesced=False)
        if served is not _NOT_SERVED:
            return served
        with flights.holding(key):
            return _compute_flight(key, args, kwargs)

    def _compute_flight(key: str, args: tuple, kwargs: dict) -> Any:
        # Re-check under the per-key lock: another flight may have committed
        # while this caller was waiting — that is the single-flight reuse.
        served = serve_if_fresh(key, coalesced=True)
        if served is not _NOT_SERVED:
            return served
        if key in control._pending_invalidations:
            # Authoritative: the caller invalidated this key but the physical
            # delete failed. Never serve the backend's stale entry; retry the
            # removal and recompute. The marker survives until either the
            # delete or the overwriting set succeeds.
            try:
                store.delete(key)
                control._pending_invalidations.discard(key)
            except Exception:
                control.delete_errors += 1
            if budget is not None:
                budget.forget(key)
            control.miss_invalidated += 1
        else:
            entry = store.get(key)
            if entry is not None:
                # Fresh entries were served above; only expired ones reach here.
                safe_delete(key)
                if budget is not None:
                    budget.forget(key)
                control.miss_expired += 1
            else:
                # Atomic pop: a concurrent clear() or marker-cap eviction must
                # not turn check-then-delete into a KeyError.
                with control._mutation_guard:
                    marker = control._invalidation_markers.pop(key, _MARKER_MISSING)
                if marker is not _MARKER_MISSING:
                    control.miss_invalidated += 1
                else:
                    control.miss_not_found += 1
        compute_started = _perf_counter()
        value = func(*args, **kwargs)
        compute_elapsed = _perf_counter() - compute_started
        control.total_compute_seconds += compute_elapsed
        control.compute_count += 1
        committed_at = now()  # TTL starts at commit, not at call start
        size: int | None = None
        if budget is not None:
            # The cache is an optimization, never a correctness dependency: a
            # failing or nonsensical size estimate must not crash a call that
            # already computed its result — skip caching instead.
            try:
                size = int(size_of(value))
            except Exception:
                control.size_estimate_failures += 1
                return value
            if size < 0:
                control.size_estimate_failures += 1
                return value
            if not budget.fits(size):
                # Oversized: compute, return, never cache, never flush the
                # cache to make room for a pathological entry.
                control.skipped_oversized += 1
                return value
            for evicted_key in budget.admit(key, size):
                safe_delete(evicted_key)
                control.evictions += 1
        try:
            store.set(
                key,
                CacheEntry(
                    value=value,
                    namespace=resolved_namespace,
                    created_at=committed_at,
                    expires_at=(
                        committed_at + ttl_seconds if ttl_seconds is not None else None
                    ),
                    size=size,
                ),
            )
            control.writes += 1
            control._pending_invalidations.discard(key)  # fresh value overwrote it
            control._saved_basis_seconds += compute_elapsed
            control._saved_basis_count += 1
        except Exception:
            # The cache is an optimization: a failed write (serialization,
            # disk) never loses the computed result. Release the budget slot
            # so a phantom entry cannot shrink future capacity.
            control.write_errors += 1
            if budget is not None:
                budget.forget(key)
        return value

    control = CacheControl(
        namespace=resolved_namespace,
        fingerprint=fingerprint,
        backend=store,
        ttl_seconds=ttl_seconds,
        max_memory_bytes=max_memory_bytes,
        budget=budget,
        key_builder=build_key,
        now=peek_now,
        flights=flights,
        code_change_invalidations=purged,
    )
    wrapper.cache = control  # type: ignore[attr-defined]
    return wrapper
