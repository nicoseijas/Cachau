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
from cachau.dependencies import (
    LabeledDependency,
    changed_labels,
    fingerprint_dependencies,
    normalize_dependencies,
)
from cachau.disk import DiskBackend
from cachau.durations import parse_ttl
from cachau.errors import ConfigurationError
from cachau.explanation import Explanation
from cachau.fingerprint import (
    function_fingerprint,
    function_namespace,
    is_jit_dispatcher,
)
from cachau.inspection import CacheEntryView, Inspection
from cachau.flight import KeyedLocks
import inspect

from cachau.keys import digest_arguments, digest_custom_key, normalize_call
from cachau.memory import MemoryBackend
from cachau.policy import LRUBudget
from cachau.profile import CacheProfile, diagnose, largest_data_arg, measure
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
# Eviction markers only enrich explain() ("evicted" vs "not_found"); dropping one
# simply loses that annotation for the oldest evicted key, never any correctness.
_EVICTION_MARKER_CAP = 4096
_MARKER_MISSING = object()
Persist = bool | str | os.PathLike
# Every tally CacheControl keeps. Naming them once gives stats() an explicit
# snapshot set and gives _bump() a checkable vocabulary (see the test that
# asserts no _bump call names anything outside this tuple).
TALLY_NAMES = (
    "hits",
    "coalesced_hits",
    "miss_not_found",
    "miss_expired",
    "miss_invalidated",
    "miss_dependency_changed",
    "writes",
    "evictions",
    "skipped_oversized",
    "dependency_race_skips",
    "size_estimate_failures",
    "write_errors",
    "delete_errors",
    "invalidations",
    "code_change_invalidations",
    "total_compute_seconds",
    "compute_count",
    "estimated_saved_seconds",
    "cold_compute_seconds",
)


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
        func: Callable[..., Any],
        size_of: SizeOf,
        dependencies: tuple[LabeledDependency, ...] = (),
        code_change_invalidations: int = 0,
        evictions: int = 0,
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
        self.miss_dependency_changed = 0
        self.writes = 0
        # Seeded: rehydrating the budget over a persistent store can already
        # have evicted, before the first call of this process.
        self.evictions = evictions
        self.skipped_oversized = 0
        self.dependency_race_skips = 0
        self.size_estimate_failures = 0
        self.write_errors = 0
        self.delete_errors = 0
        self.invalidations = 0
        self.code_change_invalidations = code_change_invalidations
        self.total_compute_seconds = 0.0
        self.compute_count = 0
        self.estimated_saved_seconds = 0.0
        self.cold_compute_seconds = 0.0
        # Reason markers: the delete succeeded; remembering the key only
        # attributes the next miss to miss_invalidated. Bounded LRU-style.
        self._invalidation_markers: OrderedDict[str, None] = OrderedDict()
        # Pending invalidations: the physical delete FAILED. Authoritative —
        # the wrapper must never serve these keys from the backend, or an
        # explicitly invalidated value would come back as a false HIT.
        self._pending_invalidations: set[str] = set()
        # Eviction markers: keys the LRU budget dropped, so explain() can say
        # "evicted" instead of "not_found". Pure annotation, bounded, cleared
        # when a key is cached again.
        self._eviction_markers: OrderedDict[str, None] = OrderedDict()
        # Savings average uses only computations whose result was actually
        # cached; skipped/oversized computes can never produce a future hit.
        self._saved_basis_seconds = 0.0
        self._saved_basis_count = 0
        self._backend = backend
        self._budget = budget
        self._key_builder = key_builder
        self._now = now
        self._flights = flights
        self._func = func
        self._size_of = size_of
        self._dependencies = dependencies
        # Guards compound mutations of the invalidation bookkeeping, which is
        # function-wide state reachable from concurrent flights of different
        # keys. Never held across backend I/O or a computation.
        self._mutation_guard = threading.Lock()
        # Guards the tallies. Single-flight only serializes callers of the
        # SAME key, so different-key flights land here concurrently, and
        # `+=` on an attribute is a read-modify-write the language does not
        # promise to be atomic — it only happens to be one under the GIL.
        # A lost increment never corrupts a cached value, but observability
        # is a documented feature, and free-threaded builds drop the
        # accidental protection entirely. Never held across I/O.
        self._counter_guard = threading.Lock()

    def _bump(self, name: str, amount: float = 1) -> None:
        """Add to one tally atomically."""
        with self._counter_guard:
            setattr(self, name, getattr(self, name) + amount)

    def _record_hit(self, *, coalesced: bool) -> None:
        """Tally a hit and the time it saved, as one indivisible update."""
        with self._counter_guard:
            self.hits += 1
            if coalesced:
                self.coalesced_hits += 1
            if self._saved_basis_count:
                # Read basis and count together: separately, a concurrent
                # commit could land between them and skew the average.
                self.estimated_saved_seconds += (
                    self._saved_basis_seconds / self._saved_basis_count
                )

    def _record_commit(self, compute_elapsed: float | None) -> None:
        """Tally a successful write; ``None`` excludes it from the savings basis."""
        with self._counter_guard:
            self.writes += 1
            if compute_elapsed is not None:
                self._saved_basis_seconds += compute_elapsed
                self._saved_basis_count += 1

    def _record_compute(self, compute_elapsed: float) -> None:
        with self._counter_guard:
            self.total_compute_seconds += compute_elapsed
            self.compute_count += 1

    def _note_eviction(self, key: str) -> None:
        """Remember that the budget evicted ``key`` (bounded, LRU-style)."""
        with self._mutation_guard:
            self._eviction_markers[key] = None
            self._eviction_markers.move_to_end(key)
            while len(self._eviction_markers) > _EVICTION_MARKER_CAP:
                self._eviction_markers.popitem(last=False)

    def _note_recached(self, key: str) -> None:
        """A fresh value now occupies ``key``; it is no longer 'evicted'."""
        with self._mutation_guard:
            self._eviction_markers.pop(key, None)

    def explain(self, *args: Any, **kwargs: Any) -> Explanation:
        """Explain what a call with these arguments would do, and why.

        Pure observation: never executes the cached function, never mutates
        cache state, counters, invalidation bookkeeping, or LRU recency. The one
        caveat the user owns: if ``depends_on`` includes a ``token(callable)``,
        checking whether a dependency changed evaluates that callable (as a real
        call would), so a side-effecting or raising token is visible here too.
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
            if key in self._invalidation_markers:
                reason = "invalidated"
            elif key in self._eviction_markers:
                reason = "evicted"
            else:
                reason = "not_found"
            # A failed write leaves nothing to find, so a plain "not_found"
            # would report a silently broken store as an innocent cold cache.
            failed_writes = self.write_errors if reason == "not_found" else 0
            return Explanation(
                outcome="MISS",
                reason=reason,
                write_errors=failed_writes or None,
                **common,
            )
        facts = {
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "size_bytes": entry.size,
        }
        if entry.is_expired(checked_at):
            return Explanation(outcome="MISS", reason="expired", **common, **facts)
        if self._dependencies:
            current = fingerprint_dependencies(self._dependencies)
            if entry.dependency_fingerprints != current:
                changed = changed_labels(entry.dependency_fingerprints, current)
                stored = entry.dependency_fingerprints or {}
                now_fp = current or {}
                diff = {
                    label: (stored.get(label), now_fp.get(label)) for label in changed
                }
                return Explanation(
                    outcome="MISS",
                    reason="dependency_changed",
                    changed_dependencies=changed,
                    dependency_diff=diff,
                    **common,
                    **facts,
                )
        return Explanation(outcome="HIT", reason="found", **common, **facts)

    def profile(self, *args: Any, repeats: int = 3, **kwargs: Any) -> CacheProfile:
        """Measure whether caching this call beats recomputing it.

        Answers the cache-economics question (GUIDELINES.md §15): is
        ``T_key + T_lookup + T_deserialize < T_recompute``? Reports the warm
        recompute time, the cache-hit cost broken into key generation and
        backend read, a verdict, the dominant cost, and a recommendation.

        Unlike ``explain()`` this is NOT pure: it runs the function ``repeats``
        times (plus one warm-up, so JIT compilation is never counted as normal
        execution). It does not touch ``stats()`` counters, and it restores cache
        state — if the entry was absent it is measured against a throwaway copy
        that is then removed, so a call with a TTL or dependencies is never left
        holding a bare entry. Side effects in the function itself do run, once
        per execution.
        """
        if repeats < 1:
            raise ConfigurationError("profile(repeats=) must be at least 1")
        key = self._key_builder(*args, **kwargs)
        # Warm up: compile the JIT specialization / fill any internal caches, and
        # capture a representative value for the read measurement below. Timing
        # is discarded — the first run is the cold one.
        value = self._func(*args, **kwargs)
        compute_seconds = measure(lambda: self._func(*args, **kwargs), repeats)
        # Key generation and backend read are cheap; measure them more times so
        # the min estimator has a clean floor to find.
        cheap_repeats = max(repeats, 20)
        key_seconds = measure(lambda: self._key_builder(*args, **kwargs), cheap_repeats)
        existed = self._backend.get(key) is not None
        if not existed:
            self._backend.set(
                key,
                CacheEntry(
                    value=value,
                    namespace=self.namespace,
                    created_at=self._now(),
                    dependency_fingerprints=fingerprint_dependencies(
                        self._dependencies
                    ),
                ),
            )
        try:
            read_seconds = measure(lambda: self._backend.get(key), cheap_repeats)
        finally:
            if not existed:
                try:
                    self._backend.delete(key)
                except Exception:  # noqa: BLE001 - measurement cleanup, best effort
                    pass
        try:
            size_bytes: int | None = int(self._size_of(value))
        except Exception:  # noqa: BLE001 - size is informational, never fatal
            size_bytes = None
        largest = largest_data_arg(normalize_call(self._func, args, kwargs))
        verdict, primary_cost, recommendation = diagnose(
            compute_seconds=compute_seconds,
            key_seconds=key_seconds,
            read_seconds=read_seconds,
            largest_data=largest,
        )
        return CacheProfile(
            namespace=self.namespace,
            key=key,
            repeats=repeats,
            compute_seconds=compute_seconds,
            key_seconds=key_seconds,
            read_seconds=read_seconds,
            size_bytes=size_bytes,
            verdict=verdict,
            primary_cost=primary_cost,
            recommendation=recommendation,
        )

    def inspect(self) -> Inspection:
        """List this function's cached entries, newest first.

        Pure observation, like ``explain()``: reads entry headers only (no
        payload is deserialized), never mutates cache state or counters. Each
        view carries creation time, size, remaining TTL, and any dependency
        fingerprints. Entries quarantined by a failed ``invalidate()`` are
        omitted — they are still physically present but will never be served.
        """
        checked_at = self._now()
        with self._mutation_guard:
            pending = set(self._pending_invalidations)
        views = [
            CacheEntryView(
                key=row.key,
                namespace=row.namespace,
                checked_at=checked_at,
                created_at=row.created_at,
                expires_at=row.expires_at,
                size_bytes=row.size,
                dependency_fingerprints=row.dependency_fingerprints,
            )
            for row in self._iter_metadata()
            if row.namespace == self.namespace and row.key not in pending
        ]
        # Newest first; entries with no recorded creation time sort last.
        views.sort(key=lambda v: (v.created_at is None, -(v.created_at or 0.0)))
        return Inspection(namespace=self.namespace, entries=tuple(views))

    def clear(self) -> None:
        """Forget every stored result for this function."""
        self._backend.clear(namespace=self.namespace)
        if self._budget is not None:
            self._budget.reset()
        with self._mutation_guard:
            self._invalidation_markers.clear()
            self._pending_invalidations.clear()
            self._eviction_markers.clear()

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
                self._bump("delete_errors")
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
            self._bump("invalidations")

    def stats(self) -> CacheStats:
        """An immutable snapshot of this function's cache activity."""
        entries = 0
        current_bytes = 0
        for row in self._iter_metadata():
            if row.namespace == self.namespace:
                entries += 1
                if row.size is not None:
                    current_bytes += row.size
        # Copy the tallies in one locked read: individually they would come
        # from different instants, so a snapshot taken under live traffic
        # could report a hit_rate that no moment ever had. The backend scan
        # above stays OUTSIDE the guard — it is I/O, and holding the tallies
        # lock across it would stall every concurrent call.
        with self._counter_guard:
            tally = {name: getattr(self, name) for name in TALLY_NAMES}
        misses = (
            tally["miss_not_found"]
            + tally["miss_expired"]
            + tally["miss_invalidated"]
            + tally["miss_dependency_changed"]
        )
        total_calls = tally["hits"] + misses
        return CacheStats(
            hits=tally["hits"],
            coalesced_hits=tally["coalesced_hits"],
            misses=misses,
            hit_rate=tally["hits"] / total_calls if total_calls else 0.0,
            miss_not_found=tally["miss_not_found"],
            miss_expired=tally["miss_expired"],
            miss_invalidated=tally["miss_invalidated"],
            miss_dependency_changed=tally["miss_dependency_changed"],
            expirations=tally["miss_expired"],
            writes=tally["writes"],
            skipped_writes=(
                tally["skipped_oversized"]
                + tally["dependency_race_skips"]
                + tally["size_estimate_failures"]
                + tally["write_errors"]
            ),
            skipped_oversized=tally["skipped_oversized"],
            dependency_race_skips=tally["dependency_race_skips"],
            size_estimate_failures=tally["size_estimate_failures"],
            write_errors=tally["write_errors"],
            delete_errors=tally["delete_errors"],
            evictions=tally["evictions"],
            invalidations=tally["invalidations"],
            code_change_invalidations=tally["code_change_invalidations"],
            entries=entries,
            current_bytes=current_bytes,
            total_compute_seconds=tally["total_compute_seconds"],
            estimated_saved_seconds=tally["estimated_saved_seconds"],
            cold_compute_seconds=tally["cold_compute_seconds"],
        )

    def _iter_metadata(self) -> Iterable[EntryMetadata]:
        return _metadata_rows(self._backend)


@overload
def cache(func: F) -> F: ...


@overload
def cache(
    *,
    ttl: int | float | str | None = None,
    max_memory: int | str | None = None,
    persist: Persist | None = None,
    namespace: str | None = None,
    key: Callable[..., Any] | None = None,
    ignore: list[str] | tuple[str, ...] | None = None,
    depends_on: list[Any] | tuple[Any, ...] | None = None,
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
    key: Callable[..., Any] | None = None,
    ignore: list[str] | tuple[str, ...] | None = None,
    depends_on: list[Any] | tuple[Any, ...] | None = None,
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
    restarts; a failed write never loses the computed result. ``depends_on``
    declares external inputs — file paths, or ``cachau.file/env/package/token``
    descriptors — that invalidate a result when they change (a ``dependency_
    changed`` miss). Exceptions are never cached; unhashable arguments fail
    loudly. ``func.cache`` exposes ``stats()``, ``clear()`` and
    ``invalidate(...)``.
    """
    if func is not None:
        return _wrap(
            func, ttl, max_memory, persist, namespace, key, ignore, depends_on,
            backend, clock, size_of,
        )
    return lambda f: _wrap(
        f, ttl, max_memory, persist, namespace, key, ignore, depends_on, backend,
        clock, size_of,
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


def _metadata_rows(store: CacheBackend) -> Iterable[EntryMetadata]:
    """Per-entry metadata, without deserializing payloads where possible.

    ``iter_metadata`` is optional in the backend protocol, so a third-party
    store that only implements ``iter_entries`` still works — it just pays
    for the deserialization.
    """
    iter_metadata = getattr(store, "iter_metadata", None)
    if iter_metadata is not None:
        return iter_metadata()
    return (
        EntryMetadata(key, entry.namespace, entry.size, entry.created_at)
        for key, entry in store.iter_entries()
    )


def _best_effort_delete(store: CacheBackend, key: str) -> None:
    # Decoration-time cleanup: a store that refuses a delete leaves a
    # still-correct entry behind, never a wrong one. Never raise into
    # the user's import.
    try:
        store.delete(key)
    except Exception:  # noqa: BLE001 - best-effort cleanup at decoration
        pass


def _rehydrate_budget(
    store: CacheBackend,
    budget: LRUBudget,
    namespace: str,
    fingerprint: str,
    rows: Iterable[EntryMetadata],
) -> int:
    """Re-seed the budget from entries already in the store; return evictions.

    A budget built empty over a non-empty persistent store believes it has
    full capacity, so ``persist=`` + ``max_memory=`` would stop enforcing the
    bound after every restart — the limit is supposed to hold ACROSS restarts,
    not merely within one process.

    Entries are re-admitted oldest-first: creation order is the only ordering
    a store preserves, since real LRU recency is process-local and dies with
    the process. Eviction therefore resumes from the oldest surviving entry.

    Entries with no recorded size (written before ``max_memory`` was set)
    cannot be budgeted without deserializing their payload, which decoration
    must not pay for; they are left in place, outside the bound, and fall out
    naturally as they are re-read and rewritten.
    """
    current_prefix = f"{namespace}:{fingerprint}:"
    sized: list[tuple[float, str, int]] = [
        (row.created_at if row.created_at is not None else 0.0, row.key, row.size)
        for row in rows
        if row.namespace == namespace
        and row.key.startswith(current_prefix)
        and row.size is not None
    ]
    # The key breaks ties, so entries written within one clock tick — or in a
    # store that records no timestamps — still rehydrate deterministically.
    sized.sort()
    evictions = 0
    for _, key, size in sized:
        if not budget.fits(size):
            # Larger than the whole budget: it could never be admitted again,
            # and leaving it would keep unbounded bytes on disk forever.
            _best_effort_delete(store, key)
            evictions += 1
            continue
        for evicted_key in budget.admit(key, size):
            _best_effort_delete(store, evicted_key)
            evictions += 1
    return evictions


def _purge_stale_fingerprints(
    store: CacheBackend, namespace: str, fingerprint: str, rows: Iterable[EntryMetadata]
) -> int:
    """Delete this namespace's entries written under a different fingerprint.

    Redefining a function (notebook cell re-run, hot reload) changes its
    fingerprint, so the old entries can never be read again — but on a shared
    long-lived backend they would keep consuming memory outside any budget's
    view. Code-change invalidation therefore reclaims the storage too.
    Returns how many entries were invalidated.
    """
    current_prefix = f"{namespace}:{fingerprint}:"
    purged = 0
    for row in rows:
        if row.namespace == namespace and not row.key.startswith(current_prefix):
            try:
                store.delete(row.key)
                purged += 1
            except Exception:  # noqa: BLE001 - best-effort cleanup at decoration
                pass
    return purged


def _validate_ignore(
    func: Callable[..., Any], ignore: list[str] | tuple[str, ...] | None
) -> frozenset[str]:
    if not ignore:
        return frozenset()
    parameters = set(inspect.signature(func).parameters)
    unknown = set(ignore) - parameters
    if unknown:
        raise ConfigurationError(
            f"ignore= names parameters that {func.__qualname__}() does not "
            f"have: {sorted(unknown)}"
        )
    return frozenset(ignore)


def _wrap(
    func: Callable[..., Any],
    ttl: int | float | str | None,
    max_memory: int | str | None,
    persist: Persist | None,
    namespace: str | None,
    key: Callable[..., Any] | None,
    ignore: list[str] | tuple[str, ...] | None,
    depends_on: list[Any] | tuple[Any, ...] | None,
    backend: CacheBackend | None,
    clock: Clock,
    size_of: SizeOf,
) -> Callable[..., Any]:
    # Fail fast: bad configuration breaks at decoration time, not on first call.
    if key is not None and ignore:
        raise ConfigurationError(
            "key= and ignore= are mutually exclusive: an explicit key already "
            "defines the full identity, so there is nothing left to ignore"
        )
    ignored_names = _validate_ignore(func, ignore)
    dependencies = normalize_dependencies(depends_on)
    ttl_seconds = parse_ttl(ttl)
    max_memory_bytes = parse_size(max_memory)
    resolved_namespace = namespace if namespace is not None else function_namespace(func)
    fingerprint = function_fingerprint(func)
    jit_boundary = is_jit_dispatcher(func)
    store: CacheBackend = _resolve_backend(persist, backend)
    budget = LRUBudget(max_memory_bytes) if max_memory_bytes is not None else None
    flights = KeyedLocks()
    # One pass over the store for both decoration-time chores: decoration
    # runs at import, and each pass is another round of file opens across the
    # whole cache directory. Purge and rehydration touch disjoint rows (one
    # takes every OTHER fingerprint, the other only this one), so a single
    # snapshot is safe even though purge deletes as it goes.
    rows = list(_metadata_rows(store))
    purged = _purge_stale_fingerprints(store, resolved_namespace, fingerprint, rows)
    rehydrated_evictions = (
        _rehydrate_budget(store, budget, resolved_namespace, fingerprint, rows)
        if budget is not None
        else 0
    )
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
        if key is not None:
            digest = digest_custom_key(func, key(*args, **kwargs))
        else:
            digest = digest_arguments(func, args, kwargs, ignore=ignored_names)
        return f"{resolved_namespace}:{fingerprint}:{digest}"

    def current_dependencies() -> dict[str, str] | None:
        # One fingerprint pass per call, shared by the freshness check and the
        # write, so a HIT and the value it might commit agree on exactly which
        # dependency state the result reflects. ``None`` when none are declared.
        return fingerprint_dependencies(dependencies)

    def safe_delete(key: str) -> None:
        # A backend delete can fail on disk (locked file, permissions). The
        # cache is an optimization: never let that block returning a value.
        # A lingering entry is still-correct data (a budget overrun at worst),
        # never a false HIT.
        try:
            store.delete(key)
        except Exception:
            control._bump("delete_errors")

    _NOT_SERVED = object()

    def serve_if_fresh(key: str, dep_fingerprints: dict[str, str] | None, *,
                       coalesced: bool) -> Any:
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
        if dependencies and entry.dependency_fingerprints != dep_fingerprints:
            # A declared dependency changed since this entry was committed:
            # the stored result no longer reflects the current inputs. Not a
            # HIT; the locked flight reclassifies and recomputes it.
            return _NOT_SERVED
        if key in control._pending_invalidations:
            # Re-check after the read: a concurrent invalidate whose physical
            # delete failed may have quarantined the key while we were reading
            # the backend. Never serve a condemned entry.
            return _NOT_SERVED
        if budget is not None:
            budget.touch(key)
        control._record_hit(coalesced=coalesced)
        return entry.value

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = build_key(*args, **kwargs)
        dep_fingerprints = current_dependencies()
        served = serve_if_fresh(key, dep_fingerprints, coalesced=False)
        if served is not _NOT_SERVED:
            return served
        with flights.holding(key):
            return _compute_flight(key, args, kwargs, dep_fingerprints)

    def _compute_flight(
        key: str, args: tuple, kwargs: dict, dep_fingerprints: dict[str, str] | None
    ) -> Any:
        # Re-check under the per-key lock: another flight may have committed
        # while this caller was waiting — that is the single-flight reuse.
        served = serve_if_fresh(key, dep_fingerprints, coalesced=True)
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
                control._bump("delete_errors")
            if budget is not None:
                budget.forget(key)
            control._bump("miss_invalidated")
        else:
            entry = store.get(key)
            if entry is not None:
                # Fresh, dependency-matching entries were served above; an entry
                # that still exists here is either expired or has a stale
                # dependency fingerprint. Attribute the miss to the right cause.
                safe_delete(key)
                if budget is not None:
                    budget.forget(key)
                if entry.is_expired(now()):
                    control._bump("miss_expired")
                else:
                    control._bump("miss_dependency_changed")
            else:
                # Atomic pop: a concurrent clear() or marker-cap eviction must
                # not turn check-then-delete into a KeyError.
                with control._mutation_guard:
                    marker = control._invalidation_markers.pop(key, _MARKER_MISSING)
                if marker is not _MARKER_MISSING:
                    control._bump("miss_invalidated")
                else:
                    control._bump("miss_not_found")
        specializations_before = -1
        if jit_boundary:
            known_signatures = getattr(func, "signatures", None)
            if known_signatures is not None:
                specializations_before = len(known_signatures)
        compute_started = _perf_counter()
        value = func(*args, **kwargs)
        compute_elapsed = _perf_counter() - compute_started
        control._record_compute(compute_elapsed)
        # Cold JIT is per SPECIALIZATION, not per function: a new dtype on a
        # later call triggers a fresh compile. Detect it by whether the
        # dispatcher grew a compiled signature during this execution, and
        # keep that one-time cost out of the savings baseline so compile
        # time is never counted as normal execution cost.
        is_cold_jit = False
        if jit_boundary:
            known_signatures = getattr(func, "signatures", None)
            if known_signatures is not None and specializations_before >= 0:
                is_cold_jit = len(known_signatures) > specializations_before
            else:
                is_cold_jit = control.compute_count == 1
            if is_cold_jit:
                control._bump("cold_compute_seconds", compute_elapsed)
        if dependencies:
            # A declared dependency may have moved WHILE func() ran. Re-observe
            # after compute and refuse to cache if it changed: stamping the
            # result with the pre-compute fingerprint would let a later call —
            # back in the original state — serve it as fresh, a false HIT. This
            # mirrors TTL's commit-time discipline (committed_at is "at commit,
            # not at call start"). It closes the common case (a dependency
            # rewritten mid-compute and left changed); it cannot close a change
            # that reverts before commit (A→B→A), which the stability contract
            # in dependencies.py excludes — cachau cannot see reads made inside
            # func(). When stable, the snapshots match and nothing changes.
            dep_after = current_dependencies()
            if dep_after != dep_fingerprints:
                control._bump("dependency_race_skips")
                return value
            dep_fingerprints = dep_after
        committed_at = now()  # TTL starts at commit, not at call start
        size: int | None = None
        if budget is not None:
            # The cache is an optimization, never a correctness dependency: a
            # failing or nonsensical size estimate must not crash a call that
            # already computed its result — skip caching instead.
            try:
                size = int(size_of(value))
            except Exception:
                control._bump("size_estimate_failures")
                return value
            if size < 0:
                control._bump("size_estimate_failures")
                return value
            if not budget.fits(size):
                # Oversized: compute, return, never cache, never flush the
                # cache to make room for a pathological entry.
                control._bump("skipped_oversized")
                return value
            for evicted_key in budget.admit(key, size):
                safe_delete(evicted_key)
                control._bump("evictions")
                control._note_eviction(evicted_key)
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
                    dependency_fingerprints=dep_fingerprints,
                ),
            )
            control._record_commit(None if is_cold_jit else compute_elapsed)
            control._pending_invalidations.discard(key)  # fresh value overwrote it
            control._note_recached(key)  # no longer 'evicted' — it's cached again
        except Exception:
            # The cache is an optimization: a failed write (serialization,
            # disk) never loses the computed result. Release the budget slot
            # so a phantom entry cannot shrink future capacity.
            control._bump("write_errors")
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
        func=func,
        size_of=size_of,
        dependencies=dependencies,
        code_change_invalidations=purged,
        evictions=rehydrated_evictions,
    )
    wrapper.cache = control  # type: ignore[attr-defined]
    return wrapper
