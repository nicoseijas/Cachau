"""Memory bounds and LRU eviction (GUIDELINES.md §5)."""

import pytest

from cachau import cache
from cachau.errors import InvalidSizeError
from cachau.memory import MemoryBackend


def test_lru_evicts_least_recently_used():
    calls = []

    @cache(max_memory=250, size_of=lambda v: 100)
    def expensive(x):
        calls.append(x)
        return x

    expensive("a")
    expensive("b")
    expensive("a")  # hit: refreshes a's recency
    expensive("c")  # budget full: evicts b (least recently used), not a
    assert calls == ["a", "b", "c"]
    expensive("a")  # still cached
    expensive("b")  # was evicted: recompute
    assert calls == ["a", "b", "c", "b"]


def test_eviction_frees_until_new_entry_fits():
    calls = []

    @cache(max_memory=300, size_of=lambda v: v)
    def expensive(size):
        calls.append(size)
        return size

    expensive(100)
    expensive(120)
    expensive(250)  # needs both prior entries evicted (220 + 250 > 300)
    assert calls == [100, 120, 250]
    assert expensive.cache.evictions == 2
    expensive(100)  # first entry was evicted; refitting evicts 250 in turn
    assert calls == [100, 120, 250, 100]
    assert expensive.cache.evictions == 3


def test_oversized_entry_computed_returned_not_cached():
    calls = []

    @cache(max_memory=100, size_of=lambda v: v)
    def expensive(size):
        calls.append(size)
        return size

    expensive(50)  # cached
    assert expensive(500) == 500  # oversized: computed and returned
    assert expensive(500) == 500  # never cached: recomputed
    assert calls == [50, 500, 500]
    expensive(50)  # the oversized entry did NOT flush the cache
    assert calls == [50, 500, 500]
    assert expensive.cache.skipped_oversized == 2


def test_evictions_are_counted():
    @cache(max_memory=100, size_of=lambda v: 60)
    def expensive(x):
        return x

    expensive(1)
    expensive(2)  # evicts 1
    assert expensive.cache.evictions == 1


def test_bounded_function_never_evicts_other_functions_entries():
    backend = MemoryBackend()
    other_calls = []

    @cache(backend=backend)
    def unbounded(x):
        other_calls.append(x)
        return x

    @cache(max_memory=100, size_of=lambda v: 80, backend=backend)
    def bounded(x):
        return x

    unbounded(1)
    bounded(1)
    bounded(2)  # evicts bounded(1), must not touch unbounded's entry
    unbounded(1)
    assert other_calls == [1]


def test_clear_resets_the_budget():
    calls = []

    @cache(max_memory=200, size_of=lambda v: 100)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    expensive(2)
    expensive.cache.clear()
    expensive(3)
    expensive(4)  # budget was reset: both fit without eviction
    expensive(3)
    expensive(4)
    assert calls == [1, 2, 3, 4]


def test_expired_entries_release_budget():
    class FakeClock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = FakeClock()
    calls = []

    @cache(max_memory=100, ttl=60, size_of=lambda v: 80, clock=clock)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    clock.now = 61.0  # entry 1 expired
    expensive(2)  # replaces it without needing an "eviction" of live data
    expensive(2)
    assert calls == [1, 2]


def test_entry_size_metadata_is_stored():
    backend = MemoryBackend()

    @cache(max_memory=1000, size_of=lambda v: 123, backend=backend)
    def expensive(x):
        return x

    expensive(1)
    ((_, entry),) = backend.iter_entries()
    assert entry.size == 123


def test_string_max_memory_end_to_end():
    @cache(max_memory="1KB", size_of=lambda v: 600)
    def expensive(x):
        return x

    expensive(1)
    expensive(2)  # 1200 > 1024: evicts entry 1
    assert expensive.cache.evictions == 1


def test_failing_size_estimator_never_crashes_a_successful_call():
    calls = []

    def broken_estimator(value):
        raise RuntimeError("estimator bug")

    @cache(max_memory=1000, size_of=broken_estimator)
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(1) == 2  # computed and returned despite estimator failure
    assert expensive(1) == 2  # never cached: recomputed
    assert calls == [1, 1]
    assert expensive.cache.size_estimate_failures == 2


def test_negative_size_estimate_is_rejected_not_admitted():
    @cache(max_memory=100, size_of=lambda v: -50)
    def expensive(x):
        return x

    assert expensive(1) == 1
    assert expensive.cache.size_estimate_failures == 1
    assert expensive.cache.evictions == 0


def test_redefining_a_function_reclaims_stale_entries():
    """Notebook workflow: a re-decorated function must not leak old-fingerprint
    entries into a shared backend outside every budget's view."""
    backend = MemoryBackend()

    def define(factor):
        namespace = "notebook.cell.expensive"
        if factor == 2:

            @cache(namespace=namespace, backend=backend)
            def expensive(x):
                return x * 2

        else:

            @cache(namespace=namespace, backend=backend)
            def expensive(x):
                return x * 3

        return expensive

    v1 = define(2)
    v1(1)
    v1(2)
    assert len(list(backend.iter_entries())) == 2
    v2 = define(3)  # redefinition purges the stale-fingerprint entries
    assert len(list(backend.iter_entries())) == 0
    assert v2(1) == 3


def test_invalid_max_memory_fails_at_decoration_time():
    with pytest.raises(InvalidSizeError):

        @cache(max_memory="10XB")
        def expensive(x):
            return x


def test_default_size_estimator_smoke():
    @cache(max_memory="10MB")
    def expensive(x):
        return b"x" * 1000

    assert expensive(1) == b"x" * 1000
    assert expensive(1) == b"x" * 1000


# --- Budget rehydration across restarts (issue #11) -------------------------
#
# A budget that starts empty over a non-empty store silently stops enforcing
# the bound: the promise of `persist=` + `max_memory=` is that the limit holds
# ACROSS restarts, not just within one process. These tests observe with
# explain(), never with a captured log — a mutated closure would change the
# function's fingerprint and purge the very entries under test.


def _define_sized(persist_dir, max_memory, namespace="rehydrate.expensive"):
    """Re-decorate the same function over an existing cache directory."""

    @cache(
        persist=str(persist_dir),
        max_memory=max_memory,
        namespace=namespace,
        size_of=lambda v: 100,
    )
    def expensive(x):
        return x

    return expensive


def _outcome(cached, x):
    return cached.cache.explain(x).outcome


def test_budget_is_rebuilt_from_persisted_entries(tmp_path):
    first = _define_sized(tmp_path, 250)
    first("a")
    first("b")
    assert first.cache.stats().current_bytes == 200

    second = _define_sized(tmp_path, 250)  # simulated restart
    second("c")  # 300 > 250: must evict, not overflow
    stats = second.cache.stats()
    assert stats.current_bytes <= 250
    assert stats.entries == 2
    assert second.cache.evictions >= 1


def test_rehydrated_budget_evicts_in_creation_order(tmp_path):
    first = _define_sized(tmp_path, 250)
    first("a")
    first("b")

    second = _define_sized(tmp_path, 250)
    second("c")  # evicts the oldest surviving entry, "a"
    assert _outcome(second, "a") == "MISS"
    assert _outcome(second, "b") == "HIT"
    assert _outcome(second, "c") == "HIT"


def test_rehydration_evicts_down_when_max_memory_shrinks(tmp_path):
    first = _define_sized(tmp_path, 350)
    first("a")
    first("b")
    first("c")
    assert first.cache.stats().current_bytes == 300

    shrunk = _define_sized(tmp_path, 150)  # bound lowered between runs
    assert shrunk.cache.stats().current_bytes <= 150


def test_rehydration_ignores_other_namespaces(tmp_path):
    other = _define_sized(tmp_path, 250, namespace="other.expensive")
    other("x")
    other("y")

    mine = _define_sized(tmp_path, 250)
    mine("a")
    mine("b")  # my own budget is 250: two entries of mine must both fit
    assert mine.cache.evictions == 0
    assert mine.cache.stats().current_bytes == 200


def test_rehydration_tolerates_entries_without_a_recorded_size(tmp_path):
    """Entries written before max_memory existed carry size=None."""
    unbounded = _define_sized(tmp_path, None)
    unbounded("a")

    bounded = _define_sized(tmp_path, 250)
    assert bounded("b") == "b"  # decoration and use must not crash
    assert bounded.cache.stats().current_bytes <= 250


# --------------------------------------------------------------------------- #
# The bound must hold under cross-key concurrency (#52)
# --------------------------------------------------------------------------- #


def test_concurrent_distinct_keys_cannot_orphan_past_the_bound(tmp_path):
    """admit -> delete -> set let another thread evict-and-delete a key whose
    file did not exist yet; the later set landed it untracked — an orphan no
    eviction could ever target. Disk grew without bound (960 files against a
    ~5-entry budget in the repro)."""
    import pathlib
    import threading

    from cachau import cache

    @cache(persist=str(tmp_path), max_memory=1200)
    def build(k):
        return "x" * 100 + str(k)

    def worker(offset):
        for i in range(40):
            build(offset + i)

    threads = [threading.Thread(target=worker, args=(t * 40,)) for t in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    files = list(pathlib.Path(tmp_path).glob("*.cachau"))
    stats = build.cache.stats()
    # Every file on disk is tracked by the budget (the safe direction is a
    # tracked-but-deleted phantom, which self-heals on the next commit).
    assert len(files) <= stats.entries
    assert stats.current_bytes <= 1200


def test_failed_write_does_not_evict_healthy_entries():
    """#52 collateral: making room for an entry that never lands must not
    cost entries that were serving hits."""
    from cachau import cache
    from cachau.memory import MemoryBackend

    class FailingSet(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.fail = False

        def set(self, key, entry):
            if self.fail:
                raise OSError("disk full")
            super().set(key, entry)

    backend = FailingSet()

    @cache(backend=backend, max_memory=500)
    def build(k):
        return "x" * 100 + str(k)

    for i in range(3):
        build(i)
    healthy = build.cache.stats().entries
    assert healthy > 0
    assert build.cache.stats().current_bytes > 300  # near-full: admit must evict
    backend.fail = True
    assert build(99) == "x" * 100 + "99"  # the value is never lost
    backend.fail = False
    stats = build.cache.stats()
    assert stats.evictions == 0
    assert stats.entries == healthy  # nobody was sacrificed for a failed write
