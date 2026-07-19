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
