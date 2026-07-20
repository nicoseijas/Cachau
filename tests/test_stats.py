"""Observability: stats(), miss reasons, per-call invalidation (GUIDELINES.md §8)."""

import dataclasses
import sys
import threading

import pytest

from cachau import cache
from cachau.memory import MemoryBackend


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_hits_misses_and_hit_rate():
    @cache
    def expensive(x):
        return x

    expensive(1)  # miss
    expensive(1)  # hit
    expensive(2)  # miss
    expensive(1)  # hit
    stats = expensive.cache.stats()
    assert stats.hits == 2
    assert stats.misses == 2
    assert stats.hit_rate == 0.5


def test_hit_rate_with_no_calls_is_zero():
    @cache
    def expensive(x):
        return x

    assert expensive.cache.stats().hit_rate == 0.0


def test_miss_reasons_distinguish_not_found_from_expired():
    clock = FakeClock()

    @cache(ttl=60, clock=clock)
    def expensive(x):
        return x

    expensive(1)  # miss_not_found
    clock.now = 61.0
    expensive(1)  # miss_expired
    stats = expensive.cache.stats()
    assert stats.miss_not_found == 1
    assert stats.miss_expired == 1
    assert stats.misses == 2
    assert stats.expirations == 1


def test_writes_and_skipped_writes():
    @cache(max_memory=100, size_of=lambda v: v)
    def expensive(size):
        return size

    expensive(50)  # written
    expensive(500)  # oversized: skipped
    stats = expensive.cache.stats()
    assert stats.writes == 1
    assert stats.skipped_oversized == 1
    assert stats.skipped_writes == 1


def test_invalidate_forces_recompute_of_that_call_only():
    calls = []

    @cache
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    expensive(2)
    expensive.cache.invalidate(1)
    expensive(1)  # recomputed
    expensive(2)  # still cached
    assert calls == [1, 2, 1]
    stats = expensive.cache.stats()
    assert stats.invalidations == 1
    assert stats.miss_invalidated == 1


def test_invalidate_normalizes_kwargs_like_calls_do():
    calls = []

    @cache
    def expensive(x, y=10):
        calls.append((x, y))
        return x + y

    expensive(1, 2)
    expensive.cache.invalidate(x=1, y=2)
    expensive(1, 2)
    assert calls == [(1, 2), (1, 2)]


def test_invalidating_an_uncached_call_is_a_noop_but_counted():
    @cache
    def expensive(x):
        return x

    expensive.cache.invalidate(99)
    assert expensive.cache.stats().invalidations == 1


def test_entries_and_current_bytes():
    @cache(max_memory=1000, size_of=lambda v: 100)
    def expensive(x):
        return x

    expensive(1)
    expensive(2)
    stats = expensive.cache.stats()
    assert stats.entries == 2
    assert stats.current_bytes == 200


def test_entries_scoped_to_this_function_on_shared_backend():
    backend = MemoryBackend()

    @cache(backend=backend)
    def first(x):
        return x

    @cache(backend=backend)
    def second(x):
        return x

    first(1)
    first(2)
    second(1)
    assert first.cache.stats().entries == 2
    assert second.cache.stats().entries == 1


def test_code_change_invalidations_are_reported(tmp_path):
    template = """
from cachau import cache

@cache(persist={d!r}, namespace="ver.expensive")
def expensive(x):
    return x * {factor}
"""
    scope_v1, scope_v2 = {}, {}
    exec(template.format(d=str(tmp_path), factor=2), scope_v1)
    scope_v1["expensive"](1)
    scope_v1["expensive"](2)

    exec(template.format(d=str(tmp_path), factor=3), scope_v2)
    assert scope_v2["expensive"].cache.stats().code_change_invalidations == 2


def test_compute_time_and_estimated_savings(monkeypatch):
    import cachau.decorator as decorator_module

    fake_time = {"now": 0.0}

    def fake_perf():
        return fake_time["now"]

    monkeypatch.setattr(decorator_module, "_perf_counter", fake_perf)

    @cache
    def expensive(x):
        fake_time["now"] += 2.0  # each computation "takes" 2 seconds
        return x

    expensive(1)  # compute: 2s
    expensive(1)  # hit: saves ~2s
    expensive(1)  # hit: saves ~2s
    stats = expensive.cache.stats()
    assert stats.total_compute_seconds == 2.0
    assert stats.estimated_saved_seconds == 4.0


def test_stats_snapshot_is_immutable():
    @cache
    def expensive(x):
        return x

    stats = expensive.cache.stats()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.hits = 99


def test_failed_invalidate_never_serves_the_stale_value(monkeypatch):
    """CRITICAL regression: invalidate + failed physical delete must not let
    the stale entry come back as a HIT."""
    backend = MemoryBackend()
    calls = []

    @cache(backend=backend)
    def expensive(x):
        calls.append(x)
        return x * 2

    expensive(1)

    real_delete = backend.delete

    def failing_delete(key):
        raise OSError("file locked")

    monkeypatch.setattr(backend, "delete", failing_delete)
    expensive.cache.invalidate(1)
    assert expensive.cache.stats().delete_errors == 1

    assert expensive(1) == 2  # recomputed, never served from the stale entry
    assert calls == [1, 1]
    assert expensive.cache.stats().miss_invalidated == 1

    monkeypatch.setattr(backend, "delete", real_delete)
    assert expensive(1) == 2  # fresh entry now serves normally
    assert calls == [1, 1]


def test_pending_invalidation_survives_until_overwrite_succeeds(monkeypatch):
    """If both the delete and the overwriting write fail, the key stays
    quarantined: the stale value is never served on any later call."""
    calls = []

    class StubbornBackend(MemoryBackend):
        fail = False

        def delete(self, key):
            if self.fail:
                raise OSError("locked")
            super().delete(key)

        def set(self, key, entry):
            if self.fail:
                raise OSError("disk full")
            super().set(key, entry)

    backend = StubbornBackend()

    @cache(backend=backend)
    def expensive(x):
        calls.append(x)
        return x * 2

    expensive(1)
    backend.fail = True
    expensive.cache.invalidate(1)
    assert expensive(1) == 2  # recompute; write fails too
    assert expensive(1) == 2  # still quarantined: recompute again
    assert calls == [1, 1, 1]
    backend.fail = False
    assert expensive(1) == 2  # delete retry succeeds; fresh write lands
    assert expensive(1) == 2  # now a real HIT
    assert calls == [1, 1, 1, 1]


def test_clear_resets_invalidation_bookkeeping():
    @cache
    def expensive(x):
        return x

    expensive(1)
    expensive.cache.invalidate(1)
    expensive.cache.clear()
    expensive(1)
    stats = expensive.cache.stats()
    assert stats.miss_invalidated == 0  # the clear, not the invalidation, wins
    assert stats.miss_not_found == 2  # initial miss + post-clear miss


def test_invalidating_an_expired_entry_reports_invalidated():
    """Explicit caller action wins over lazy expiry in reason attribution."""
    clock = FakeClock()

    @cache(ttl=60, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    clock.now = 61.0  # expired but not yet reaped
    expensive.cache.invalidate(1)
    expensive(1)
    stats = expensive.cache.stats()
    assert stats.miss_invalidated == 1
    assert stats.miss_expired == 0


def test_savings_average_ignores_uncacheable_computes(monkeypatch):
    import cachau.decorator as decorator_module

    fake_time = {"now": 0.0}
    monkeypatch.setattr(decorator_module, "_perf_counter", lambda: fake_time["now"])

    durations = {1: 2.0, 2: 100.0}

    @cache(max_memory=100, size_of=lambda v: 1000 if v == 2 else 10)
    def expensive(x):
        fake_time["now"] += durations[x]
        return x

    expensive(1)  # cached, 2s
    expensive(2)  # oversized (never cacheable), 100s — must not pollute average
    expensive(1)  # hit: credits ~2s, not ~51s
    assert expensive.cache.stats().estimated_saved_seconds == 2.0


def test_invalidation_marker_set_is_bounded(monkeypatch):
    import cachau.decorator as decorator_module

    monkeypatch.setattr(decorator_module, "_INVALIDATION_MARKER_CAP", 3)

    @cache
    def expensive(x):
        return x

    for i in range(10):
        expensive.cache.invalidate(i)
    assert len(expensive.cache._invalidation_markers) == 3
    assert expensive.cache.stats().invalidations == 10


def test_counters_survive_expired_and_eviction_flows():
    clock = FakeClock()

    @cache(ttl=60, max_memory=100, size_of=lambda v: 60, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    expensive(2)  # evicts 1
    clock.now = 61.0
    expensive(2)  # expired
    stats = expensive.cache.stats()
    assert stats.evictions == 1
    assert stats.miss_expired == 1
    assert stats.writes == 3


# --- Counter thread-safety (issue #15) --------------------------------------
#
# Single-flight only serializes callers of the SAME key, so flights for
# different keys mutate the tallies concurrently — and `+=` on an attribute
# is not atomic even under the GIL (it is LOAD / ADD / STORE). Lost
# increments never corrupt a cached value, but they do corrupt observability.


def _hammer_counters(threads=8, rounds=400, keys=16):
    @cache
    def expensive(x):
        return x

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force preemption inside the read-modify-write
    barrier = threading.Barrier(threads)

    def worker():
        barrier.wait()
        for i in range(rounds):
            expensive(i % keys)

    try:
        workers = [threading.Thread(target=worker) for _ in range(threads)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
    finally:
        sys.setswitchinterval(switch_interval)
    return expensive.cache.stats(), threads * rounds


def test_counters_do_not_lose_increments_under_concurrency():
    stats, total_calls = _hammer_counters()
    assert stats.hits + stats.misses == total_calls


def test_miss_reasons_stay_consistent_under_concurrency():
    stats, _ = _hammer_counters()
    assert (
        stats.miss_not_found + stats.miss_expired + stats.miss_invalidated
        == stats.misses
    )
    assert stats.writes == stats.misses  # every miss commits exactly once


def test_stats_never_holds_the_counter_lock_across_backend_io():
    """The tallies lock must never be held across I/O, or a slow backend
    stalls every concurrent call instead of just the observer."""
    control_box = {}

    class ProbingBackend(MemoryBackend):
        def iter_metadata(self):
            control = control_box.get("control")
            if control is None:  # decoration-time purge, before cache exists
                return super().iter_metadata()
            acquired = control._counter_guard.acquire(blocking=False)
            if acquired:
                control._counter_guard.release()
            control_box["free_during_io"] = acquired
            return super().iter_metadata()

    @cache(backend=ProbingBackend())
    def expensive(x):
        return x

    control_box["control"] = expensive.cache
    expensive(1)
    expensive.cache.stats()
    assert control_box["free_during_io"] is True


def test_stats_snapshot_is_internally_consistent():
    stats, total_calls = _hammer_counters()
    assert stats.hits + stats.misses == total_calls
    expected_rate = stats.hits / (stats.hits + stats.misses)
    assert stats.hit_rate == expected_rate


def test_every_bump_target_is_a_declared_tally():
    """`_bump` is stringly-typed: a rename would break it silently at runtime,
    invisible to any type checker. This is the check that isn't."""
    import ast
    import inspect

    import cachau.decorator as decorator

    tree = ast.parse(inspect.getsource(decorator))
    targets = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_bump"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert targets, "no _bump call sites found — did the helper get renamed?"
    assert targets <= set(decorator.TALLY_NAMES)

    @cache
    def expensive(x):
        return x

    for name in decorator.TALLY_NAMES:
        assert hasattr(expensive.cache, name)
