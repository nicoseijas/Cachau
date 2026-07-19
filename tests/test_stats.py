"""Observability: stats(), miss reasons, per-call invalidation (GUIDELINES.md §8)."""

import dataclasses

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
