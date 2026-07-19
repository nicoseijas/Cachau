"""explain(): pure observation of why a call would hit or miss (GUIDELINES.md §8)."""

import dataclasses

import pytest

from cachau import cache
from cachau.errors import UnhashableArgumentError
from cachau.memory import MemoryBackend


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_hit_explanation_carries_entry_facts():
    clock = FakeClock(now=100.0)

    @cache(ttl=60, max_memory=1000, size_of=lambda v: 123, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    clock.now = 110.0
    explanation = expensive.cache.explain(1)
    assert explanation.outcome == "HIT"
    assert explanation.reason == "found"
    assert explanation.created_at == 100.0
    assert explanation.expires_at == 160.0
    assert explanation.ttl_remaining_seconds == 50.0
    assert explanation.age_seconds == 10.0
    assert explanation.size_bytes == 123


def test_miss_not_found():
    @cache
    def expensive(x):
        return x

    explanation = expensive.cache.explain(42)
    assert explanation.outcome == "MISS"
    assert explanation.reason == "not_found"
    assert explanation.created_at is None


def test_miss_expired_reports_when_it_died():
    clock = FakeClock(now=0.0)

    @cache(ttl=60, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    clock.now = 90.0
    explanation = expensive.cache.explain(1)
    assert explanation.outcome == "MISS"
    assert explanation.reason == "expired"
    assert explanation.expires_at == 60.0
    assert explanation.expired_seconds_ago == 30.0


def test_miss_invalidated():
    @cache
    def expensive(x):
        return x

    expensive(1)
    expensive.cache.invalidate(1)
    explanation = expensive.cache.explain(1)
    assert explanation.outcome == "MISS"
    assert explanation.reason == "invalidated"


def test_pending_invalidation_is_never_explained_as_hit(monkeypatch):
    backend = MemoryBackend()

    @cache(backend=backend)
    def expensive(x):
        return x

    expensive(1)
    monkeypatch.setattr(
        backend, "delete", lambda key: (_ for _ in ()).throw(OSError("locked"))
    )
    expensive.cache.invalidate(1)
    explanation = expensive.cache.explain(1)
    assert explanation.outcome == "MISS"
    assert explanation.reason == "invalidated"


def test_explain_is_pure_it_does_not_touch_stats():
    @cache
    def expensive(x):
        return x

    expensive(1)
    before = expensive.cache.stats()
    expensive.cache.explain(1)
    expensive.cache.explain(99)
    assert expensive.cache.stats() == before


def test_explain_does_not_consume_invalidation_markers():
    @cache
    def expensive(x):
        return x

    expensive(1)
    expensive.cache.invalidate(1)
    assert expensive.cache.explain(1).reason == "invalidated"
    assert expensive.cache.explain(1).reason == "invalidated"  # still there
    expensive(1)  # the real call still gets the right reason
    assert expensive.cache.stats().miss_invalidated == 1


def test_explain_does_not_remove_expired_entries():
    clock = FakeClock()

    @cache(ttl=60, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    clock.now = 61.0
    assert expensive.cache.explain(1).reason == "expired"
    assert expensive.cache.explain(1).reason == "expired"  # entry still on record
    expensive(1)
    assert expensive.cache.stats().miss_expired == 1


def test_explain_does_not_refresh_lru_recency():
    calls = []

    @cache(max_memory=250, size_of=lambda v: 100)
    def expensive(x):
        calls.append(x)
        return x

    expensive("a")
    expensive("b")
    for _ in range(5):
        expensive.cache.explain("a")  # observation must not make "a" recent
    expensive("c")  # evicts "a" (still least recently USED)
    expensive("b")
    expensive("a")
    assert calls == ["a", "b", "c", "a"]


def test_explain_normalizes_arguments_like_calls():
    @cache
    def expensive(x, y=10):
        return x + y

    expensive(1, 2)
    assert expensive.cache.explain(x=1, y=2).outcome == "HIT"
    assert expensive.cache.explain(1, y=2).outcome == "HIT"


def test_explain_fails_loudly_on_unhashable_arguments():
    class Opaque:
        pass

    @cache
    def expensive(data):
        return data

    with pytest.raises(UnhashableArgumentError):
        expensive.cache.explain(Opaque())


def test_no_ttl_means_no_expiry_fields():
    @cache
    def expensive(x):
        return x

    expensive(1)
    explanation = expensive.cache.explain(1)
    assert explanation.outcome == "HIT"
    assert explanation.expires_at is None
    assert explanation.ttl_remaining_seconds is None


def test_explanation_is_immutable():
    @cache
    def expensive(x):
        return x

    explanation = expensive.cache.explain(1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        explanation.outcome = "HIT"


def test_explain_never_advances_the_clock_ratchet():
    """A clock spike observed only through explain() must not poison the
    created_at/expires_at of later real writes."""
    clock = FakeClock(now=100.0)
    from cachau.memory import MemoryBackend

    backend = MemoryBackend()

    @cache(ttl=60, clock=clock, backend=backend)
    def expensive(x):
        return x

    expensive(1)  # committed at t=100
    clock.now = 100_000.0  # transient spike, seen only by explain
    expensive.cache.explain(2)
    clock.now = 105.0  # clock corrects itself
    expensive(2)  # real write must not inherit the spike
    entries = {key: entry for key, entry in backend.iter_entries()}
    created = sorted(entry.created_at for entry in entries.values())
    assert created == [100.0, 105.0]


def test_explain_does_not_delete_corrupt_disk_entries(tmp_path):
    """Observation is non-destructive even for corrupt files; the next real
    call performs the cleanup, not explain()."""

    @cache(persist=str(tmp_path))
    def expensive(x):
        return x * 2

    expensive(1)
    (path,) = list(tmp_path.glob("*.cachau"))
    path.write_bytes(b"garbage")

    assert expensive.cache.explain(1).reason == "not_found"
    assert path.exists()  # explain left the corrupt file alone
    assert expensive(1) == 2  # the real call cleans up and recomputes
    assert expensive.cache.explain(1).outcome == "HIT"


def test_formatting_boundaries():
    from cachau.explanation import _format_bytes, _format_duration, _format_timestamp

    assert _format_bytes(0) == "0 B"
    assert _format_bytes(1023) == "1023 B"
    assert _format_bytes(1024) == "1.0 KB"
    assert _format_duration(59) == "59s"
    assert _format_duration(60) == "1m 0s"
    assert _format_duration(3600) == "1h 0m"
    assert _format_duration(86400) == "1d 0h"
    assert _format_timestamp(None) == "unknown"
    assert _format_timestamp(0.0).endswith("UTC")
    assert _format_timestamp(-1e18) == "unknown"


def test_str_rendering_hit():
    clock = FakeClock(now=100.0)

    @cache(ttl="1h", max_memory=10_000, size_of=lambda v: 2048, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    clock.now = 160.0
    text = str(expensive.cache.explain(1))
    assert "HIT" in text
    assert "Reason" in text
    assert "found" in text
    assert "2.0 KB" in text
    assert "59m" in text  # remaining TTL rendered human-readably


def test_str_rendering_miss_expired():
    clock = FakeClock()

    @cache(ttl=60, clock=clock)
    def expensive(x):
        return x

    expensive(1)
    clock.now = 90.0
    text = str(expensive.cache.explain(1))
    assert "MISS" in text
    assert "expired" in text
    assert "30s ago" in text
