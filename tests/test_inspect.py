"""inspect(): browse a function's cached entries (GUIDELINES.md §8, §12).

Pure observation, header-only (no payload deserialized), notebook-friendly.
"""

import cachau
import pytest

from cachau import cache
from cachau.inspection import CacheEntryView, Inspection
from cachau.memory import MemoryBackend


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_inspect_lists_cached_entries():
    @cache
    def f(n):
        return n * n

    f(1)
    f(2)
    f(3)
    insp = f.cache.inspect()
    assert isinstance(insp, Inspection)
    assert len(insp) == 3
    assert all(isinstance(v, CacheEntryView) for v in insp)


def test_inspect_empty_is_a_clean_empty_listing():
    @cache
    def f(n):
        return n

    insp = f.cache.inspect()
    assert len(insp) == 0
    assert list(insp) == []
    assert "No cached entries" in str(insp)


def test_inspect_is_newest_first():
    clock = FakeClock(now=100.0)

    @cache(clock=clock)
    def f(n):
        return n

    f(1)  # created at 100
    clock.now = 200.0
    f(2)  # created at 200
    clock.now = 300.0
    f(3)  # created at 300
    insp = f.cache.inspect()
    assert [v.created_at for v in insp] == [300.0, 200.0, 100.0]


def test_inspect_is_namespace_isolated():
    backend = MemoryBackend()

    @cache(backend=backend)
    def a(n):
        return n

    @cache(backend=backend)
    def b(n):
        return n

    a(1)
    a(2)
    b(1)
    assert len(a.cache.inspect()) == 2  # only a's entries, not b's
    assert len(b.cache.inspect()) == 1


def test_inspect_view_exposes_ttl_and_dependencies(monkeypatch):
    clock = FakeClock(now=0.0)
    monkeypatch.setenv("MODE", "fast")

    @cache(ttl=3600, clock=clock, depends_on=[cachau.env("MODE")])
    def f(n):
        return n

    f(1)
    clock.now = 600.0
    (view,) = list(f.cache.inspect())
    assert view.age_seconds == 600.0
    assert view.ttl_remaining_seconds == 3000.0
    assert not view.is_expired
    assert view.dependency_fingerprints == {"env:MODE": "v:fast"}
    assert view.digest  # the argument-identifying part of the key


def test_inspect_reports_expired_entries():
    clock = FakeClock(now=0.0)

    @cache(ttl=60, clock=clock)
    def f(n):
        return n

    f(1)
    clock.now = 120.0
    (view,) = list(f.cache.inspect())
    assert view.is_expired
    assert view.ttl_remaining_seconds is None
    assert "EXPIRED" in str(view)


def test_inspect_total_bytes_with_sizes():
    @cache(max_memory=10_000, size_of=lambda v: 100)
    def f(n):
        return n

    f(1)
    f(2)
    insp = f.cache.inspect()
    assert insp.total_bytes == 200


def test_inspect_is_pure_it_does_not_touch_stats():
    @cache
    def f(n):
        return n

    f(1)
    before = f.cache.stats()
    f.cache.inspect()
    f.cache.inspect()
    assert f.cache.stats() == before


def test_inspect_omits_pending_invalidations(monkeypatch):
    backend = MemoryBackend()

    @cache(backend=backend)
    def f(n):
        return n

    f(1)
    f(2)
    # Make the physical delete fail so the key is quarantined but still present.
    monkeypatch.setattr(
        backend, "delete", lambda key: (_ for _ in ()).throw(OSError("locked"))
    )
    f.cache.invalidate(1)
    # The invalidated-but-undeletable entry must not appear as if it were cached.
    assert len(f.cache.inspect()) == 1


def test_inspect_is_indexable_and_iterable():
    @cache
    def f(n):
        return n

    f(1)
    insp = f.cache.inspect()
    assert insp[0] is next(iter(insp))
    assert len(list(insp)) == 1


def test_inspection_is_immutable():
    import dataclasses

    @cache
    def f(n):
        return n

    f(1)
    insp = f.cache.inspect()
    with pytest.raises(dataclasses.FrozenInstanceError):
        insp.namespace = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        insp[0].key = "x"
