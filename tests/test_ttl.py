"""TTL semantics: starts at commit, expires lazily (GUIDELINES.md §4)."""

import pytest

from cachau import cache
from cachau.errors import InvalidTTLError
from cachau.memory import MemoryBackend


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_entry_expires_after_ttl():
    clock = FakeClock()
    calls = []

    @cache(ttl=60, clock=clock)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    clock.advance(61)
    expensive(1)
    assert calls == [1, 1]


def test_entry_survives_within_ttl():
    clock = FakeClock()
    calls = []

    @cache(ttl=60, clock=clock)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    clock.advance(59)
    expensive(1)
    assert calls == [1]


def test_ttl_starts_at_commit_not_at_call_start():
    clock = FakeClock()
    calls = []

    @cache(ttl=60, clock=clock)
    def slow(x):
        calls.append(x)
        clock.advance(50)  # long computation: commit happens at t=50
        return x

    slow(1)  # called at t=0, committed at t=50, expires at t=110
    clock.advance(50)  # t=100 — expired if TTL had started at call start
    slow(1)
    assert calls == [1]
    clock.advance(11)  # t=111 — now past commit + 60
    slow(1)
    assert calls == [1, 1]


def test_expired_entry_is_removed_lazily():
    clock = FakeClock()
    backend = MemoryBackend()

    @cache(ttl="30s", clock=clock, backend=backend)
    def expensive(x):
        return x

    expensive(1)
    assert len(list(backend.iter_entries())) == 1
    clock.advance(31)
    # No background worker: the stale entry is still stored until observed.
    assert len(list(backend.iter_entries())) == 1
    expensive(1)  # observation triggers removal + recompute + fresh commit
    entries = list(backend.iter_entries())
    assert len(entries) == 1
    assert entries[0][1].created_at == clock.now


def test_string_ttl_end_to_end():
    clock = FakeClock()
    calls = []

    @cache(ttl="10m", clock=clock)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    clock.advance(599)
    expensive(1)
    clock.advance(2)
    expensive(1)
    assert calls == [1, 1]


def test_no_ttl_never_expires():
    clock = FakeClock()
    calls = []

    @cache(clock=clock)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    clock.advance(10**9)
    expensive(1)
    assert calls == [1]


def test_invalid_ttl_fails_at_decoration_time():
    with pytest.raises(InvalidTTLError):

        @cache(ttl="10x")
        def expensive(x):
            return x


def test_expires_at_metadata_is_stored():
    clock = FakeClock(now=100.0)
    backend = MemoryBackend()

    @cache(ttl=60, clock=clock, backend=backend)
    def expensive(x):
        return x

    expensive(1)
    ((_, entry),) = backend.iter_entries()
    assert entry.created_at == 100.0
    assert entry.expires_at == 160.0
