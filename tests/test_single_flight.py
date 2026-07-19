"""Same-key single-flight: N concurrent callers, one computation (GUIDELINES.md §10)."""

import threading
import time

from cachau import cache
from cachau.flight import KeyedLocks


def run_threads(target, count):
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    return threads


def test_concurrent_same_key_computes_once():
    compute_started = threading.Event()
    release_compute = threading.Event()
    calls = []
    results = []

    @cache
    def slow(x):
        calls.append(x)
        compute_started.set()
        assert release_compute.wait(timeout=10)
        return x * 2

    leader = threading.Thread(target=lambda: results.append(slow(1)))
    leader.start()
    assert compute_started.wait(timeout=10)

    followers = run_threads(lambda: results.append(slow(1)), 2)
    time.sleep(0.3)  # let followers reach the per-key lock
    release_compute.set()
    leader.join(timeout=10)
    for thread in followers:
        thread.join(timeout=10)

    assert results == [2, 2, 2]
    assert calls == [1]  # one computation for three callers
    stats = slow.cache.stats()
    assert stats.misses == 1
    assert stats.hits == 2
    assert stats.coalesced_hits == 2


def test_different_keys_never_serialize():
    """A slow computation for one key must not block another key (no global lock)."""
    first_started = threading.Event()
    second_finished = threading.Event()
    outcomes = {}

    @cache
    def slow(x):
        if x == 1:
            first_started.set()
            # Deadlock detector: if key 2 were serialized behind key 1, this
            # wait could never be satisfied.
            assert second_finished.wait(timeout=10), "second key was blocked"
            return "one"
        return "two"

    first = threading.Thread(target=lambda: outcomes.setdefault(1, slow(1)))
    first.start()
    assert first_started.wait(timeout=10)

    outcomes[2] = slow(2)  # runs while key 1 is still computing
    second_finished.set()
    first.join(timeout=10)

    assert outcomes == {1: "one", 2: "two"}


def test_leader_exception_lets_a_follower_compute():
    compute_started = threading.Event()
    release_compute = threading.Event()
    attempts = []
    results = []
    errors = []

    @cache
    def flaky(x):
        attempts.append(x)
        if len(attempts) == 1:
            compute_started.set()
            assert release_compute.wait(timeout=10)
            raise ValueError("leader fails")
        return x * 2

    def leader_call():
        try:
            flaky(1)
        except ValueError as exc:
            errors.append(exc)

    leader = threading.Thread(target=leader_call)
    leader.start()
    assert compute_started.wait(timeout=10)

    follower = threading.Thread(target=lambda: results.append(flaky(1)))
    follower.start()
    time.sleep(0.2)  # follower reaches the per-key lock
    release_compute.set()
    leader.join(timeout=10)
    follower.join(timeout=10)

    assert len(errors) == 1  # the exception was not cached or swallowed
    assert results == [2]  # the follower computed for itself
    assert attempts == [1, 1]


def test_lock_registry_does_not_leak():
    locks = KeyedLocks()
    with locks.holding("a"):
        assert locks.active_keys() == 1
    with locks.holding("b"):
        pass
    assert locks.active_keys() == 0


def test_sequential_behavior_is_unchanged():
    calls = []

    @cache
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    expensive(1)
    expensive(2)
    assert calls == [1, 2]
    stats = expensive.cache.stats()
    assert stats.hits == 1
    assert stats.coalesced_hits == 0


def test_stress_many_threads_many_keys():
    """No corruption or deadlock with concurrent mixed keys."""
    calls = []
    call_lock = threading.Lock()

    @cache(max_memory=10_000, size_of=lambda v: 10)
    def expensive(x):
        with call_lock:
            calls.append(x)
        return x * 2

    def worker(seed):
        for i in range(20):
            key = (seed + i) % 8
            assert expensive(key) == key * 2

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    # Every distinct key computed at least once and results were always right;
    # single-flight and caching keep recomputation bounded by the key count.
    assert set(calls) == set(range(8))
    assert len(calls) == 8
