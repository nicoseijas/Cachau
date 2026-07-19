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


def wait_for_waiters(cached_func, count, timeout=10.0):
    """Deterministically wait until `count` threads hold/wait on flight locks."""
    deadline = time.monotonic() + timeout
    while cached_func.cache._flights.total_waiters() < count:
        assert time.monotonic() < deadline, "waiters never arrived"
        time.sleep(0.005)


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
    wait_for_waiters(slow, 3)  # leader + both followers on the per-key lock
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
    wait_for_waiters(flaky, 2)  # leader + follower on the per-key lock
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


def test_invalidate_racing_a_read_is_never_served_stale(monkeypatch):
    """A quarantining invalidate that lands between the fast path's pending
    check and its backend read must still prevent serving the stale entry."""
    from cachau.memory import MemoryBackend

    state = {"armed": False}

    class RacingBackend(MemoryBackend):
        def get(self, key):
            entry = super().get(key)
            if state["armed"]:
                state["armed"] = False
                # Simulate a concurrent invalidate whose physical delete
                # fails, interleaved exactly after this read began.
                raising = OSError("locked")

                def failing_delete(k):
                    raise raising

                original = self.delete
                self.delete = failing_delete
                try:
                    expensive.cache.invalidate(1)
                finally:
                    self.delete = original
            return entry

    backend = RacingBackend()
    calls = []

    @cache(backend=backend)
    def expensive(x):
        calls.append(x)
        return x * 2

    expensive(1)
    state["armed"] = True
    assert expensive(1) == 2  # must recompute: the entry was condemned mid-read
    assert calls == [1, 1]
    assert expensive.cache.stats().miss_invalidated == 1


def test_stress_concurrent_invalidates_and_calls():
    """clear()/invalidate() racing live flights: no crashes, correct values."""
    stop = threading.Event()
    failures = []

    @cache
    def expensive(x):
        return x * 2

    def caller(seed):
        try:
            for i in range(300):
                key = (seed + i) % 4
                assert expensive(key) == key * 2
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            failures.append(exc)

    def chaos():
        try:
            while not stop.is_set():
                expensive.cache.invalidate(1)
                expensive.cache.invalidate(2)
                expensive.cache.clear()
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    chaos_thread = threading.Thread(target=chaos)
    chaos_thread.start()
    callers = [threading.Thread(target=caller, args=(n,)) for n in range(6)]
    for thread in callers:
        thread.start()
    for thread in callers:
        thread.join(timeout=60)
    stop.set()
    chaos_thread.join(timeout=10)

    assert failures == []


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
