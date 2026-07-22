"""coalesce="processes": cross-process single-flight for persistent caches (#35).

The design invariant: the mechanism may only REDUCE duplicate compute — it can
never block a caller beyond a bounded wait, and every failure mode degrades to
the uncoordinated behavior (compute redundantly; atomic writes keep the store
safe), never to a deadlock or a wrong value.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

from cachau import cache
from cachau.backend import CacheEntry
from cachau.errors import ConfigurationError
from cachau.interprocess import ProcessLock


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_coalesce_rejects_unknown_values():
    with pytest.raises(ConfigurationError):

        @cache(coalesce="fleet")
        def f(x):
            return x


def test_processes_requires_a_shared_on_disk_store():
    with pytest.raises(ConfigurationError):

        @cache(coalesce="processes")  # memory backend: nothing to share
        def f(x):
            return x


def test_threads_is_the_default_and_creates_no_lock_files(tmp_path):
    @cache(persist=str(tmp_path))
    def f(x):
        return x

    f(1)
    assert not list(tmp_path.glob("*.lock"))


# --------------------------------------------------------------------------- #
# Winner path
# --------------------------------------------------------------------------- #


def test_winner_computes_normally_and_leaves_no_lock_behind(tmp_path):
    calls = []

    @cache(persist=str(tmp_path), coalesce="processes")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(1) == 2
    assert f(1) == 2  # plain HIT afterwards
    assert calls == [1]
    assert not list(tmp_path.glob("*.lock"))
    stats = f.cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1


# --------------------------------------------------------------------------- #
# Waiter paths
# --------------------------------------------------------------------------- #


def test_waiter_is_served_by_another_processes_commit(tmp_path):
    calls = []

    @cache(persist=str(tmp_path), coalesce="processes")
    def f(x):
        calls.append(x)
        return x * 2

    control = f.cache
    key = control._key_builder(1)
    backend = control._backend
    backend.lock_path(key).write_text("held by another process")

    def commit_from_elsewhere():
        time.sleep(0.3)
        backend.set(
            key, CacheEntry(value=2, namespace=control.namespace, created_at=time.time())
        )

    committer = threading.Thread(target=commit_from_elsewhere)
    committer.start()
    try:
        assert f(1) == 2
    finally:
        committer.join()
    assert calls == []  # never computed locally
    stats = control.stats()
    assert stats.hits == 1
    assert stats.process_coalesced_hits == 1
    assert stats.misses == 0


def test_stale_lock_is_broken_and_the_caller_computes(tmp_path):
    calls = []

    @cache(persist=str(tmp_path), coalesce="processes")
    def f(x):
        calls.append(x)
        return x * 2

    control = f.cache
    lock_path = control._backend.lock_path(control._key_builder(1))
    lock_path.write_text("crashed holder")
    long_dead = time.time() - 10_000
    os.utime(lock_path, (long_dead, long_dead))
    assert f(1) == 2
    assert calls == [1]
    assert control.stats().stale_locks_broken == 1
    assert not list(tmp_path.glob("*.lock"))  # winner released its own lock


def test_bounded_wait_then_compute_anyway(tmp_path, monkeypatch):
    monkeypatch.setattr("cachau.decorator._PROCESS_WAIT_DEFAULT", 0.2)
    calls = []

    @cache(persist=str(tmp_path), coalesce="processes")
    def f(x):
        calls.append(x)
        return x * 2

    control = f.cache
    lock_path = control._backend.lock_path(control._key_builder(1))
    lock_path.write_text("live but silent holder")  # fresh: never goes stale here
    started = time.perf_counter()
    assert f(1) == 2
    waited = time.perf_counter() - started
    assert calls == [1]  # computed anyway — a wedged holder cannot hang callers
    assert waited < 5.0
    assert control.stats().process_flight_timeouts == 1
    assert lock_path.exists()  # not ours: a timeout must not delete a live lock


# --------------------------------------------------------------------------- #
# ProcessLock
# --------------------------------------------------------------------------- #


def test_lock_is_exclusive(tmp_path):
    first = ProcessLock(tmp_path / "k.lock")
    second = ProcessLock(tmp_path / "k.lock")
    assert first.try_acquire() is True
    assert second.try_acquire() is False
    first.release()
    assert second.try_acquire() is True
    second.release()


def test_release_only_removes_its_own_lock(tmp_path):
    path = tmp_path / "k.lock"
    lock = ProcessLock(path)
    assert lock.try_acquire()
    path.write_text("broken as stale and re-acquired by someone else")
    lock.release()
    assert path.exists()  # not ours anymore: left alone


# --------------------------------------------------------------------------- #
# The real thing: a cold burst across processes computes once
# --------------------------------------------------------------------------- #


def test_cold_burst_across_real_processes_computes_once(tmp_path):
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    worker = tmp_path / "worker.py"
    worker.write_text(
        f"""
import pathlib
import sys
import time
import uuid

sys.path.insert(0, {src!r})
from cachau import cache

base = pathlib.Path({str(tmp_path)!r})


@cache(persist=str(base / "store"), coalesce="processes")
def build(n):
    (base / f"compute-{{uuid.uuid4().hex}}.marker").write_text("x")
    time.sleep(1.0)
    return n * 2


print(build(21))
""",
        encoding="utf-8",
    )
    workers = [
        subprocess.Popen(
            [sys.executable, str(worker)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    outputs = []
    for process in workers:
        stdout, stderr = process.communicate(timeout=90)
        assert process.returncode == 0, stderr
        outputs.append(stdout.strip())
    assert outputs == ["42", "42", "42"]
    markers = list(tmp_path.glob("compute-*.marker"))
    assert len(markers) == 1  # exactly one process computed; the burst coalesced
    assert not list((tmp_path / "store").glob("*.lock"))


# --------------------------------------------------------------------------- #
# Heartbeat: staleness decoupled from compute duration
# --------------------------------------------------------------------------- #


def test_heartbeat_prevents_preemption_of_a_healthy_slow_holder(
    tmp_path, monkeypatch
):
    """The downstream stress-test finding: with staleness tied to a fixed
    threshold, a waiter declares a healthy still-computing holder's lock stale
    and preempts it (2 computes instead of 1). The heartbeat keeps a live
    holder's lock fresh no matter how long the compute runs."""
    # Margins sized for loaded CI runners: a false break needs the holder's
    # heartbeat thread starved past 1.5s (seen flaking at 0.3s), while the
    # discriminating power is intact — without the heartbeat the lock's age
    # reaches the full 3s compute, well past the threshold, deterministically.
    monkeypatch.setattr("cachau.interprocess._HEARTBEAT_SECONDS", 0.1)
    monkeypatch.setattr("cachau.decorator._PROCESS_STALE_AFTER", 1.5)
    computes = []

    def make():  # two functions, same namespace+body: same key, separate flights
        @cache(persist=str(tmp_path), coalesce="processes", namespace="shared")
        def f(x):
            computes.append(x)
            time.sleep(3.0)  # far longer than the stale threshold
            return x * 2

        return f

    holder_fn, waiter_fn = make(), make()
    results = {}
    holder = threading.Thread(target=lambda: results.setdefault("h", holder_fn(1)))
    holder.start()
    time.sleep(0.5)  # let the holder win the lock and get deep into computing
    results["w"] = waiter_fn(1)
    holder.join()
    assert results == {"h": 2, "w": 2}
    assert len(computes) == 1  # the healthy holder was never preempted
    waiter_stats = waiter_fn.cache.stats()
    assert waiter_stats.stale_locks_broken == 0
    assert waiter_stats.process_coalesced_hits == 1


def test_heartbeat_refreshes_the_lock_mtime(tmp_path, monkeypatch):
    monkeypatch.setattr("cachau.interprocess._HEARTBEAT_SECONDS", 0.05)
    lock = ProcessLock(tmp_path / "k.lock")
    assert lock.try_acquire()
    try:
        time.sleep(0.4)
        age = lock.age_seconds()
        assert age is not None and age < 0.3  # kept fresh well past creation
    finally:
        lock.release()


def test_heartbeat_stops_on_release(tmp_path, monkeypatch):
    monkeypatch.setattr("cachau.interprocess._HEARTBEAT_SECONDS", 0.05)
    lock = ProcessLock(tmp_path / "k.lock")
    assert lock.try_acquire()
    assert lock._heartbeat_thread is not None and lock._heartbeat_thread.is_alive()
    lock.release()
    lock._heartbeat_thread.join(2.0)
    assert not lock._heartbeat_thread.is_alive()


# --------------------------------------------------------------------------- #
# Stale-break re-check: never break a just-reacquired fresh lock (#58)
# --------------------------------------------------------------------------- #


def test_break_stale_skips_a_lock_that_became_fresh_again(tmp_path):
    """Thundering-break window: waiter C measures the DEAD holder's lock as
    stale, but before C unlinks, waiter A breaks it and re-acquires. C must
    re-check and leave A's fresh lock alone, or two processes compute."""
    lock_path = tmp_path / "key.lock"
    lock_path.write_text("fresh new holder")  # A's re-acquired lock, mtime now
    lock = ProcessLock(lock_path)
    assert lock.break_stale(stale_after=5.0) is False
    assert lock_path.exists()


def test_break_stale_removes_a_lock_that_is_still_stale(tmp_path):
    lock_path = tmp_path / "key.lock"
    lock_path.write_text("crashed holder")
    long_dead = time.time() - 10_000
    os.utime(lock_path, (long_dead, long_dead))
    lock = ProcessLock(lock_path)
    assert lock.break_stale(stale_after=5.0) is True
    assert not lock_path.exists()


def test_break_stale_handles_a_lock_that_vanished(tmp_path):
    lock = ProcessLock(tmp_path / "gone.lock")
    assert lock.break_stale(stale_after=5.0) is False
