"""Real-process stress for coalesce="processes" (#35).

The existing ``test_process_flight`` suite drives the crash and wedge paths with
*synthetic* lock files (``lock_path.write_text(...)`` + backdated mtimes). This
suite instead reproduces them with **real OS processes** that win the actual
advisory lock through the real mechanism and are then killed mid-compute or left
wedged — the failure modes exactly as they occur on the downstream 48-core farm
that reported the issue. Compute is counted by on-disk markers (one per real
``build`` execution), so the assertions hold across processes without sharing
per-process stats.

Design invariant under test: the mechanism may only REDUCE duplicate compute; it
never blocks a caller past a bounded wait and never deadlocks. Every failure
degrades to computing (atomic writes keep the store safe), never to a hang or a
wrong value.
"""

import os
import pathlib
import subprocess
import sys
import time

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# Small time knobs so the OS-level scenarios run fast but non-flaky: a crashed
# holder is recovered a few heartbeats after it stops beating; a wedged holder
# is out-waited within the bounded deadline. The wait must comfortably exceed
# the 0.4s compute plus interpreter-startup skew across K spawned processes,
# or a cold-burst waiter times out and duplicates the compute on a loaded box.
_HEARTBEAT = 0.1
_STALE_AFTER = 0.6
_WAIT_DEFAULT = 5.0

_WORKER = """
import os, pathlib, sys, time, uuid
sys.path.insert(0, {src!r})
import cachau.interprocess as _ip
import cachau.decorator as _dec
_ip._HEARTBEAT_SECONDS = {hb}
_dec._PROCESS_STALE_AFTER = {stale}
_dec._PROCESS_WAIT_DEFAULT = {wait}
from cachau import cache

base = pathlib.Path({base!r})
role = sys.argv[1]

@cache(persist=str(base / "store"), coalesce="processes", namespace="shared")
def build(n):
    (base / ("compute-" + uuid.uuid4().hex + ".marker")).write_text(role)
    if role in ("hold", "wedge"):
        time.sleep(60)          # keep the lock (crash killed externally; wedge stays alive+beating)
    else:
        time.sleep(0.4)
    return n * 2

print(build(21), flush=True)
"""


def _write_worker(tmp_path: pathlib.Path) -> pathlib.Path:
    worker = tmp_path / "sworker.py"
    worker.write_text(
        _WORKER.format(src=SRC, hb=_HEARTBEAT, stale=_STALE_AFTER,
                       wait=_WAIT_DEFAULT, base=str(tmp_path)),
        encoding="utf-8",
    )
    return worker


def _spawn(worker: pathlib.Path, role: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(worker), role],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _markers(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    return list(tmp_path.glob("compute-*.marker"))


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _store_locks(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    return list((tmp_path / "store").glob("*.lock"))


# --------------------------------------------------------------------------- #
# Happy path at wider fan-out than the K=3 unit test.
# --------------------------------------------------------------------------- #


def test_real_cold_burst_k6_computes_once(tmp_path):
    worker = _write_worker(tmp_path)
    procs = [_spawn(worker, "normal") for _ in range(6)]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=90)
        assert p.returncode == 0, err
        outs.append(out.strip())
    assert outs == ["42"] * 6
    assert len(_markers(tmp_path)) == 1        # exactly one real compute; five coalesced
    assert not _store_locks(tmp_path)          # winner released its lock


# --------------------------------------------------------------------------- #
# A real holder is KILLED mid-compute: survivors break the stale lock and
# exactly one recomputes. Nobody hangs.
# --------------------------------------------------------------------------- #


def test_real_killed_holder_is_recovered(tmp_path):
    worker = _write_worker(tmp_path)
    holder = _spawn(worker, "hold")
    try:
        # wait until the holder actually owns the lock AND is inside build()
        assert _wait_until(
            lambda: _store_locks(tmp_path) and len(_markers(tmp_path)) == 1, timeout=30
        ), "holder never acquired the lock / started computing"
        holder.kill()              # hard kill: heartbeat stops, lock left behind stale
        holder.wait(timeout=10)
    finally:
        if holder.poll() is None:
            holder.kill()

    started = time.perf_counter()
    survivors = [_spawn(worker, "normal") for _ in range(4)]
    outs = []
    for p in survivors:
        out, err = p.communicate(timeout=90)
        assert p.returncode == 0, err
        outs.append(out.strip())
    elapsed = time.perf_counter() - started

    assert outs == ["42"] * 4                  # all correct despite the crash
    # killed holder's marker (1) + exactly one recompute after the stale break.
    assert len(_markers(tmp_path)) == 2
    assert not _store_locks(tmp_path)          # stale lock broken and recomputer released it
    assert elapsed < 60                        # bounded — no deadlock on the dead holder's lock


# --------------------------------------------------------------------------- #
# A real holder is WEDGED (alive, heartbeating, never returns): survivors must
# NOT break its live lock — they run out their bounded wait and degrade to
# computing. No hang, correct values.
# --------------------------------------------------------------------------- #


def test_real_wedged_holder_forces_bounded_degrade(tmp_path):
    worker = _write_worker(tmp_path)
    wedged = _spawn(worker, "wedge")
    try:
        assert _wait_until(
            lambda: _store_locks(tmp_path) and len(_markers(tmp_path)) == 1, timeout=30
        ), "wedged holder never acquired the lock"

        started = time.perf_counter()
        survivors = [_spawn(worker, "normal") for _ in range(4)]
        outs = []
        for p in survivors:
            out, err = p.communicate(timeout=90)
            assert p.returncode == 0, err
            outs.append(out.strip())
        elapsed = time.perf_counter() - started

        assert outs == ["42"] * 4              # correct: a wedged holder can't hang callers
        # Holder's marker (1) + at least one survivor that timed out and
        # degraded to computing. How many degrade is timing-dependent: the
        # first degrader COMMITS its result, and any survivor still polling
        # is served by that commit instead of computing — the mechanism
        # reducing duplicate compute, which is the invariant, not a count.
        assert 2 <= len(_markers(tmp_path)) <= 5
        assert _store_locks(tmp_path)          # the live lock is untouched (still the wedged holder's)
        # each survivor waited ~ _WAIT_DEFAULT then computed; nowhere near the holder's 60s sleep.
        assert elapsed < 30
    finally:
        wedged.kill()
        wedged.wait(timeout=10)
