"""Cross-process single-flight: advisory per-key lock files (#35).

In-process, ``KeyedLocks`` already coalesces concurrent callers of one key.
Across processes there is no shared memory — but a persistent cache already
shares a directory, so the flight token becomes a file: ``O_CREAT | O_EXCL``
is atomic on POSIX and Windows, which makes "exactly one winner" cheap to
arbitrate without a coordinator.

The design invariant: this mechanism may only REDUCE duplicate compute — it
must never block a caller beyond a bounded wait and never deadlock. Every
failure mode degrades to the uncoordinated behavior (compute redundantly;
atomic entry writes keep the store safe):

- the winner crashes → its heartbeat stops, the lock goes stale and is broken,
  and waiters have ALSO been waiting against their own deadline, after which
  they compute anyway;
- two waiters break a stale lock together → the exclusive create arbitrates
  exactly one new winner;
- a foreign process deletes our held lock mid-compute → at most one extra
  computer, no worse than before this module existed;
- clock skew between machines misjudges staleness → an early break (duplicate
  compute) or a longer-but-bounded wait, never corruption.

Staleness means "the holder stopped HEARTBEATING", never "the compute is
taking long". The holder refreshes the lock's mtime every ``_HEARTBEAT_SECONDS``
from a daemon thread, so the two failure modes separate cleanly: a crashed
holder stops beating and is recovered within a few heartbeats; a wedged-but-
alive holder keeps beating, and waiters never touch its lock — they run out
their own deadline and degrade to computing. Tying staleness to compute time
instead is a trap the downstream 48-core stress test hit on its first run: any
fixed threshold below the compute duration preempts a HEALTHY holder (two
computes instead of one), and any threshold above it slows real-crash recovery.

Liveness is judged by heartbeat age alone. Pid-based checks are deliberately
absent: there is no portable probe (on Windows, ``os.kill(pid, 0)`` does not
test the process — it TERMINATES it), and the same stress test confirmed
timestamp-only recovery suffices. The pid recorded inside the file is for a
human holding a stuck directory, nothing else.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
import uuid
from typing import Any, Callable

# Distinguishes "no fresh entry yet" from a cached value that is legitimately
# None. Never leaves this module's callers.
MISSING = object()

# The holder refreshes its lock's mtime this often. Stale thresholds must sit
# a few multiples above it (k = 4-5 measured well downstream) with margin for
# coarse filesystem timestamps; see _PROCESS_STALE_AFTER in decorator.py.
_HEARTBEAT_SECONDS = 1.0


class ProcessLock:
    """One advisory lock file, owned via a unique token.

    Ownership matters on release: our lock may have been judged stale and
    broken while we computed, and the path may now hold ANOTHER process's
    lock — deleting that would double-break it. Release therefore removes the
    file only when it still contains our token. The read-then-unlink pair is
    not atomic; losing that tiny race deletes a successor's lock, which costs
    its waiters one bounded re-arbitration, never a wrong value.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = pathlib.Path(path)
        self._token = f"{os.getpid()}:{uuid.uuid4().hex}".encode()
        self._held = False
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    def try_acquire(self) -> bool:
        try:
            descriptor = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError:
            # Unwritable directory, path trouble: degrade to uncoordinated
            # compute rather than failing the call over a diagnostic file.
            return False
        try:
            os.write(descriptor, self._token)
        finally:
            os.close(descriptor)
        self._held = True
        self._start_heartbeat()
        return True

    def _start_heartbeat(self) -> None:
        # Daemon: dies with the process, which is exactly the signal — a
        # crashed holder stops beating and its lock goes stale. A live holder
        # beats even through a long GIL-holding compute (the interpreter
        # schedules threads every switch interval; only a C extension that
        # never releases the GIL for several heartbeats could starve this,
        # and that misjudgment costs one duplicate compute, never a hang).
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._beat, name="cachau-lock-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _beat(self) -> None:
        assert self._heartbeat_stop is not None
        while not self._heartbeat_stop.wait(_HEARTBEAT_SECONDS):
            try:
                os.utime(self._path)
            except OSError:
                return  # lock gone (foreign break): nothing left to keep fresh

    def age_seconds(self) -> float | None:
        """Seconds since the holder's last heartbeat, or ``None`` if no lock.

        A held lock's mtime advances every ``_HEARTBEAT_SECONDS``, so age
        measures silence, not how long the compute has been running.
        """
        try:
            return max(0.0, time.time() - os.stat(self._path).st_mtime)
        except OSError:
            return None

    def break_stale(self, stale_after: float) -> bool:
        """Best-effort removal of a lock that is STILL stale; True if removed.

        When a holder dies, every waiter crosses the staleness threshold
        together, and an unconditional unlink lets a late breaker remove the
        fresh lock a faster one just re-acquired via break -> O_EXCL — two
        computers for one key (#58). Re-checking the age immediately before
        the unlink shrinks that window from the whole measure-to-unlink gap
        to a stat-to-unlink TOCTOU of microseconds: a just-reacquired lock
        has age ~0 and is left alone. Losing the residual race stays benign
        (a bounded duplicate compute, atomic commits, correct values).
        """
        age = self.age_seconds()
        if age is None or age <= stale_after:
            return False
        try:
            self._path.unlink()
        except OSError:
            return False
        return True

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()  # daemon thread exits on its next wake
        try:
            content = self._path.read_bytes()
        except OSError:
            return
        if content != self._token:
            return  # broken as stale and re-acquired by someone else: not ours
        try:
            self._path.unlink()
        except OSError:
            pass


def coordinate(
    lock: ProcessLock,
    fresh_value: Callable[[], Any],
    *,
    max_wait: float,
    stale_after: float,
    poll_initial: float = 0.01,
    poll_cap: float = 0.2,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, Any, int]:
    """Join the cross-process flight for one key.

    ``fresh_value`` probes the store and returns the servable value or
    ``MISSING``. Returns ``(outcome, value, stale_broken)`` where outcome is
    ``"acquired"`` (this caller computes and must release the lock),
    ``"served"`` (another process committed while we waited), or ``"timeout"``
    (the deadline passed: compute without coordination — the wait is the only
    cost a wedged holder can ever impose).
    """
    deadline = monotonic() + max_wait
    interval = poll_initial
    stale_broken = 0
    while True:
        value = fresh_value()
        if value is not MISSING:
            return "served", value, stale_broken
        if lock.try_acquire():
            return "acquired", None, stale_broken
        age = lock.age_seconds()
        if age is not None and age > stale_after:
            if lock.break_stale(stale_after):
                stale_broken += 1
            continue  # retry the acquire immediately; O_EXCL picks one winner
        if monotonic() >= deadline:
            return "timeout", None, stale_broken
        sleep(interval)
        interval = min(interval * 1.5, poll_cap)
