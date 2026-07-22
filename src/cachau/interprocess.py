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

- the winner crashes → its lock goes stale and is broken by age, and waiters
  have ALSO been waiting against their own deadline, after which they compute
  anyway;
- two waiters break a stale lock together → the exclusive create arbitrates
  exactly one new winner;
- a foreign process deletes our held lock mid-compute → at most one extra
  computer, no worse than before this module existed;
- clock skew between machines misjudges staleness → an early break (duplicate
  compute) or a longer-but-bounded wait, never corruption.

Liveness is judged by lock-file age alone. Pid-based checks are deliberately
absent: there is no portable probe (on Windows, ``os.kill(pid, 0)`` does not
test the process — it TERMINATES it), and age plus a bounded wait already
guarantees progress. The pid recorded inside the file is for a human holding
a stuck directory, nothing else.
"""

from __future__ import annotations

import os
import pathlib
import time
import uuid
from typing import Any, Callable

# Distinguishes "no fresh entry yet" from a cached value that is legitimately
# None. Never leaves this module's callers.
MISSING = object()


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
        return True

    def age_seconds(self) -> float | None:
        """Age of whatever lock file currently exists, or ``None`` if gone."""
        try:
            return max(0.0, time.time() - os.stat(self._path).st_mtime)
        except OSError:
            return None

    def break_stale(self) -> None:
        """Best-effort removal of a lock presumed dead by age."""
        try:
            self._path.unlink()
        except OSError:
            pass

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
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
            lock.break_stale()
            stale_broken += 1
            continue  # retry the acquire immediately; O_EXCL picks one winner
        if monotonic() >= deadline:
            return "timeout", None, stale_broken
        sleep(interval)
        interval = min(interval * 1.5, poll_cap)
