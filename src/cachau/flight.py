"""Same-key single-flight coordination.

N concurrent callers of the same absent key should not compute N times: one
leader computes and commits, the rest wait on a per-key lock and reuse the
committed result. Synchronization is strictly per key — there is no global
compute lock, so independent keys never serialize (GUIDELINES.md §10) and
thread configuration (e.g. Numba ``parallel=True``) is never touched.

The per-key lock is non-reentrant: a cached function must not call itself
with identical arguments (which would be infinite recursion regardless).
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterator


class KeyedLocks:
    """Refcounted per-key mutexes: the registry never leaks finished keys."""

    def __init__(self) -> None:
        self._registry_guard = threading.Lock()  # protects the dict, held briefly
        self._slots: dict[str, list] = {}  # key -> [lock, waiter_count]

    @contextlib.contextmanager
    def holding(self, key: str) -> Iterator[None]:
        with self._registry_guard:
            slot = self._slots.get(key)
            if slot is None:
                slot = [threading.Lock(), 0]
                self._slots[key] = slot
            slot[1] += 1
        slot[0].acquire()
        try:
            yield
        finally:
            slot[0].release()
            with self._registry_guard:
                slot[1] -= 1
                if slot[1] == 0:
                    self._slots.pop(key, None)

    def active_keys(self) -> int:
        with self._registry_guard:
            return len(self._slots)
