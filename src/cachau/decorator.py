"""The public ``@cache`` decorator and the per-function control surface."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar, overload

from cachau.backend import CacheBackend, CacheEntry
from cachau.durations import parse_ttl
from cachau.fingerprint import function_fingerprint, function_namespace
from cachau.keys import digest_arguments
from cachau.memory import MemoryBackend
from cachau.policy import LRUBudget
from cachau.sizes import estimate_size, parse_size

F = TypeVar("F", bound=Callable[..., Any])
Clock = Callable[[], float]
SizeOf = Callable[[Any], int]

_default_backend = MemoryBackend()


class CacheControl:
    """Attached to every cached function as ``func.cache``."""

    def __init__(
        self,
        *,
        namespace: str,
        fingerprint: str,
        backend: CacheBackend,
        ttl_seconds: float | None,
        max_memory_bytes: int | None,
        budget: LRUBudget | None,
    ) -> None:
        self.namespace = namespace
        self.fingerprint = fingerprint
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_bytes
        self.evictions = 0
        self.skipped_oversized = 0
        self._backend = backend
        self._budget = budget

    def clear(self) -> None:
        """Forget every stored result for this function."""
        self._backend.clear(namespace=self.namespace)
        if self._budget is not None:
            self._budget.reset()


@overload
def cache(func: F) -> F: ...


@overload
def cache(
    *,
    ttl: int | float | str | None = None,
    max_memory: int | str | None = None,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
    clock: Clock = ...,
    size_of: SizeOf = ...,
) -> Callable[[F], F]: ...


def cache(
    func: Callable[..., Any] | None = None,
    *,
    ttl: int | float | str | None = None,
    max_memory: int | str | None = None,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
    clock: Clock = time.time,
    size_of: SizeOf = estimate_size,
) -> Any:
    """Cache a function's results, keyed by its normalized arguments.

    Usable bare (``@cache``) or configured (``@cache(ttl="1h",
    max_memory="2GB")``). TTL accepts seconds or readable strings and starts
    when the result is committed. ``max_memory`` accepts bytes or size strings
    (``"512MB"``, ``"2GB"``; binary units) and bounds this function's entries
    with LRU eviction — an entry larger than the whole budget is computed and
    returned but never cached. Exceptions are never cached; unhashable
    arguments fail loudly.
    """
    if func is not None:
        return _wrap(func, ttl, max_memory, namespace, backend, clock, size_of)
    return lambda f: _wrap(f, ttl, max_memory, namespace, backend, clock, size_of)


def _wrap(
    func: Callable[..., Any],
    ttl: int | float | str | None,
    max_memory: int | str | None,
    namespace: str | None,
    backend: CacheBackend | None,
    clock: Clock,
    size_of: SizeOf,
) -> Callable[..., Any]:
    # Fail fast: bad configuration breaks at decoration time, not on first call.
    ttl_seconds = parse_ttl(ttl)
    max_memory_bytes = parse_size(max_memory)
    resolved_namespace = namespace if namespace is not None else function_namespace(func)
    fingerprint = function_fingerprint(func)
    store: CacheBackend = backend if backend is not None else _default_backend
    budget = LRUBudget(max_memory_bytes) if max_memory_bytes is not None else None
    last_observed = float("-inf")

    def now() -> float:
        # TTL uses wall-clock time because expires_at must survive process
        # restarts once persistence lands. Wall clocks can step backward (NTP,
        # VM resume); clamping to the last observed reading keeps time monotone
        # for this function so an entry can never appear to grow younger.
        nonlocal last_observed
        last_observed = max(last_observed, clock())
        return last_observed

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (
            f"{resolved_namespace}:{fingerprint}:"
            f"{digest_arguments(func, args, kwargs)}"
        )
        entry = store.get(key)
        if entry is not None:
            if not entry.is_expired(now()):
                if budget is not None:
                    budget.touch(key)
                return entry.value
            store.delete(key)
            if budget is not None:
                budget.forget(key)
        value = func(*args, **kwargs)
        committed_at = now()  # TTL starts at commit, not at call start
        size: int | None = None
        if budget is not None:
            size = int(size_of(value))
            if not budget.fits(size):
                # Oversized: compute, return, never cache, never flush the
                # cache to make room for a pathological entry.
                control.skipped_oversized += 1
                return value
            for evicted_key in budget.admit(key, size):
                store.delete(evicted_key)
                control.evictions += 1
        store.set(
            key,
            CacheEntry(
                value=value,
                namespace=resolved_namespace,
                created_at=committed_at,
                expires_at=(
                    committed_at + ttl_seconds if ttl_seconds is not None else None
                ),
                size=size,
            ),
        )
        return value

    control = CacheControl(
        namespace=resolved_namespace,
        fingerprint=fingerprint,
        backend=store,
        ttl_seconds=ttl_seconds,
        max_memory_bytes=max_memory_bytes,
        budget=budget,
    )
    wrapper.cache = control  # type: ignore[attr-defined]
    return wrapper
