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

F = TypeVar("F", bound=Callable[..., Any])
Clock = Callable[[], float]

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
    ) -> None:
        self.namespace = namespace
        self.fingerprint = fingerprint
        self.ttl_seconds = ttl_seconds
        self._backend = backend

    def clear(self) -> None:
        """Forget every stored result for this function."""
        self._backend.clear(namespace=self.namespace)


@overload
def cache(func: F) -> F: ...


@overload
def cache(
    *,
    ttl: int | float | str | None = None,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
    clock: Clock = ...,
) -> Callable[[F], F]: ...


def cache(
    func: Callable[..., Any] | None = None,
    *,
    ttl: int | float | str | None = None,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
    clock: Clock = time.time,
) -> Any:
    """Cache a function's results, keyed by its normalized arguments.

    Usable bare (``@cache``) or configured (``@cache(ttl="1h")``). TTL accepts
    seconds or readable strings (``"30s"``, ``"10m"``, ``"2h"``, ``"7d"``); it
    starts when the result is committed and expires lazily on access.
    Exceptions are never cached; unhashable arguments fail loudly.
    """
    if func is not None:
        return _wrap(func, ttl, namespace, backend, clock)
    return lambda f: _wrap(f, ttl, namespace, backend, clock)


def _wrap(
    func: Callable[..., Any],
    ttl: int | float | str | None,
    namespace: str | None,
    backend: CacheBackend | None,
    clock: Clock,
) -> Callable[..., Any]:
    ttl_seconds = parse_ttl(ttl)  # fail fast: a bad ttl breaks at decoration time
    resolved_namespace = namespace if namespace is not None else function_namespace(func)
    fingerprint = function_fingerprint(func)
    store: CacheBackend = backend if backend is not None else _default_backend
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
                return entry.value
            store.delete(key)
        value = func(*args, **kwargs)
        committed_at = now()  # TTL starts at commit, not at call start
        store.set(
            key,
            CacheEntry(
                value=value,
                namespace=resolved_namespace,
                created_at=committed_at,
                expires_at=(
                    committed_at + ttl_seconds if ttl_seconds is not None else None
                ),
            ),
        )
        return value

    wrapper.cache = CacheControl(  # type: ignore[attr-defined]
        namespace=resolved_namespace,
        fingerprint=fingerprint,
        backend=store,
        ttl_seconds=ttl_seconds,
    )
    return wrapper
