"""The public ``@cache`` decorator and the per-function control surface."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

from cachau.backend import CacheBackend, CacheEntry
from cachau.fingerprint import function_fingerprint, function_namespace
from cachau.keys import digest_arguments
from cachau.memory import MemoryBackend

F = TypeVar("F", bound=Callable[..., Any])

_default_backend = MemoryBackend()


class CacheControl:
    """Attached to every cached function as ``func.cache``."""

    def __init__(self, namespace: str, fingerprint: str, backend: CacheBackend) -> None:
        self.namespace = namespace
        self.fingerprint = fingerprint
        self._backend = backend

    def clear(self) -> None:
        """Forget every stored result for this function."""
        self._backend.clear(namespace=self.namespace)


@overload
def cache(func: F) -> F: ...


@overload
def cache(
    *, namespace: str | None = None, backend: CacheBackend | None = None
) -> Callable[[F], F]: ...


def cache(
    func: Callable[..., Any] | None = None,
    *,
    namespace: str | None = None,
    backend: CacheBackend | None = None,
) -> Any:
    """Cache a function's results, keyed by its normalized arguments.

    Usable bare (``@cache``) or configured (``@cache(namespace="features.v2")``).
    Exceptions are never cached; unhashable arguments fail loudly.
    """
    if func is not None:
        return _wrap(func, namespace, backend)
    return lambda f: _wrap(f, namespace, backend)


def _wrap(
    func: Callable[..., Any],
    namespace: str | None,
    backend: CacheBackend | None,
) -> Callable[..., Any]:
    resolved_namespace = namespace if namespace is not None else function_namespace(func)
    fingerprint = function_fingerprint(func)
    store: CacheBackend = backend if backend is not None else _default_backend

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (
            f"{resolved_namespace}:{fingerprint}:"
            f"{digest_arguments(func, args, kwargs)}"
        )
        entry = store.get(key)
        if entry is not None:
            return entry.value
        value = func(*args, **kwargs)
        store.set(key, CacheEntry(value=value, namespace=resolved_namespace))
        return value

    wrapper.cache = CacheControl(resolved_namespace, fingerprint, store)  # type: ignore[attr-defined]
    return wrapper
