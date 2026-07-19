"""Function identity: namespace and implementation fingerprint.

Changing a function's implementation must invalidate previously cached results
(GUIDELINES.md §7). The fingerprint hashes the code object — bytecode, names,
and constants, recursing into nested code objects so their memory addresses
never leak into the digest. It deliberately ignores volatile details (line
numbers, filenames) so that moving a function does not invalidate its cache.

Known limitation (Phase 0): the fingerprint covers only the function's own
code object. Values captured by closure and the implementations of other
functions it calls (globals, imports) are not included — if they affect the
result, pass them as arguments or declare them as dependencies.
"""

from __future__ import annotations

import hashlib
import types
from typing import Any, Callable

from cachau.errors import ConfigurationError, UnhashableArgumentError
from cachau.keys import _digest_value


# Numba target options that can change observable numeric semantics
# (GUIDELINES.md §14): fastmath reassociates floats, parallel changes
# reduction order, boundscheck raises where UB stood, error_model changes
# division semantics, nopython/forceobj switch execution modes.
_SEMANTIC_TARGET_OPTIONS = (
    "boundscheck",
    "error_model",
    "fastmath",
    "forceobj",
    "nopython",
    "parallel",
)


def _dispatcher_parts(
    func: Callable[..., Any],
) -> tuple[Callable[..., Any], dict] | None:
    """Detect a JIT dispatcher (Numba-style) by duck-typing, never by import.

    A dispatcher wraps the original Python function (``py_func``) and carries
    its compile options (``targetoptions``); its own ``repr``/memory identity
    must never be part of a fingerprint.
    """
    py_func = getattr(func, "py_func", None)
    options = getattr(func, "targetoptions", None)
    if callable(py_func) and isinstance(options, dict):
        return py_func, options
    return None


def is_jit_dispatcher(func: Callable[..., Any]) -> bool:
    return _dispatcher_parts(func) is not None


def _stable_option_value(value: Any) -> Any:
    # fastmath may be a set of LLVM flags; order must not matter.
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(str(item) for item in value))
    return value


def function_namespace(func: Callable[..., Any]) -> str:
    """Stable identity for *which function this is*: ``module.qualname``."""
    return f"{func.__module__}.{func.__qualname__}"


def function_fingerprint(func: Callable[..., Any]) -> str:
    """Stable identity for *what the function does*: a digest of its code.

    Covers the code object AND closure-captured values: two functions from
    the same factory with different captured parameters compute different
    results and must never share a fingerprint. For JIT dispatchers, the
    digest additionally covers the semantically relevant compile options
    (``fastmath``, ``parallel``, ..., and ``locals=`` type forcing) —
    changing them changes observable results, so it must invalidate.
    """
    return _fingerprint(func, seen=set())


def _fingerprint(func: Callable[..., Any], seen: set[int]) -> str:
    seen.add(id(func))
    hasher = hashlib.sha256()
    parts = _dispatcher_parts(func)
    if parts is not None:
        py_func, options = parts
        seen.add(id(py_func))
        _feed_function(hasher, py_func, seen)
        relevant = sorted(
            (name, _stable_option_value(options[name]))
            for name in _SEMANTIC_TARGET_OPTIONS
            if name in options
        )
        hasher.update(b"|target-options:")
        hasher.update(repr(relevant).encode())
        forced_locals = getattr(func, "locals", None)
        if isinstance(forced_locals, dict) and forced_locals:
            hasher.update(b"|locals:")
            hasher.update(
                repr(sorted((k, str(v)) for k, v in forced_locals.items())).encode()
            )
    else:
        if getattr(func, "__code__", None) is None:
            raise ConfigurationError(
                f"cannot fingerprint {type(func).__qualname__!r}: it is neither "
                f"a plain Python function nor a py_func-carrying dispatcher "
                f"(@vectorize-style ufuncs are unsupported at the boundary — "
                f"wrap the call in a plain function)"
            )
        _feed_function(hasher, func, seen)
    return hasher.hexdigest()


def _feed_function(hasher: Any, func: Callable[..., Any], seen: set[int]) -> None:
    _feed_code(hasher, func.__code__)
    _feed_closure(hasher, func, seen)


def _feed_closure(hasher: Any, func: Callable[..., Any], seen: set[int]) -> None:
    """Fold closure-captured values into the identity.

    Captured values determine results just as arguments do (a factory-made
    kernel closing over ``n`` computes ``x + n``). Hashable captures use the
    same type-tagged digest as arguments; captured functions recurse into
    their own fingerprint; opaque captures fall back to instance identity —
    stable within the process, deliberately unstable across restarts, so
    persisted reuse degrades to a safe MISS instead of guessing.
    """
    code = func.__code__
    closure = getattr(func, "__closure__", None) or ()
    if not code.co_freevars or not closure:
        return
    hasher.update(b"|closure:")
    for name, cell in zip(code.co_freevars, closure):
        hasher.update(name.encode())
        hasher.update(b"=")
        try:
            contents = cell.cell_contents
        except ValueError:  # self-referential/unbound cell (recursive def)
            hasher.update(b"<unbound>")
            continue
        try:
            hasher.update(_digest_value(contents))
        except UnhashableArgumentError:
            if callable(contents) and getattr(contents, "__code__", None) is not None:
                hasher.update(b"<function:")
                if id(contents) in seen:  # self/mutually-recursive capture
                    hasher.update(b"recursive")
                else:
                    hasher.update(_fingerprint(contents, seen).encode())
                hasher.update(b">")
            else:
                hasher.update(
                    f"<opaque:{type(contents).__qualname__}:{id(contents)}>".encode()
                )


def _feed_code(hasher: Any, code: types.CodeType) -> None:
    hasher.update(code.co_code)
    hasher.update(",".join(code.co_names).encode())
    hasher.update(",".join(code.co_varnames).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _feed_code(hasher, const)
        else:
            hasher.update(repr(const).encode())
