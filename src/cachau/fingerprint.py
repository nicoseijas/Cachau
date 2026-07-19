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

    For JIT dispatchers, the digest covers the original Python function plus
    the semantically relevant compile options — changing ``fastmath`` or
    ``parallel`` changes observable results, so it must invalidate.
    """
    hasher = hashlib.sha256()
    parts = _dispatcher_parts(func)
    if parts is not None:
        py_func, options = parts
        _feed_code(hasher, py_func.__code__)
        relevant = sorted(
            (name, _stable_option_value(options[name]))
            for name in _SEMANTIC_TARGET_OPTIONS
            if name in options
        )
        hasher.update(b"|target-options:")
        hasher.update(repr(relevant).encode())
    else:
        _feed_code(hasher, func.__code__)
    return hasher.hexdigest()


def _feed_code(hasher: Any, code: types.CodeType) -> None:
    hasher.update(code.co_code)
    hasher.update(",".join(code.co_names).encode())
    hasher.update(",".join(code.co_varnames).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _feed_code(hasher, const)
        else:
            hasher.update(repr(const).encode())
