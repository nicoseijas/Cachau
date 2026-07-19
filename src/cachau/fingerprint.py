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


def function_namespace(func: Callable[..., Any]) -> str:
    """Stable identity for *which function this is*: ``module.qualname``."""
    return f"{func.__module__}.{func.__qualname__}"


def function_fingerprint(func: Callable[..., Any]) -> str:
    """Stable identity for *what the function does*: a digest of its code."""
    hasher = hashlib.sha256()
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
