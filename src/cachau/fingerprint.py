"""Function identity: namespace and implementation fingerprint.

Changing a function's implementation must invalidate previously cached results
(GUIDELINES.md §7). The fingerprint hashes the code object — bytecode, names,
and constants, recursing into nested code objects so their memory addresses
never leak into the digest. It deliberately ignores volatile details (line
numbers, filenames) so that moving a function does not invalidate its cache.

Known limitation (Phase 0): values captured by closure are not part of the
fingerprint; treat them as arguments or dependencies if they affect results.
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
    return hasher.hexdigest()[:16]


def _feed_code(hasher: Any, code: types.CodeType) -> None:
    hasher.update(code.co_code)
    hasher.update(",".join(code.co_names).encode())
    hasher.update(",".join(code.co_varnames).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _feed_code(hasher, const)
        else:
            hasher.update(repr(const).encode())
