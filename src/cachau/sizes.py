"""Size parsing and in-memory size estimation.

What "size" means in cachau (GUIDELINES.md §5): the **approximate deep
in-memory size of the stored value in bytes**, measured once at commit time.
It is an estimate for budgeting, not an exact accounting — array-like objects
report their buffer (``nbytes``/``memory_usage``), containers add their
children (shared references counted once), everything else falls back to
``sys.getsizeof``.

Limit units are binary: ``KB`` = 1024 B, ``MB`` = 1024², ``GB`` = 1024³,
``TB`` = 1024⁴.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

from cachau.errors import InvalidSizeError

_UNIT_BYTES = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

# Fallback when an object's type reports no __sizeof__ at all.
_DEFAULT_OBJECT_SIZE = 64


def parse_size(value: Any) -> int | None:
    """Normalize a memory limit to bytes, or ``None`` when unbounded."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidSizeError(f"max_memory must be bytes or a size string, got {value!r}")
    if isinstance(value, int):
        size = value
    elif isinstance(value, str):
        text = value.upper()
        unit = next(
            (u for u in ("KB", "MB", "GB", "TB", "B") if text.endswith(u)), None
        )
        if unit is None:
            raise InvalidSizeError(
                f"max_memory string must end in one of "
                f"{sorted(_UNIT_BYTES)} (binary units, KB = 1024 B), got {value!r}"
            )
        try:
            size = int(float(text[: -len(unit)]) * _UNIT_BYTES[unit])
        except ValueError:
            raise InvalidSizeError(
                f"max_memory string must be '<number><unit>', e.g. '512MB' or "
                f"'1.5GB', got {value!r}"
            ) from None
    else:
        raise InvalidSizeError(
            f"max_memory must be an int (bytes) or a size string, got "
            f"{type(value).__qualname__}"
        )
    if size <= 0:
        raise InvalidSizeError(f"max_memory must be positive, got {value!r}")
    return size


def estimate_size(value: Any) -> int:
    """Approximate deep in-memory size of ``value`` in bytes."""
    return _walk(value, seen=set())


def _walk(value: Any, seen: set[int]) -> int:
    if id(value) in seen:
        return 0
    seen.add(id(value))

    memory_usage = getattr(value, "memory_usage", None)
    if callable(memory_usage):  # pandas DataFrame/Series duck-typing
        try:
            reported = memory_usage(deep=True)
            return int(reported.sum() if hasattr(reported, "sum") else reported)
        except Exception:  # noqa: BLE001 - fall through to generic estimation
            pass

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(nbytes, bool):  # NumPy duck-typing
        return nbytes + sys.getsizeof(value, _DEFAULT_OBJECT_SIZE)

    base = sys.getsizeof(value, _DEFAULT_OBJECT_SIZE)
    if isinstance(value, dict):
        return base + sum(
            _walk(key, seen) + _walk(item, seen) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return base + sum(_walk(item, seen) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return base + sum(
            _walk(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        )
    return base
