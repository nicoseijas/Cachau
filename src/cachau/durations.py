"""Parsing of human-readable TTL values.

Accepted forms: a positive number of seconds (``60``, ``0.5``) or a string
with a unit suffix (``"30s"``, ``"10m"``, ``"2h"``, ``"7d"``). Anything else
fails fast with :class:`InvalidTTLError` — a silently misread TTL would make
cache behavior unpredictable.
"""

from __future__ import annotations

import math
from typing import Any

from cachau.errors import InvalidTTLError

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_ttl(value: Any) -> float | None:
    """Normalize a TTL to seconds, or ``None`` when no TTL applies."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidTTLError(f"ttl must be a number or duration string, got {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        unit = value[-1:]
        scale = _UNIT_SECONDS.get(unit)
        if scale is None:
            raise InvalidTTLError(
                f"ttl string must end in one of {sorted(_UNIT_SECONDS)}, got {value!r}"
            )
        try:
            seconds = float(value[:-1]) * scale
        except ValueError:
            raise InvalidTTLError(f"ttl string has no numeric part: {value!r}") from None
    else:
        raise InvalidTTLError(
            f"ttl must be a number or duration string, got {type(value).__qualname__}"
        )
    if not math.isfinite(seconds) or seconds <= 0:
        raise InvalidTTLError(f"ttl must be a positive finite duration, got {value!r}")
    return seconds
