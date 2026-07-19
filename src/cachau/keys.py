"""Invocation normalization and argument digests.

A cache key must represent normalized arguments: semantically equivalent calls
(positional vs. keyword, defaults applied) share a digest, while values that
merely compare equal across types (``1 == 1.0 == True``) never collide — every
value is hashed with an explicit type tag.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import pathlib
from typing import Any, Callable

from cachau.errors import UnhashableArgumentError


def normalize_call(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[str, Any], ...]:
    """Bind a call to ``func``'s signature and return its canonical form."""
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.items())


def digest_arguments(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    """Digest a normalized invocation of ``func`` into a stable hex string."""
    hasher = hashlib.sha256()
    for name, value in normalize_call(func, args, kwargs):
        hasher.update(name.encode())
        hasher.update(b"=")
        try:
            _feed(hasher, value)
        except UnhashableArgumentError as exc:
            raise UnhashableArgumentError(
                f"argument {name!r} of {func.__qualname__}() cannot be hashed: "
                f"{exc}. Provide an explicit key= or exclude it with ignore=."
            ) from None
        hasher.update(b";")
    return hasher.hexdigest()


def _digest_value(value: Any) -> bytes:
    hasher = hashlib.sha256()
    _feed(hasher, value)
    return hasher.digest()


def _feed(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"none:")
    elif isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        hasher.update(b"int:" + str(value).encode())
    elif isinstance(value, float):
        hasher.update(b"float:" + repr(value).encode())
    elif isinstance(value, complex):
        hasher.update(b"complex:" + repr(value).encode())
    elif isinstance(value, str):
        hasher.update(b"str:")
        hasher.update(value.encode())
    elif isinstance(value, (bytes, bytearray)):
        hasher.update(b"bytes:")
        hasher.update(bytes(value))
    elif isinstance(value, pathlib.PurePath):
        hasher.update(b"path:")
        hasher.update(str(value).encode())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        hasher.update(b"dataclass:")
        hasher.update(type(value).__qualname__.encode())
        for field in dataclasses.fields(value):
            hasher.update(field.name.encode())
            hasher.update(b"=")
            _feed(hasher, getattr(value, field.name))
    elif isinstance(value, tuple):
        hasher.update(b"tuple:" + str(len(value)).encode() + b":")
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, list):
        hasher.update(b"list:" + str(len(value)).encode() + b":")
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, dict):
        hasher.update(b"dict:" + str(len(value)).encode() + b":")
        # Sort by each key's own digest so insertion order is irrelevant even
        # for key types that are not mutually orderable.
        items = sorted(value.items(), key=lambda kv: _digest_value(kv[0]))
        for key, item in items:
            _feed(hasher, key)
            hasher.update(b"->")
            _feed(hasher, item)
    elif isinstance(value, (set, frozenset)):
        hasher.update(b"set:" + str(len(value)).encode() + b":")
        for member_digest in sorted(_digest_value(member) for member in value):
            hasher.update(member_digest)
    else:
        raise UnhashableArgumentError(
            f"type {type(value).__qualname__!r} has no cachau hashing support"
        )
