"""Invocation normalization and argument digests.

A cache key must represent normalized arguments: semantically equivalent calls
(positional vs. keyword, defaults applied) share a digest, while values that
merely compare equal across types (``1 == 1.0 == True``) never collide — every
value is hashed with an explicit type tag.

Encoding scheme (collision safety): the byte stream fed to the hasher must be
unambiguously decodable, or two different values could concatenate to the same
bytes (delimiter injection). Every emission is ``tag + NUL + 8-byte big-endian
length + payload``; containers emit their element count and then self-delimiting
children; dict/set members are folded to fixed-length digests before being
combined. No variable-length content is ever used as its own delimiter.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import inspect
import pathlib
from typing import Any, Callable

from cachau.errors import UnhashableArgumentError

_LENGTH_BYTES = 8


def normalize_call(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[str, Any], ...]:
    """Bind a call to ``func``'s signature and return its canonical form."""
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.items())


def digest_arguments(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ignore: frozenset[str] = frozenset(),
) -> str:
    """Digest a normalized invocation of ``func`` into a stable hex string.

    ``ignore`` names top-level parameters excluded from identity — declared
    irrelevant by the user (loggers, progress callbacks), never silently.
    """
    hasher = hashlib.sha256()
    for name, value in normalize_call(func, args, kwargs):
        if name in ignore:
            continue
        _emit(hasher, b"arg", name.encode())
        try:
            _feed(hasher, value)
        except UnhashableArgumentError as exc:
            raise UnhashableArgumentError(
                f"argument {name!r} of {func.__qualname__}() cannot be hashed: "
                f"{exc}. Provide an explicit key= or exclude it with ignore=."
            ) from None
    return hasher.hexdigest()


def digest_custom_key(func: Callable[..., Any], key_value: Any) -> str:
    """Digest the result of a user-supplied ``key=`` callable."""
    hasher = hashlib.sha256()
    _emit(hasher, b"custom-key", b"")
    try:
        _feed(hasher, key_value)
    except UnhashableArgumentError as exc:
        raise UnhashableArgumentError(
            f"the key= result for {func.__qualname__}() cannot be hashed: {exc}"
        ) from None
    return hasher.hexdigest()


def _emit(hasher: Any, tag: bytes, payload: bytes) -> None:
    hasher.update(tag)
    hasher.update(b"\x00")
    hasher.update(len(payload).to_bytes(_LENGTH_BYTES, "big"))
    hasher.update(payload)


def _emit_count(hasher: Any, tag: bytes, count: int) -> None:
    hasher.update(tag)
    hasher.update(b"\x00")
    hasher.update(count.to_bytes(_LENGTH_BYTES, "big"))


def _digest_value(value: Any) -> bytes:
    hasher = hashlib.sha256()
    _feed(hasher, value)
    return hasher.digest()


def _type_identity(value: Any) -> bytes:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}".encode()


def _feed(hasher: Any, value: Any) -> None:
    if value is None:
        _emit(hasher, b"none", b"")
    elif isinstance(value, enum.Enum):
        _emit(hasher, b"enum", _type_identity(value) + b"." + value.name.encode())
    elif isinstance(value, bool):
        _emit(hasher, b"bool", b"\x01" if value else b"\x00")
    elif isinstance(value, int):
        _emit(hasher, b"int", str(value).encode())
    elif isinstance(value, float):
        _emit(hasher, b"float", repr(value).encode())
    elif isinstance(value, complex):
        _emit(hasher, b"complex", repr(value).encode())
    elif isinstance(value, str):
        _emit(hasher, b"str", value.encode())
    elif isinstance(value, (bytes, bytearray)):
        _emit(hasher, b"bytes", bytes(value))
    elif isinstance(value, pathlib.PureWindowsPath):
        _emit(hasher, b"path-win", str(value).encode())
    elif isinstance(value, pathlib.PurePath):
        _emit(hasher, b"path-posix", str(value).encode())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        _emit(hasher, b"dataclass", _type_identity(value))
        _emit_count(hasher, b"fields", len(fields))
        for field in fields:
            _emit(hasher, b"field", field.name.encode())
            _feed(hasher, getattr(value, field.name))
    elif isinstance(value, tuple):
        _emit_count(hasher, b"tuple", len(value))
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, list):
        _emit_count(hasher, b"list", len(value))
        for item in value:
            _feed(hasher, item)
    elif isinstance(value, dict):
        _emit_count(hasher, b"dict", len(value))
        # Fold each pair to fixed-length digests, sorted by key digest, so
        # insertion order is irrelevant and key/value boundaries are exact.
        pairs = sorted(
            (_digest_value(key), _digest_value(item)) for key, item in value.items()
        )
        for key_digest, item_digest in pairs:
            hasher.update(key_digest)
            hasher.update(item_digest)
    elif isinstance(value, frozenset):
        _emit_count(hasher, b"frozenset", len(value))
        for member_digest in sorted(_digest_value(member) for member in value):
            hasher.update(member_digest)
    elif isinstance(value, set):
        _emit_count(hasher, b"set", len(value))
        for member_digest in sorted(_digest_value(member) for member in value):
            hasher.update(member_digest)
    else:
        raise UnhashableArgumentError(
            f"type {type(value).__qualname__!r} has no cachau hashing support"
        )
