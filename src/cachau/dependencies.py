"""External dependency fingerprinting (GUIDELINES.md §7, §13).

A cached result can depend on more than its arguments and the function's own
code: a file on disk, an environment variable, an installed package version, or
any user-defined token. ``depends_on=[...]`` declares those, and cachau folds
their current fingerprints into each stored entry. On the next call the
fingerprints are recomputed and compared; a change makes the entry a MISS with
reason ``dependency_changed`` — the stale entry is dropped and the function runs
again.

The fingerprints live in the ENTRY, never in the cache key. A changed
dependency must overwrite the SAME key rather than orphan a stale entry under an
old one, and the resulting miss must be attributable to the dependency, not to a
key that was not found. Fingerprinting is pure observation — ``explain()``
computes it without side effects — with the single exception the user owns:
``token(callable)`` runs their callable.

Missing resources fingerprint to an explicit ``<absent>``/``<unset>`` sentinel
rather than raising, so a file appearing or disappearing, or a package being
installed or removed, invalidates conservatively instead of crashing a call.

Stability contract: a declared dependency must be stable for the duration of a
single call. Cachau samples each dependency's fingerprint before AND after the
function runs and refuses to cache a result computed across a detected change
(counted as ``dependency_race_skips``). But it observes only those two instants
at the call boundary — it cannot see reads the function makes internally, so a
dependency that changes and then reverts within one call (A→B→A) is
indistinguishable from one that never moved. If a dependency can change while a
cached function runs, treat it as an argument (so it enters the key) rather than
relying on ``depends_on`` alone.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Protocol, runtime_checkable

from cachau.errors import ConfigurationError, UnhashableArgumentError
from cachau.fingerprint import (
    function_fingerprint,
    function_namespace,
    unwrap_function,
)
from cachau.keys import _digest_value

_ABSENT = "<absent>"
_UNSET = "<unset>"
# Content hashes are truncated: 128 bits is far past collision risk for change
# detection, and keeping the digest short keeps the entry header small (it is
# re-read, header-only, on every access of a persistent entry).
_CONTENT_DIGEST_CHARS = 32


@runtime_checkable
class Dependency(Protocol):
    """An external input whose current state can be fingerprinted."""

    def label(self) -> str | None:
        """Stable identity, or ``None`` to accept a positional fallback."""
        ...

    def fingerprint(self) -> str:
        """The dependency's current state as a short, comparable string."""
        ...


@dataclass(frozen=True)
class FileDependency:
    """A file whose change should invalidate results.

    ``on="content"`` (default) reads and hashes the bytes: correctness first,
    so a file replaced with different content of the same size — or copied over
    with a preserved modification time (``cp -p``, ``rsync --times``, a coarse
    filesystem) — still invalidates. ``on="mtime"`` fingerprints modification
    time plus size instead: cheap (no read), but a heuristic — it MISSES a
    same-size change that keeps the same mtime, which would serve a stale result
    as fresh. Reserve ``mtime`` for large files where hashing every lookup is
    the bottleneck and mtime can be trusted.

    Content hashing reads the file without locking it: if another process
    rewrites it non-atomically during the read, the hash reflects a torn view —
    which matches neither the old nor the new content, so the result is a
    conservative recompute (never a false HIT), not a correctness problem.
    """

    path: os.PathLike[str] | str
    on: str = "content"

    def __post_init__(self) -> None:
        if self.on not in ("mtime", "content"):
            raise ConfigurationError(
                f"file dependency on= must be 'mtime' or 'content', got {self.on!r}"
            )

    def label(self) -> str:
        return f"file:{os.fspath(self.path)}"

    def fingerprint(self) -> str:
        path = os.fspath(self.path)
        if self.on == "content":
            try:
                with open(path, "rb") as handle:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()[:_CONTENT_DIGEST_CHARS]
            except OSError:
                return _ABSENT
        try:
            stat = os.stat(path)
        except OSError:
            return _ABSENT
        return f"{stat.st_mtime_ns}:{stat.st_size}"


@dataclass(frozen=True)
class EnvDependency:
    """An environment variable whose value should invalidate results.

    An unset variable is distinct from an empty one: ``<unset>`` versus ``v:``.
    """

    name: str

    def label(self) -> str:
        return f"env:{self.name}"

    def fingerprint(self) -> str:
        value = os.environ.get(self.name)
        return _UNSET if value is None else f"v:{value}"


@dataclass(frozen=True)
class PackageDependency:
    """An installed distribution whose version should invalidate results.

    ``name`` is the distribution name as it appears on PyPI (``"scikit-learn"``,
    not the import name ``sklearn``). A package that is not installed
    fingerprints to ``<absent>``, so installing or removing it invalidates.
    """

    name: str

    def label(self) -> str:
        return f"package:{self.name}"

    def fingerprint(self) -> str:
        try:
            return importlib.metadata.version(self.name)
        except Exception:  # noqa: BLE001
            # Not installed (PackageNotFoundError) OR unreadable metadata
            # (corrupt/partial dist-info raising a parse/zip error). Either way
            # this runs on every call, including a would-be HIT: degrade to the
            # absent sentinel like FileDependency does with OSError, never crash
            # a call the cache could otherwise serve (GUIDELINES.md §9, §11).
            return _ABSENT


@dataclass(frozen=True)
class TokenDependency:
    """A user-supplied value or zero-argument callable.

    A plain value is hashed as given; a callable is evaluated on every check and
    its result hashed — the escape hatch for anything cachau cannot observe on
    its own (a schema version, a remote ETag, a git SHA). Give it a ``name`` so
    persisted entries can match it across restarts; without one it is labelled
    by its position in ``depends_on``.

    Unlike file/env/package dependencies, a token has no "absent" state: if the
    callable RAISES, cachau cannot determine the dependency's value, so it cannot
    prove a cached result is fresh — the exception propagates to the caller
    rather than being swallowed (swallowing it to a sentinel could serve a stale
    result as fresh). A callable that can fail transiently should guard itself
    and return a stable value, or you accept that its failure surfaces at the
    call site even when the result was already cached.
    """

    value: Any
    name: str | None = None

    def label(self) -> str | None:
        return f"token:{self.name}" if self.name is not None else None

    def fingerprint(self) -> str:
        resolved = self.value() if callable(self.value) else self.value
        try:
            return _digest_value(resolved).hex()[:_CONTENT_DIGEST_CHARS]
        except UnhashableArgumentError as exc:
            raise ConfigurationError(
                f"token dependency value cannot be hashed: {exc}. Return a "
                f"hashable value (str, int, bytes, ...) from the token."
            ) from None


@dataclass(frozen=True)
class CodeDependency:
    """A callable whose implementation should invalidate results.

    Closes the transitive-code gap: the function fingerprint covers only the
    cached function's own code (plus closures), so a module-level helper called
    by global lookup can change without invalidating — a false HIT after an
    edit-and-restart. Declaring ``depends_on=[cachau.code(helper)]`` folds the
    helper's implementation fingerprint (bytecode, constants, closure captures —
    the same digest used for the cached function itself) into every entry, so
    editing the helper misses with ``dependency_changed``.

    The descriptor holds the function OBJECT and is validated at declaration:
    non-callables and callables without a code object (builtins, C extensions)
    fail loudly here, not on the first call. A cachau-cached function (or any
    ``functools.wraps`` wrapper) is unwrapped to the user's implementation.
    Rebinding the global NAME to a different function within one process is not
    observed — the same contract as a closure capture; the edit-and-restart
    scenario is the one this exists for.
    """

    func: Callable[..., Any]
    name: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise ConfigurationError(
                f"cachau.code() takes a function, got {self.func!r}"
            )
        unwrapped = unwrap_function(self.func)
        object.__setattr__(self, "func", unwrapped)
        function_fingerprint(unwrapped)  # unfingerprintable? fail at declaration

    def label(self) -> str:
        if self.name is not None:
            return f"code:{self.name}"
        return f"code:{function_namespace(self.func)}"

    def fingerprint(self) -> str:
        return function_fingerprint(self.func)[:_CONTENT_DIGEST_CHARS]


def file(path: os.PathLike[str] | str, *, on: str = "content") -> FileDependency:
    """Declare a file dependency; see :class:`FileDependency`."""
    return FileDependency(path, on)


def env(name: str) -> EnvDependency:
    """Declare an environment-variable dependency; see :class:`EnvDependency`."""
    return EnvDependency(name)


def package(name: str) -> PackageDependency:
    """Declare an installed-package dependency; see :class:`PackageDependency`."""
    return PackageDependency(name)


def token(value: Any, *, name: str | None = None) -> TokenDependency:
    """Declare a user-defined dependency token; see :class:`TokenDependency`."""
    return TokenDependency(value, name)


def code(func: Callable[..., Any], *, name: str | None = None) -> CodeDependency:
    """Declare a helper-implementation dependency; see :class:`CodeDependency`."""
    return CodeDependency(func, name)


class LabeledDependency(NamedTuple):
    """A dependency paired with the stable label its fingerprint is stored under."""

    label: str
    dependency: Dependency


def normalize_dependencies(
    depends_on: object | None,
) -> tuple[LabeledDependency, ...]:
    """Validate and label a ``depends_on`` declaration at decoration time.

    A bare string or ``os.PathLike`` is a file dependency (matching the
    documented ``depends_on=["data.csv"]`` shorthand); a descriptor from
    ``cachau.file/env/package/token`` is used as-is. Anything else fails loudly
    here, at decoration, rather than on the first call. Labels must be unique so
    each dependency occupies its own slot in the stored fingerprint map;
    unnamed tokens are labelled by position.
    """
    if depends_on is None:
        return ()
    if isinstance(depends_on, (str, bytes)) or not _is_iterable(depends_on):
        raise ConfigurationError(
            "depends_on= must be a list of dependencies (file paths or "
            f"cachau.file/env/package/token descriptors), got {depends_on!r}"
        )
    labeled: list[LabeledDependency] = []
    seen: set[str] = set()
    for index, spec in enumerate(depends_on):
        dependency = _coerce(spec)
        label = dependency.label() or f"token[{index}]"
        if label in seen:
            raise ConfigurationError(
                f"duplicate dependency {label!r} in depends_on=: give each one a "
                f"distinct path/name (tokens accept name=)"
            )
        seen.add(label)
        labeled.append(LabeledDependency(label, dependency))
    return tuple(labeled)


def fingerprint_dependencies(
    labeled: tuple[LabeledDependency, ...],
) -> dict[str, str] | None:
    """Fingerprint every declared dependency, or ``None`` when none are declared.

    ``None`` (no dependencies) is deliberately distinct from ``{}`` so a stored
    entry can record "this function had no dependencies" unambiguously.
    """
    if not labeled:
        return None
    return {item.label: item.dependency.fingerprint() for item in labeled}


def changed_labels(
    stored: dict[str, str] | None, current: dict[str, str] | None
) -> tuple[str, ...]:
    """The labels whose fingerprint differs between a stored entry and now."""
    stored = stored or {}
    current = current or {}
    return tuple(
        sorted(
            label
            for label in stored.keys() | current.keys()
            if stored.get(label) != current.get(label)
        )
    )


def _coerce(spec: object) -> Dependency:
    if isinstance(spec, Dependency) and not isinstance(spec, (str, bytes)):
        return spec
    if isinstance(spec, (str, os.PathLike)):
        return FileDependency(spec)
    raise ConfigurationError(
        f"depends_on= entries must be a file path or a cachau.file/env/package/"
        f"token descriptor, got {spec!r}"
    )


def _is_iterable(value: object) -> bool:
    try:
        iter(value)  # type: ignore[call-overload]
    except TypeError:
        return False
    return True
