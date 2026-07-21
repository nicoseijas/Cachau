"""DiskBackend: persistent, atomic, corruption-safe entry storage.

File format (versioned explicitly, GUIDELINES.md §6) — one file per entry,
named ``sha256(key).cachau``:

    cachau-entry/<FORMAT_VERSION>\\n
    <metadata JSON: key, namespace, created_at, expires_at, size, serializer>\\n
    <payload bytes (pickle)>

Writes are atomic: serialize to a temp file in the same directory, flush and
fsync, then ``os.replace`` (atomic on POSIX and Windows). A reader never sees
a half-written entry. The guarantee is read-atomicity, not full crash
durability: the rename's directory update is fsynced on POSIX (best effort)
but a power loss immediately after commit may revert to the previous entry
state — which is a controlled MISS, never a corrupt read. Temp files from a
killed process are swept (best effort) on ``clear()``.

Any incompatibility — unknown version, corrupt metadata, undecodable payload,
truncation — degrades to a MISS: the file is removed (best effort) and ``get``
returns ``None``. Corruption is never a user-facing error.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import uuid
from typing import Any, Iterator

from cachau.backend import CacheEntry, EntryMetadata

FORMAT_VERSION = 1
_MAGIC_PREFIX = b"cachau-entry/"
_SUFFIX = ".cachau"
_SERIALIZER = "pickle"


class DiskBackend:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = pathlib.Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> pathlib.Path:
        return self._directory / (hashlib.sha256(key.encode()).hexdigest() + _SUFFIX)

    def get(self, key: str) -> CacheEntry | None:
        return self._load(key, remove_corrupt=True)

    def peek(self, key: str) -> CacheEntry | None:
        """Side-effect-free read: never removes corrupt files (observation)."""
        return self._load(key, remove_corrupt=False)

    def _load(self, key: str, *, remove_corrupt: bool) -> CacheEntry | None:
        path = self._path_for(key)
        loaded = _read_entry(path, remove_corrupt=remove_corrupt)
        if loaded is None:
            return None
        stored_key, entry = loaded
        if stored_key != key:  # hash-file collision or foreign file: never trust it
            return None
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        payload = pickle.dumps(entry.value, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "key": key,
            "namespace": entry.namespace,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "size": entry.size,
            "dependency_fingerprints": entry.dependency_fingerprints,
            "serializer": _SERIALIZER,
        }
        header = (
            _MAGIC_PREFIX
            + str(FORMAT_VERSION).encode()
            + b"\n"
            + json.dumps(metadata).encode()
            + b"\n"
        )
        final_path = self._path_for(key)
        temp_path = final_path.with_name(
            f"{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(temp_path, "wb") as handle:
                handle.write(header)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, final_path)
            self._fsync_directory()
        finally:
            temp_path.unlink(missing_ok=True)

    def _fsync_directory(self) -> None:
        # POSIX: make the rename itself durable. Windows cannot fsync a
        # directory handle this way; read-atomicity holds regardless.
        try:
            fd = os.open(self._directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def delete(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)

    def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            for stale_temp in self._directory.glob("*.tmp"):
                stale_temp.unlink(missing_ok=True)
        for path in self._directory.glob(f"*{_SUFFIX}"):
            if namespace is None:
                path.unlink(missing_ok=True)
                continue
            metadata = _read_metadata(path)
            if metadata is not None and metadata.get("namespace") == namespace:
                path.unlink(missing_ok=True)

    def iter_entries(self) -> Iterator[tuple[str, CacheEntry]]:
        for path in sorted(self._directory.glob(f"*{_SUFFIX}")):
            loaded = _read_entry(path, remove_corrupt=False)
            if loaded is not None:
                yield loaded

    def iter_metadata(self) -> Iterator[EntryMetadata]:
        """Yield per-entry metadata without deserializing payloads.

        Metadata-only decisions (stale-fingerprint purges, namespace clears,
        stats) must not pay for — or depend on — unpickling every stored value.
        """
        for path in sorted(self._directory.glob(f"*{_SUFFIX}")):
            metadata = _read_metadata(path)
            if metadata is not None:
                key = metadata.get("key")
                namespace = metadata.get("namespace")
                if isinstance(key, str) and isinstance(namespace, str):
                    # Unreadable numbers degrade to None rather than dropping
                    # the row: a metadata-only consumer (stale purges) must
                    # still see — and be able to reclaim — a damaged entry.
                    yield EntryMetadata(
                        key,
                        namespace,
                        _optional_number(metadata.get("size"), int),
                        _optional_number(metadata.get("created_at"), float),
                        _optional_number(metadata.get("expires_at"), float),
                        _lenient_dependencies(metadata.get("dependency_fingerprints")),
                    )


# Header lines are a magic string plus a small JSON object, so a single
# modest read covers them. The loop below still grows if a pathological key
# makes the header longer, and gives up rather than reading an entire large
# payload looking for a newline that corruption removed.
_HEADER_CHUNK = 8 * 1024
_HEADER_LIMIT = 1024 * 1024


def _parse_header(prefix: bytes) -> tuple[dict[str, Any], int]:
    """Parse the two header lines; return the metadata and the payload offset."""
    first_nl = prefix.index(b"\n")
    magic = prefix[:first_nl]
    if magic != _MAGIC_PREFIX + str(FORMAT_VERSION).encode():
        raise ValueError(f"unknown format: {magic!r}")
    second_nl = prefix.index(b"\n", first_nl + 1)
    metadata: dict[str, Any] = json.loads(prefix[first_nl + 1 : second_nl])
    return metadata, second_nl + 1


def _split_file(content: bytes) -> tuple[dict[str, Any], bytes]:
    """Split raw file bytes into (metadata, payload); raises on any mismatch."""
    metadata, offset = _parse_header(content)
    return metadata, content[offset:]


def _read_metadata(path: pathlib.Path) -> dict[str, Any] | None:
    """Read only the header, never the payload.

    This runs for every entry in the directory on every decoration of a
    persistent cache — that is, at import. Reading whole files here would make
    startup scale with the SIZE of what is cached instead of the NUMBER of
    entries, so a cache holding 1,000 x 1 MB results would move a gigabyte
    through memory before the program did any work of its own.
    """
    try:
        with path.open("rb") as handle:
            prefix = b""
            while prefix.count(b"\n") < 2 and len(prefix) < _HEADER_LIMIT:
                chunk = handle.read(_HEADER_CHUNK)
                if not chunk:
                    break
                prefix += chunk
        metadata, _ = _parse_header(prefix)
        return metadata
    except Exception:  # noqa: BLE001 - any corruption degrades to a controlled miss
        return None


def _checked_str(metadata: dict[str, Any], field: str) -> str:
    value = metadata[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} is not a string: {value!r}")
    return value


def _checked_timestamp(
    metadata: dict[str, Any], field: str, *, optional: bool
) -> float | None:
    value = metadata[field]
    if value is None and optional:
        return None
    # bool is a subclass of int: a JSON `true` here is corruption, not a time.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not a timestamp: {value!r}")
    return float(value)  # JSON has one number type; an int is a valid instant


def _checked_size(metadata: dict[str, Any], field: str) -> int | None:
    value = metadata[field]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} is not a byte count: {value!r}")
    return value


def _checked_dependencies(metadata: dict[str, Any]) -> dict[str, str] | None:
    """Parse the stored dependency fingerprints, or raise so the read misses.

    Absent (entries written before ``depends_on`` existed) degrades to ``None``
    — backward compatible, and a function that now declares dependencies then
    sees a mismatch and recomputes. A present-but-malformed value is corruption:
    raise so the whole read degrades to a controlled MISS, never a silent
    freshness check against garbage.
    """
    value = metadata.get("dependency_fingerprints")
    if value is None:
        return None
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError(f"dependency_fingerprints is not a str->str map: {value!r}")
    return value


def _lenient_dependencies(value: Any) -> dict[str, str] | None:
    """Coerce a metadata dependency map for observation, or None if malformed.

    The read-path validator (``_checked_dependencies``) RAISES on a bad map so a
    corrupt entry degrades to a MISS. A metadata scan (inspect/stats) must not
    raise over one damaged row, so here a malformed value simply reads as None.
    """
    if isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        return value
    return None


def _optional_number(value: Any, kind: type) -> Any:
    """Coerce a metadata number, or None if it is absent or not a number."""
    # bool is a subclass of int and never a legitimate size or timestamp.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if kind is int and not isinstance(value, int):
        return None
    return kind(value)


def _entry_from_metadata(
    metadata: dict[str, Any], value: Any
) -> tuple[str, CacheEntry]:
    """Build a validated entry, or raise so the caller degrades to a MISS.

    A header can be syntactically valid JSON and still be nonsense. Without
    this check the bad value survives the read and detonates later — a
    wrong-typed ``expires_at`` raises inside ``is_expired``, surfacing as an
    unrelated TypeError in user code. Validating here keeps corruption inside
    the one place that already knows how to answer it: a controlled MISS.
    """
    return _checked_str(metadata, "key"), CacheEntry(
        value=value,
        namespace=_checked_str(metadata, "namespace"),
        created_at=_checked_timestamp(metadata, "created_at", optional=False),
        expires_at=_checked_timestamp(metadata, "expires_at", optional=True),
        size=_checked_size(metadata, "size"),
        dependency_fingerprints=_checked_dependencies(metadata),
    )


def _read_entry(
    path: pathlib.Path, *, remove_corrupt: bool = True
) -> tuple[str, CacheEntry] | None:
    """Parse one entry file; on any incompatibility return ``None`` (a MISS)."""
    try:
        content = path.read_bytes()
    except OSError:
        return None
    try:
        metadata, payload = _split_file(content)
        value = pickle.loads(payload)
        return _entry_from_metadata(metadata, value)
    except Exception:  # noqa: BLE001 - any corruption degrades to a controlled miss
        if remove_corrupt:
            path.unlink(missing_ok=True)
        return None
