"""DiskBackend: persistent, atomic, corruption-safe entry storage.

File format (versioned explicitly, GUIDELINES.md §6) — one file per entry,
named ``sha256(key).cachau``:

    cachau-entry/<FORMAT_VERSION>\\n
    <metadata JSON: key, namespace, created_at, expires_at, size, serializer>\\n
    <payload bytes (pickle)>

Writes are atomic: serialize to a temp file in the same directory, flush and
fsync, then ``os.replace`` (atomic on POSIX and Windows). A reader never sees
a half-written entry.

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

from cachau.backend import CacheEntry

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
        path = self._path_for(key)
        loaded = _read_entry(path)
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
        finally:
            temp_path.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)

    def clear(self, namespace: str | None = None) -> None:
        for path in self._directory.glob(f"*{_SUFFIX}"):
            if namespace is None:
                path.unlink(missing_ok=True)
                continue
            loaded = _read_entry(path, remove_corrupt=False)
            if loaded is not None and loaded[1].namespace == namespace:
                path.unlink(missing_ok=True)

    def iter_entries(self) -> Iterator[tuple[str, CacheEntry]]:
        for path in sorted(self._directory.glob(f"*{_SUFFIX}")):
            loaded = _read_entry(path, remove_corrupt=False)
            if loaded is not None:
                yield loaded


def _read_entry(
    path: pathlib.Path, *, remove_corrupt: bool = True
) -> tuple[str, CacheEntry] | None:
    """Parse one entry file; on any incompatibility return ``None`` (a MISS)."""
    try:
        content = path.read_bytes()
    except OSError:
        return None
    try:
        first_nl = content.index(b"\n")
        magic = content[:first_nl]
        if magic != _MAGIC_PREFIX + str(FORMAT_VERSION).encode():
            raise ValueError(f"unknown format: {magic!r}")
        second_nl = content.index(b"\n", first_nl + 1)
        metadata: dict[str, Any] = json.loads(content[first_nl + 1 : second_nl])
        value = pickle.loads(content[second_nl + 1 :])
        entry = CacheEntry(
            value=value,
            namespace=metadata["namespace"],
            created_at=metadata["created_at"],
            expires_at=metadata["expires_at"],
            size=metadata["size"],
        )
        return metadata["key"], entry
    except Exception:  # noqa: BLE001 - any corruption degrades to a controlled miss
        if remove_corrupt:
            path.unlink(missing_ok=True)
        return None
