"""DiskBackend: atomic, versioned, corruption-safe storage (GUIDELINES.md §6)."""

import json

from cachau.backend import CacheEntry
from cachau.disk import FORMAT_VERSION, DiskBackend


def entry(value, namespace="ns", **kwargs):
    return CacheEntry(value=value, namespace=namespace, **kwargs)


def test_get_returns_none_for_missing_key(tmp_path):
    backend = DiskBackend(tmp_path)
    assert backend.get("missing") is None


def test_set_then_get_round_trip(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry({"a": [1, 2]}, created_at=5.0, expires_at=65.0, size=99))
    stored = backend.get("k")
    assert stored is not None
    assert stored.value == {"a": [1, 2]}
    assert stored.namespace == "ns"
    assert stored.created_at == 5.0
    assert stored.expires_at == 65.0
    assert stored.size == 99


def test_persists_across_backend_instances(tmp_path):
    DiskBackend(tmp_path).set("k", entry(42))
    fresh = DiskBackend(tmp_path)
    assert fresh.get("k").value == 42


def test_delete_removes_entry(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry(1))
    backend.delete("k")
    assert backend.get("k") is None
    backend.delete("k")  # deleting again is a no-op


def test_clear_all(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("a", entry(1))
    backend.set("b", entry(2))
    backend.clear()
    assert backend.get("a") is None
    assert backend.get("b") is None


def test_clear_by_namespace(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("a", entry(1, namespace="one"))
    backend.set("b", entry(2, namespace="two"))
    backend.clear(namespace="one")
    assert backend.get("a") is None
    assert backend.get("b").value == 2


def test_iter_entries(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("a", entry(1))
    backend.set("b", entry(2))
    listing = dict(backend.iter_entries())
    assert set(listing) == {"a", "b"}
    assert listing["a"].value == 1


def test_no_temp_files_left_behind(tmp_path):
    backend = DiskBackend(tmp_path)
    for i in range(20):
        backend.set(f"k{i}", entry(i))
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_truncated_file_degrades_to_miss_and_is_removed(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry(1))
    (path,) = list(tmp_path.glob("*.cachau"))
    path.write_bytes(path.read_bytes()[:10])
    assert backend.get("k") is None
    assert list(tmp_path.glob("*.cachau")) == []


def test_garbage_file_degrades_to_miss(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry(1))
    (path,) = list(tmp_path.glob("*.cachau"))
    path.write_bytes(b"\x00\xff garbage \x00" * 10)
    assert backend.get("k") is None


def test_unknown_format_version_degrades_to_miss(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry(1))
    (path,) = list(tmp_path.glob("*.cachau"))
    content = path.read_bytes()
    header_end = content.index(b"\n")
    path.write_bytes(b"cachau-entry/999" + content[header_end:])
    assert backend.get("k") is None


def test_corrupt_metadata_json_degrades_to_miss(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry(1))
    (path,) = list(tmp_path.glob("*.cachau"))
    content = path.read_bytes()
    first_nl = content.index(b"\n")
    second_nl = content.index(b"\n", first_nl + 1)
    corrupted = content[: first_nl + 1] + b"{not json]\n" + content[second_nl + 1 :]
    path.write_bytes(corrupted)
    assert backend.get("k") is None


def test_corrupt_payload_degrades_to_miss(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry([1, 2, 3]))
    (path,) = list(tmp_path.glob("*.cachau"))
    content = path.read_bytes()
    first_nl = content.index(b"\n")
    second_nl = content.index(b"\n", first_nl + 1)
    path.write_bytes(content[: second_nl + 1] + b"\x00\x01\x02 not a pickle")
    assert backend.get("k") is None


def test_iter_entries_skips_corrupt_files(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("good", entry(1))
    backend.set("bad", entry(2))
    files = sorted(tmp_path.glob("*.cachau"))
    files[0].write_bytes(b"garbage")
    listing = dict(backend.iter_entries())
    assert len(listing) == 1


def test_format_version_recorded_in_header(tmp_path):
    backend = DiskBackend(tmp_path)
    backend.set("k", entry(1))
    (path,) = list(tmp_path.glob("*.cachau"))
    content = path.read_bytes()
    first_nl = content.index(b"\n")
    assert content[:first_nl] == f"cachau-entry/{FORMAT_VERSION}".encode()
    second_nl = content.index(b"\n", first_nl + 1)
    meta = json.loads(content[first_nl + 1 : second_nl])
    assert meta["key"] == "k"
    assert meta["serializer"] == "pickle"
