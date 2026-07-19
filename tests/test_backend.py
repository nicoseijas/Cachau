"""MemoryBackend honors the minimal CacheBackend contract (GUIDELINES.md §13)."""

from cachau.backend import CacheEntry
from cachau.memory import MemoryBackend


def entry(value, namespace="ns"):
    return CacheEntry(value=value, namespace=namespace)


def test_get_returns_none_for_missing_key():
    backend = MemoryBackend()
    assert backend.get("missing") is None


def test_set_then_get_round_trip():
    backend = MemoryBackend()
    backend.set("k", entry(42))
    stored = backend.get("k")
    assert stored is not None
    assert stored.value == 42


def test_delete_removes_entry():
    backend = MemoryBackend()
    backend.set("k", entry(1))
    backend.delete("k")
    assert backend.get("k") is None


def test_delete_missing_key_is_noop():
    backend = MemoryBackend()
    backend.delete("missing")


def test_clear_removes_everything():
    backend = MemoryBackend()
    backend.set("a", entry(1))
    backend.set("b", entry(2))
    backend.clear()
    assert backend.get("a") is None
    assert backend.get("b") is None


def test_clear_by_namespace_only_touches_that_namespace():
    backend = MemoryBackend()
    backend.set("a", entry(1, namespace="one"))
    backend.set("b", entry(2, namespace="two"))
    backend.clear(namespace="one")
    assert backend.get("a") is None
    assert backend.get("b") is not None


def test_iter_entries_yields_key_entry_pairs():
    backend = MemoryBackend()
    backend.set("a", entry(1))
    backend.set("b", entry(2))
    assert dict(backend.iter_entries()) == {
        "a": backend.get("a"),
        "b": backend.get("b"),
    }
