"""End-to-end persistence semantics (GUIDELINES.md §6, §9)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cachau import cache
from cachau.errors import ConfigurationError


SOURCE = """
from cachau import cache

@cache(persist=PERSIST_DIR, namespace="shared.expensive")
def expensive(x):
    log.append(x)
    return x * 2
"""


def define(persist_dir, log):
    scope = {"PERSIST_DIR": persist_dir, "log": log}
    exec(SOURCE, scope)
    return scope["expensive"]


def test_survives_interpreter_restart(tmp_path):
    """Same source re-executed (fresh decorator, fresh backend) hits the disk."""
    log = []
    first = define(str(tmp_path), log)
    assert first(3) == 6
    assert log == [3]

    second = define(str(tmp_path), log)  # simulated restart: same code, new everything
    assert second(3) == 6
    assert log == [3]  # served from disk, not recomputed


def test_persists_across_real_processes(tmp_path):
    """Mandatory GUIDELINES test: a separate Python process sees the entry."""
    import os

    script = textwrap.dedent(
        f"""
        from cachau import cache

        @cache(persist={str(tmp_path)!r}, namespace="xproc.expensive")
        def expensive(x):
            print("COMPUTED")
            return x * 2

        print("RESULT", expensive(3))
        """
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}

    def run():
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )

    first = run()
    assert first.returncode == 0, first.stderr
    assert "COMPUTED" in first.stdout
    assert "RESULT 6" in first.stdout

    second = run()
    assert second.returncode == 0, second.stderr
    assert "COMPUTED" not in second.stdout  # served from disk
    assert "RESULT 6" in second.stdout


def test_persist_true_uses_default_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @cache(persist=True)
    def expensive(x):
        return x + 1

    assert expensive(1) == 2
    default_dir = tmp_path / ".cachau"
    assert default_dir.is_dir()
    assert list(default_dir.glob("*.cachau"))


def test_code_change_invalidates_persisted_results(tmp_path):
    scope_v1, scope_v2 = {}, {}
    template = """
from cachau import cache

@cache(persist={d!r}, namespace="ver.expensive")
def expensive(x):
    return x * {factor}
"""
    exec(template.format(d=str(tmp_path), factor=2), scope_v1)
    assert scope_v1["expensive"](5) == 10

    exec(template.format(d=str(tmp_path), factor=3), scope_v2)
    assert scope_v2["expensive"](5) == 15  # stale x*2 result must not be reused


def test_serialization_failure_returns_result_and_is_counted(tmp_path):
    calls = []

    @cache(persist=str(tmp_path))
    def expensive(x):
        calls.append(x)
        return lambda: x  # lambdas cannot be pickled

    result = expensive(1)
    assert callable(result) and result() == 1  # correct result despite failure
    assert expensive.cache.write_errors == 1
    expensive(1)  # never cached: recomputed, again counted
    assert calls == [1, 1]
    assert expensive.cache.write_errors == 2


def test_set_failure_keeps_budget_consistent(tmp_path):
    """After a failed write the budget must not track a phantom entry."""
    calls = []

    @cache(persist=str(tmp_path), max_memory=1000, size_of=lambda v: 900)
    def expensive(x):
        calls.append(x)
        return (lambda: x) if x == 1 else x  # x=1 fails serialization

    expensive(1)  # write fails; budget must not keep 900 phantom bytes
    expensive(2)  # fits only if the phantom was released
    expensive(2)
    assert calls == [1, 2]
    assert expensive.cache.evictions == 0


def test_persist_and_backend_are_mutually_exclusive(tmp_path):
    from cachau.memory import MemoryBackend

    with pytest.raises(ConfigurationError):

        @cache(persist=str(tmp_path), backend=MemoryBackend())
        def expensive(x):
            return x


def test_ttl_applies_to_persisted_entries(tmp_path):
    class FakeClock:
        now = 1000.0

        def __call__(self):
            return self.now

    clock = FakeClock()
    calls = []

    @cache(persist=str(tmp_path), ttl=60, clock=clock)
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    clock.now = 1061.0
    expensive(1)
    assert calls == [1, 1]


def test_delete_failure_on_expiry_never_blocks_the_call(tmp_path, monkeypatch):
    class FakeClock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = FakeClock()
    from cachau.disk import DiskBackend

    backend = DiskBackend(tmp_path)

    @cache(ttl=60, clock=clock, backend=backend)
    def expensive(x):
        return x * 2

    expensive(1)
    clock.now = 61.0

    def failing_delete(key):
        raise OSError("file locked by antivirus")

    monkeypatch.setattr(backend, "delete", failing_delete)
    assert expensive(1) == 2  # expired + undeletable: still recomputes and returns
    assert expensive.cache.delete_errors == 1


def test_delete_failure_on_eviction_never_loses_the_result(tmp_path, monkeypatch):
    from cachau.disk import DiskBackend

    backend = DiskBackend(tmp_path)

    @cache(max_memory=100, size_of=lambda v: 60, backend=backend)
    def expensive(x):
        return x * 2

    expensive(1)

    def failing_delete(key):
        raise OSError("file locked")

    monkeypatch.setattr(backend, "delete", failing_delete)
    assert expensive(2) == 4  # eviction delete fails: computed value survives
    assert expensive.cache.evictions == 1
    assert expensive.cache.delete_errors == 1


def test_stale_purge_does_not_deserialize_payloads(tmp_path):
    """Decoration-time purge is metadata-only: an unloadable payload in the
    same namespace must neither crash decoration nor survive the purge."""
    from cachau.backend import CacheEntry
    from cachau.disk import DiskBackend

    backend = DiskBackend(tmp_path)
    backend.set(
        "ver.expensive:oldfingerprint:digest",
        CacheEntry(value=1, namespace="ver.expensive"),
    )
    (path,) = list(tmp_path.glob("*.cachau"))
    content = path.read_bytes()
    second_nl = content.index(b"\n", content.index(b"\n") + 1)
    path.write_bytes(content[: second_nl + 1] + b"\x00 undecodable payload")

    @cache(persist=str(tmp_path), namespace="ver.expensive")
    def expensive(x):
        return x

    assert expensive(1) == 1
    remaining = [
        p for p in tmp_path.glob("*.cachau") if b"oldfingerprint" in p.read_bytes()
    ]
    assert remaining == []


def test_clear_sweeps_stale_temp_files(tmp_path):
    from cachau.disk import DiskBackend

    backend = DiskBackend(tmp_path)
    (tmp_path / "orphan.12345.deadbeef.tmp").write_bytes(b"partial write")
    backend.clear()
    assert list(tmp_path.glob("*.tmp")) == []


def test_clear_removes_persisted_entries(tmp_path):
    @cache(persist=str(tmp_path))
    def expensive(x):
        return x

    expensive(1)
    assert list(tmp_path.glob("*.cachau"))
    expensive.cache.clear()
    assert not list(tmp_path.glob("*.cachau"))


def test_wrong_typed_metadata_recomputes_instead_of_raising(tmp_path):
    """Corruption degrades to a miss, never a mysterious error (issue #12)."""
    import json

    @cache(persist=str(tmp_path), namespace="corrupt.expensive")
    def expensive(x):
        return x * 2

    assert expensive(3) == 6
    (path,) = list(tmp_path.glob("*.cachau"))
    content = path.read_bytes()
    first_nl = content.index(b"\n")
    second_nl = content.index(b"\n", first_nl + 1)
    metadata = json.loads(content[first_nl + 1 : second_nl])
    metadata["expires_at"] = "bad"
    path.write_bytes(
        content[: first_nl + 1] + json.dumps(metadata).encode() + content[second_nl:]
    )

    assert expensive(3) == 6  # recomputed, no TypeError
    assert expensive.cache.stats().misses == 2


def test_decoration_reads_each_entry_header_once(tmp_path, monkeypatch):
    """Purge and budget rehydration must share one pass over the store.

    Decoration happens at import; every extra pass is another round of file
    opens over the whole cache directory.
    """
    import cachau.disk as disk

    original = disk._read_metadata
    reads = []

    def counting(path):
        reads.append(path)
        return original(path)

    @cache(persist=str(tmp_path), max_memory="10MB", namespace="scan.expensive")
    def expensive(x):
        return x

    for i in range(20):
        expensive(i)

    monkeypatch.setattr(disk, "_read_metadata", counting)

    @cache(persist=str(tmp_path), max_memory="10MB", namespace="scan.expensive")
    def redecorated(x):
        return x

    assert len(reads) == 20
    assert len(set(reads)) == 20


class _CountingHandle:
    """File handle proxy that records how many bytes are actually delivered."""

    def __init__(self, handle, tally):
        self._handle = handle
        self._tally = tally

    def read(self, *args):
        data = self._handle.read(*args)
        self._tally.append(len(data))
        return data

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


def test_decoration_does_not_read_payload_bytes(tmp_path, monkeypatch):
    """The metadata pass must read headers, not bodies.

    Decoration scans the whole cache directory, and it happens at import. If
    that scan reads each file in full, its cost scales with the SIZE of what
    is cached rather than the NUMBER of entries: 1,000 x 1 MB entries would
    pull 1 GB off disk before the program does any work of its own. Only the
    two header lines are needed to purge and to rebuild the LRU budget.
    """
    import pathlib

    payload_bytes = 1_000_000
    entries = 5

    @cache(persist=str(tmp_path), namespace="payload.expensive")
    def expensive(x):
        return b"x" * payload_bytes

    for i in range(entries):
        expensive(i)

    delivered: list[int] = []
    original_open = pathlib.Path.open
    original_read_bytes = pathlib.Path.read_bytes

    def counting_open(self, mode="r", *args, **kwargs):
        handle = original_open(self, mode, *args, **kwargs)
        if self.suffix == ".cachau":
            return _CountingHandle(handle, delivered)
        return handle

    def counting_read_bytes(self, *args, **kwargs):
        data = original_read_bytes(self, *args, **kwargs)
        if self.suffix == ".cachau":
            delivered.append(len(data))
        return data

    monkeypatch.setattr(pathlib.Path, "open", counting_open)
    monkeypatch.setattr(pathlib.Path, "read_bytes", counting_read_bytes)

    @cache(persist=str(tmp_path), max_memory="10MB", namespace="payload.expensive")
    def redecorated(x):
        return b""

    total = sum(delivered)
    assert total < 64 * 1024, (
        f"decoration read {total:,} bytes for {entries} entries whose payloads "
        f"total {entries * payload_bytes:,} — it is reading the bodies"
    )
