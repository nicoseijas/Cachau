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


def test_clear_removes_persisted_entries(tmp_path):
    @cache(persist=str(tmp_path))
    def expensive(x):
        return x

    expensive(1)
    assert list(tmp_path.glob("*.cachau"))
    expensive.cache.clear()
    assert not list(tmp_path.glob("*.cachau"))
