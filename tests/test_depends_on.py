"""depends_on=: external dependency invalidation (GUIDELINES.md §7, §8).

A declared dependency (file, env var, package version, or user token) that
changes since a result was committed makes the next call a MISS with reason
`dependency_changed`, drops the stale entry, and recomputes. The stored
fingerprint lives in the entry — never in the key — so the same key is reused
and the miss is attributable to the dependency, not to a key-not-found.
"""

import dataclasses
import importlib.metadata
import os

import cachau
import pytest

from cachau import cache
from cachau.dependencies import (
    EnvDependency,
    FileDependency,
    PackageDependency,
    changed_labels,
    fingerprint_dependencies,
    normalize_dependencies,
)
from cachau.errors import ConfigurationError
from cachau.memory import MemoryBackend


# --------------------------------------------------------------------------- #
# File dependencies
# --------------------------------------------------------------------------- #


def test_unchanged_file_is_a_hit(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b,c")
    calls = []

    @cache(depends_on=[str(data)])
    def load(n):
        calls.append(n)
        return n

    assert load(1) == 1
    assert load(1) == 1  # file unchanged → served from cache
    assert calls == [1]


def test_changed_file_recomputes(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b,c")
    calls = []

    @cache(depends_on=[str(data)])  # default: content hash
    def load(n):
        calls.append(n)
        return len(data.read_text())

    first = load(1)
    data.write_text("a,b,c,d,e,f,g")
    second = load(1)
    assert calls == [1, 1]  # recomputed once
    assert first == 5 and second == 13
    assert load.cache.stats().miss_dependency_changed == 1


def test_default_file_mode_is_content():
    assert FileDependency("x").on == "content"
    assert cachau.file("x").on == "content"


def test_content_default_catches_same_size_same_mtime_change(tmp_path):
    """The correctness win of the content default: a same-size replacement that
    preserves mtime (cp -p / rsync --times / coarse FS) still invalidates —
    where mtime+size would serve the stale result as a false HIT."""
    data = tmp_path / "data.bin"
    data.write_bytes(b"AAAAA")
    os.utime(data, (1000, 1000))
    calls = []

    @cache(depends_on=[str(data)])
    def load(n):
        calls.append(n)
        return data.read_bytes()

    assert load(1) == b"AAAAA"
    data.write_bytes(b"BBBBB")  # different content, SAME size
    os.utime(data, (1000, 1000))  # SAME mtime
    assert load(1) == b"BBBBB"  # content hash catches it — no false HIT
    assert calls == [1, 1]


def test_mtime_mode_is_opt_in(tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(b"AAAAA")
    os.utime(data, (1000, 1000))
    calls = []

    @cache(depends_on=[cachau.file(str(data), on="mtime")])
    def load(n):
        calls.append(n)
        return data.read_bytes()

    load(1)
    data.write_bytes(b"BBBBB")  # same size
    os.utime(data, (1000, 1000))  # same mtime → mtime heuristic cannot see it
    load(1)
    assert calls == [1]  # documented mtime limitation: served from cache


def test_content_mode_ignores_mtime_only_touches(tmp_path):
    data = tmp_path / "data.csv"
    data.write_bytes(b"payload")
    calls = []

    @cache(depends_on=[cachau.file(str(data), on="content")])
    def load(n):
        calls.append(n)
        return n

    load(1)
    # Rewrite identical bytes: content hash is unchanged, so still a HIT even
    # though the mtime almost certainly moved.
    data.write_bytes(b"payload")
    load(1)
    assert calls == [1]

    data.write_bytes(b"different")
    load(1)
    assert calls == [1, 1]


def test_missing_file_fingerprints_as_absent_then_appears(tmp_path):
    data = tmp_path / "later.csv"
    calls = []

    @cache(depends_on=[str(data)])
    def load(n):
        calls.append(n)
        return n

    load(1)  # file absent → committed against <absent>
    data.write_text("now here")
    load(1)  # file appeared → dependency changed → recompute
    assert calls == [1, 1]
    assert load.cache.stats().miss_dependency_changed == 1


# --------------------------------------------------------------------------- #
# Env, package, token dependencies
# --------------------------------------------------------------------------- #


def test_env_var_change_recomputes(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "fast")
    calls = []

    @cache(depends_on=[cachau.env("PIPELINE_MODE")])
    def run(n):
        calls.append(n)
        return n

    run(1)
    run(1)
    assert calls == [1]
    monkeypatch.setenv("PIPELINE_MODE", "slow")
    run(1)
    assert calls == [1, 1]


def test_env_unset_is_distinct_from_empty(monkeypatch):
    monkeypatch.delenv("MAYBE", raising=False)
    calls = []

    @cache(depends_on=[cachau.env("MAYBE")])
    def run(n):
        calls.append(n)
        return n

    run(1)  # unset
    monkeypatch.setenv("MAYBE", "")  # now set-but-empty: a real change
    run(1)
    assert calls == [1, 1]


def test_package_version_dependency_hits_when_stable():
    calls = []

    @cache(depends_on=[cachau.package("pytest")])
    def run(n):
        calls.append(n)
        return n

    run(1)
    run(1)  # pytest's version does not change mid-process
    assert calls == [1]


def test_token_callable_reevaluated_each_call():
    version = {"v": 1}
    calls = []

    @cache(depends_on=[cachau.token(lambda: version["v"], name="schema")])
    def run(n):
        calls.append(n)
        return n

    run(1)
    run(1)
    assert calls == [1]
    version["v"] = 2
    run(1)
    assert calls == [1, 1]


def test_token_literal_value():
    calls = []

    @cache(depends_on=[cachau.token("v3", name="build")])
    def run(n):
        calls.append(n)
        return n

    run(1)
    run(1)
    assert calls == [1]


# --------------------------------------------------------------------------- #
# Multiple dependencies, isolation
# --------------------------------------------------------------------------- #


def test_only_the_changed_dependency_triggers_recompute(monkeypatch, tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("x")
    monkeypatch.setenv("MODE", "a")
    calls = []

    @cache(depends_on=[str(data), cachau.env("MODE")])
    def run(n):
        calls.append(n)
        return n

    run(1)
    run(1)
    assert calls == [1]
    monkeypatch.setenv("MODE", "b")
    run(1)
    assert calls == [1, 1]


def test_dependencies_do_not_leak_across_arguments(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("x")

    @cache(depends_on=[str(data)])
    def run(n):
        return n

    run(1)
    run(2)
    assert run.cache.stats().entries == 2  # each argument still its own entry
    data.write_text("changed!!")
    run(1)  # invalidates only the 1-entry it re-reads
    assert run.cache.stats().miss_dependency_changed == 1


# --------------------------------------------------------------------------- #
# explain()
# --------------------------------------------------------------------------- #


def test_explain_reports_dependency_changed_and_which(monkeypatch, tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("x")
    monkeypatch.setenv("MODE", "a")

    @cache(depends_on=[str(data), cachau.env("MODE")])
    def run(n):
        return n

    run(1)
    assert run.cache.explain(1).outcome == "HIT"
    monkeypatch.setenv("MODE", "b")
    explanation = run.cache.explain(1)
    assert explanation.outcome == "MISS"
    assert explanation.reason == "dependency_changed"
    assert explanation.changed_dependencies == ("env:MODE",)
    assert "env:MODE" in str(explanation)


def test_explain_is_still_pure_with_dependencies(monkeypatch):
    monkeypatch.setenv("MODE", "a")

    @cache(depends_on=[cachau.env("MODE")])
    def run(n):
        return n

    run(1)
    before = run.cache.stats()
    monkeypatch.setenv("MODE", "b")
    run.cache.explain(1)  # observation must not recompute or recount
    assert run.cache.stats() == before


# --------------------------------------------------------------------------- #
# Robustness: mid-compute change, failing dependency sources, corruption
# --------------------------------------------------------------------------- #


def test_dependency_changed_during_compute_is_not_cached():
    """A dependency that moves while the function runs (and stays moved) must not
    be cached under the pre-compute fingerprint — that would be a false HIT."""
    state = {"dep": "A"}
    calls = []

    @cache(depends_on=[cachau.token(lambda: state["dep"], name="d")])
    def f(n):
        calls.append(n)
        state["dep"] = "B"  # changes during compute and stays changed
        return state["dep"]

    assert f(1) == "B"
    # Pre-compute fp was fp("A"), post-compute fp is fp("B") → not cached.
    assert f.cache.stats().dependency_race_skips == 1
    f(1)  # recompute; state is stable "B" now → this one caches
    assert f(1) == "B"  # served from cache
    assert calls == [1, 1]  # third call was a HIT


def test_package_metadata_error_degrades_to_absent(monkeypatch):
    """A non-PackageNotFoundError from importlib.metadata must not crash a call
    the cache could otherwise serve — degrade to the absent sentinel."""

    def boom(name):
        raise ValueError("corrupt METADATA")

    monkeypatch.setattr(importlib.metadata, "version", boom)

    @cache(depends_on=[cachau.package("whatever")])
    def g(n):
        return n * 2

    assert g(1) == 2  # does not raise
    assert g(1) == 2


def test_token_callable_exception_propagates():
    """A raising token callable surfaces at the call site (documented): cachau
    cannot verify freshness, so it must not swallow it into a false HIT."""
    state = {"fail": False}

    def tok():
        if state["fail"]:
            raise RuntimeError("source unreachable")
        return "v1"

    @cache(depends_on=[cachau.token(tok, name="s")])
    def compute(n):
        return n * 10

    assert compute(5) == 50  # cached
    state["fail"] = True
    with pytest.raises(RuntimeError):
        compute(5)  # even though the value is cached, the token cannot be checked


def test_corrupt_dependency_fingerprints_degrades_to_miss(tmp_path):
    """A persisted entry with a malformed dependency_fingerprints field is
    corruption: the read must degrade to a MISS, never raise into user code."""
    import json

    # Shared namespace + identical body → both decorations produce the same key,
    # so the second one reads the first's (now corrupted) persisted entry.
    @cache(persist=str(tmp_path), namespace="shared", depends_on=[cachau.token("v", name="x")])
    def load(n):
        return n * 3

    load(1)
    (path,) = list(tmp_path.glob("*.cachau"))
    raw = path.read_bytes()
    first_nl = raw.index(b"\n")
    second_nl = raw.index(b"\n", first_nl + 1)
    metadata = json.loads(raw[first_nl + 1 : second_nl])
    metadata["dependency_fingerprints"] = ["not", "a", "dict"]  # corrupt
    corrupted = (
        raw[: first_nl + 1] + json.dumps(metadata).encode() + raw[second_nl:]
    )
    path.write_bytes(corrupted)

    @cache(persist=str(tmp_path), namespace="shared", depends_on=[cachau.token("v", name="x")])
    def load2(n):
        return n * 3

    assert load2.cache.explain(1).reason in ("not_found", "dependency_changed")
    assert load2(1) == 3  # recomputes cleanly


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_dependency_fingerprint_survives_persistence(tmp_path):
    cache_dir = tmp_path / "cache"
    data = tmp_path / "d.csv"
    data.write_text("original")

    def build():
        @cache(persist=str(cache_dir), depends_on=[str(data)])
        def load(n):
            return data.read_text()

        return load

    load = build()
    assert load(1) == "original"

    # Simulate a process restart: fresh decoration over the same store.
    data.write_text("edited after restart")
    load2 = build()
    assert load2(1) == "edited after restart"  # dependency change detected on disk
    assert load2.cache.stats().miss_dependency_changed == 1


def test_entry_written_without_dependencies_is_not_trusted(tmp_path):
    """An entry committed before depends_on existed (no stored fingerprint) must
    not be trusted once the function declares a dependency: recompute."""
    backend = MemoryBackend()
    calls = []

    @cache(backend=backend, depends_on=[cachau.token("v", name="x")])
    def load(n):
        calls.append(n)
        return n

    load(1)
    # Simulate a legacy entry: strip the dependency fingerprint the way an
    # on-disk entry written before this feature would have (field absent → None).
    (key, entry) = next(iter(backend.iter_entries()))
    backend.set(key, dataclasses.replace(entry, dependency_fingerprints=None))

    load(1)  # stored None != current {token:x} → dependency changed → recompute
    assert calls == [1, 1]
    assert load.cache.stats().miss_dependency_changed == 1


# --------------------------------------------------------------------------- #
# Configuration validation (fail loud, at decoration)
# --------------------------------------------------------------------------- #


def test_bare_string_is_a_file_dependency():
    (labeled,) = normalize_dependencies(["data.csv"])
    assert isinstance(labeled.dependency, FileDependency)
    assert labeled.label == "file:data.csv"


def test_descriptors_pass_through():
    labeled = normalize_dependencies(
        [cachau.env("A"), cachau.package("pytest")]
    )
    assert labeled[0].label == "env:A"
    assert isinstance(labeled[0].dependency, EnvDependency)
    assert isinstance(labeled[1].dependency, PackageDependency)


def test_unnamed_tokens_get_positional_labels():
    labeled = normalize_dependencies([cachau.token(1), cachau.token(2)])
    assert [item.label for item in labeled] == ["token[0]", "token[1]"]


def test_duplicate_dependency_labels_fail():
    with pytest.raises(ConfigurationError):
        normalize_dependencies(["data.csv", "data.csv"])


def test_non_iterable_depends_on_fails():
    with pytest.raises(ConfigurationError):
        normalize_dependencies(42)


def test_bare_string_depends_on_is_rejected():
    # A single string is iterable; treating it as a list of characters would be
    # a nasty surprise, so require an actual list.
    with pytest.raises(ConfigurationError):
        normalize_dependencies("data.csv")


def test_bad_file_mode_fails():
    with pytest.raises(ConfigurationError):
        cachau.file("x", on="nonsense")


def test_unknown_descriptor_type_fails():
    with pytest.raises(ConfigurationError):
        normalize_dependencies([object()])


def test_unhashable_token_fails_loudly():
    @cache(depends_on=[cachau.token(lambda: object())])
    def run(n):
        return n

    with pytest.raises(ConfigurationError):
        run(1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_fingerprint_dependencies_none_when_empty():
    assert fingerprint_dependencies(()) is None


def test_changed_labels_diffs_both_directions():
    assert changed_labels({"a": "1"}, {"a": "2"}) == ("a",)
    assert changed_labels(None, {"a": "1"}) == ("a",)
    assert changed_labels({"a": "1"}, None) == ("a",)
    assert changed_labels({"a": "1"}, {"a": "1"}) == ()


def test_no_dependencies_behaves_like_before():
    calls = []

    @cache
    def run(n):
        calls.append(n)
        return n

    run(1)
    run(1)
    assert calls == [1]
    assert run.cache.stats().miss_dependency_changed == 0
