"""Explicit keying: key= and ignore= (GUIDELINES.md §2)."""

import pytest

from cachau import cache
from cachau.errors import ConfigurationError, UnhashableArgumentError


def test_ignore_excludes_arguments_from_identity():
    calls = []

    @cache(ignore=["logger", "progress_callback"])
    def run(data, logger=None, progress_callback=None):
        calls.append(data)
        return data * 2

    assert run(1, logger="a") == 2
    assert run(1, logger="b", progress_callback=print) == 2  # same entry
    assert run(2) == 4
    assert calls == [1, 2]


def test_ignored_arguments_may_be_unhashable():
    """The whole point of ignore=: opaque collaborators don't need hashing."""

    class OpaqueLogger:
        pass

    calls = []

    @cache(ignore=["logger"])
    def run(data, logger=None):
        calls.append(data)
        return data

    assert run(1, logger=OpaqueLogger()) == 1
    assert run(1, logger=OpaqueLogger()) == 1
    assert calls == [1]


def test_ignore_unknown_parameter_fails_at_decoration_time():
    with pytest.raises(ConfigurationError):

        @cache(ignore=["nonexistent"])
        def run(data):
            return data


def test_ignore_all_parameters_still_caches_by_function():
    calls = []

    @cache(ignore=["x"])
    def run(x):
        calls.append(x)
        return "fixed"

    assert run(1) == "fixed"
    assert run(2) == "fixed"  # deliberately shared: x was declared irrelevant
    assert calls == [1]


def test_explicit_key_replaces_argument_hashing():
    calls = []

    @cache(key=lambda dataset, version: version)
    def process(dataset, version):
        calls.append(version)
        return len(dataset)

    assert process([1, 2, 3], "v1") == 3
    assert process([9, 9, 9, 9], "v1") == 3  # same version: same entry (declared)
    assert process([1], "v2") == 1
    assert calls == ["v1", "v2"]


def test_explicit_key_still_invalidates_on_code_change(tmp_path):
    template = """
from cachau import cache

@cache(persist={d!r}, namespace="keyed.process", key=lambda x, version: version)
def process(x, version):
    return x * {factor}
"""
    scope_v1, scope_v2 = {}, {}
    exec(template.format(d=str(tmp_path), factor=2), scope_v1)
    assert scope_v1["process"](5, "v1") == 10
    exec(template.format(d=str(tmp_path), factor=3), scope_v2)
    assert scope_v2["process"](5, "v1") == 15  # fingerprint still in the key


def test_unhashable_custom_key_fails_loudly():
    class Opaque:
        pass

    @cache(key=lambda x: Opaque())
    def run(x):
        return x

    with pytest.raises(UnhashableArgumentError):
        run(1)


def test_key_and_ignore_are_mutually_exclusive():
    with pytest.raises(ConfigurationError):

        @cache(key=lambda x: x, ignore=["x"])
        def run(x):
            return x


def test_explain_and_invalidate_use_the_custom_key():
    @cache(key=lambda dataset, version: version)
    def process(dataset, version):
        return len(dataset)

    process([1, 2], "v1")
    assert process.cache.explain([9, 9, 9], "v1").outcome == "HIT"
    process.cache.invalidate([], "v1")
    assert process.cache.explain([1, 2], "v1").outcome == "MISS"
