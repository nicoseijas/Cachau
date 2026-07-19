"""End-to-end decorator semantics (GUIDELINES.md §16, Phase 0 subset)."""

import pytest

from cachau import cache
from cachau.errors import UnhashableArgumentError


def test_hit_after_first_computation():
    calls = []

    @cache
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(3) == 6
    assert expensive(3) == 6
    assert calls == [3]


def test_miss_on_different_arguments():
    calls = []

    @cache
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(1) == 2
    assert expensive(2) == 4
    assert calls == [1, 2]


def test_kwargs_and_positional_share_an_entry():
    calls = []

    @cache
    def expensive(x, y=10):
        calls.append((x, y))
        return x + y

    assert expensive(1, 2) == 3
    assert expensive(x=1, y=2) == 3
    assert expensive(1, y=2) == 3
    assert calls == [(1, 2)]


def test_decorator_with_parentheses():
    @cache()
    def expensive(x):
        return x + 1

    assert expensive(1) == 2


def test_different_functions_never_collide():
    @cache
    def double(x):
        return x * 2

    @cache
    def identity(x):
        return x

    assert double(2) == 4
    assert identity(2) == 2


def test_identical_bodies_in_different_functions_do_not_collide():
    @cache
    def first(x):
        return x

    @cache
    def second(x):
        return x

    assert first(1) == 1
    assert second(1) == 1
    stats_keys = {first.cache.namespace, second.cache.namespace}
    assert len(stats_keys) == 2


def test_explicit_namespace_override():
    @cache(namespace="features.v2")
    def build(x):
        return x

    assert build.cache.namespace == "features.v2"


def test_exceptions_are_not_cached():
    attempts = []

    @cache
    def flaky(x):
        attempts.append(x)
        if len(attempts) == 1:
            raise ValueError("first call fails")
        return x

    with pytest.raises(ValueError):
        flaky(1)
    assert flaky(1) == 1
    assert attempts == [1, 1]


def test_clear_forgets_results():
    calls = []

    @cache
    def expensive(x):
        calls.append(x)
        return x

    expensive(1)
    expensive.cache.clear()
    expensive(1)
    assert calls == [1, 1]


def test_clear_is_per_function():
    first_calls = []
    second_calls = []

    @cache
    def first(x):
        first_calls.append(x)
        return x

    @cache
    def second(x):
        second_calls.append(x)
        return x

    first(1)
    second(1)
    first.cache.clear()
    first(1)
    second(1)
    assert first_calls == [1, 1]
    assert second_calls == [1]


def test_unhashable_argument_fails_loudly_at_call():
    class Opaque:
        pass

    @cache
    def expensive(data):
        return data

    with pytest.raises(UnhashableArgumentError):
        expensive(Opaque())


def test_wrapped_metadata_preserved():
    @cache
    def documented(x):
        """Docstring survives."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring survives."


def test_code_change_invalidates(tmp_path):
    """Same function name/module, different implementation => different entries."""
    import importlib.util
    import sys

    def load(source, tag):
        path = tmp_path / f"mod_{tag}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("cachau_test_mod", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["cachau_test_mod"] = module
        spec.loader.exec_module(module)
        return module.compute

    from cachau.fingerprint import function_fingerprint

    v1 = load("def compute(x):\n    return x * 2\n", "v1")
    v2 = load("def compute(x):\n    return x * 3\n", "v2")
    assert function_fingerprint(v1) != function_fingerprint(v2)
