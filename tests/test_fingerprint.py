"""Function identity: code fingerprint and namespaces (GUIDELINES.md §2-3, §7)."""

import os
import subprocess
import sys
from pathlib import Path

from cachau.fingerprint import function_fingerprint, function_namespace


def test_same_function_is_stable():
    def f(x):
        return x * 2

    assert function_fingerprint(f) == function_fingerprint(f)


def test_implementation_change_changes_fingerprint():
    def f(x):
        return x * 2

    def g(x):
        return x * 3

    assert function_fingerprint(f) != function_fingerprint(g)


def test_identical_bodies_share_fingerprint():
    def f(x):
        return x + 1

    def g(x):
        return x + 1

    assert function_fingerprint(f) == function_fingerprint(g)


def test_closure_values_are_part_of_identity():
    def make(n):
        def compute(x):
            return x + n

        return compute

    assert function_fingerprint(make(1)) != function_fingerprint(make(2))
    assert function_fingerprint(make(1)) == function_fingerprint(make(1))


def test_opaque_closure_captures_are_instance_stable():
    class Opaque:
        pass

    shared = Opaque()

    def make(obj):
        def compute(x):
            return (x, obj)

        return compute

    assert function_fingerprint(make(shared)) == function_fingerprint(make(shared))
    assert function_fingerprint(make(shared)) != function_fingerprint(make(Opaque()))


def test_captured_functions_recurse_into_their_fingerprint():
    def make(helper):
        def compute(x):
            return helper(x)

        return compute

    def double(x):
        return x * 2

    def triple(x):
        return x * 3

    assert function_fingerprint(make(double)) != function_fingerprint(make(triple))


def test_recursive_function_with_unbound_cell_does_not_crash():
    def make():
        def factorial(n):
            return 1 if n <= 1 else n * factorial(n - 1)

        return factorial

    assert function_fingerprint(make()) == function_fingerprint(make())


def test_default_namespace_is_module_qualified():
    def f(x):
        return x

    namespace = function_namespace(f)
    assert namespace.endswith(".test_default_namespace_is_module_qualified.<locals>.f")
    assert namespace.startswith(f.__module__)


def test_functions_with_same_name_in_different_scopes_do_not_collide():
    def make(factor):
        def compute(x):
            return x * factor

        return compute

    def other_compute(x):
        return x * 2

    assert function_namespace(make(2)) != function_namespace(other_compute)


# --- Cross-process stability (issue #13) ------------------------------------
#
# repr() of a set/frozenset iterates in hash order, which PYTHONHASHSEED
# randomizes per process. The peephole optimizer turns `x in {"a", "b"}` into
# a frozenset constant, so ordinary code hits this. A fingerprint that moves
# between runs silently defeats persist=: every restart is a guaranteed MISS.

_SEEDED_SOURCE = """
from cachau.fingerprint import function_fingerprint


def f(x):
    return x in {"alpha", "beta", "gamma", "delta", "epsilon"}


print(function_fingerprint(f))
"""


def _fingerprint_with_seed(seed):
    env = {
        **os.environ,
        "PYTHONHASHSEED": str(seed),
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", _SEEDED_SOURCE],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_set_constants_fingerprint_stably_across_hash_seeds():
    digests = {_fingerprint_with_seed(seed) for seed in (1, 2, 3, 4)}
    assert len(digests) == 1


def test_different_set_constants_still_differ():
    def f(x):
        return x in {"alpha", "beta"}

    def g(x):
        return x in {"alpha", "gamma"}

    assert function_fingerprint(f) != function_fingerprint(g)


def test_set_constant_is_not_confused_with_other_containers():
    def with_set(x):
        return x in {"alpha", "beta"}

    def with_tuple(x):
        return x in ("alpha", "beta")

    def with_frozenset(x):
        return x in frozenset(("alpha", "beta"))

    digests = {
        function_fingerprint(with_set),
        function_fingerprint(with_tuple),
        function_fingerprint(with_frozenset),
    }
    assert len(digests) == 3


def test_nested_and_mixed_type_set_constants_do_not_crash():
    def f(x):
        return x in {("a", 1), ("b", 2), frozenset({"c"})}

    assert function_fingerprint(f) == function_fingerprint(f)


def test_unorderable_mixed_set_constant_does_not_crash():
    """Canonicalization must not assume the elements are mutually sortable."""

    def f(x):
        return x in {"alpha", 1, None, 2.5, b"raw"}

    assert function_fingerprint(f) == function_fingerprint(f)


# --- Unambiguous framing of fed components (issue #16) ----------------------
#
# co_code stores constant INDICES, not values, so two functions differing only
# in numeric literals have byte-identical bytecode. If the constants are then
# concatenated without framing, `(None, 1, 23)` and `(None, 12, 3)` both
# render "None123" — same fingerprint, same key, and the cache serves one
# function's value for the other. A false HIT is the worst failure this
# library can produce.


def test_numeric_constants_do_not_collide_across_boundaries():
    def f(x):
        return x + 1 + 23

    def g(x):
        return x + 12 + 3

    assert f.__code__.co_code == g.__code__.co_code  # the premise of the bug
    assert function_fingerprint(f) != function_fingerprint(g)


def test_float_constants_do_not_collide_across_boundaries():
    def f(x):
        return x * 1.5 + 2.0

    def g(x):
        return x * 1.0 + 5.2

    assert function_fingerprint(f) != function_fingerprint(g)


def test_string_constants_do_not_collide_across_boundaries():
    def f(x):
        return ("ab", "c", x)

    def g(x):
        return ("a", "bc", x)

    assert function_fingerprint(f) != function_fingerprint(g)


def test_constant_count_alone_changes_the_fingerprint():
    def f(x):
        return x + 12

    def g(x):
        return x + 1 + 2

    assert function_fingerprint(f) != function_fingerprint(g)


def test_name_and_constant_boundaries_do_not_bleed_into_each_other():
    """A name must never be confusable with a constant that spells it."""

    def f(x):
        return {"ab": x}

    def g(x):
        return {"a": {"b": x}}

    assert function_fingerprint(f) != function_fingerprint(g)


def test_redefined_function_with_only_literal_changes_invalidates():
    """The notebook path: same qualname, edited body, must not serve the old value."""
    from cachau import cache
    from cachau.memory import MemoryBackend

    backend = MemoryBackend()
    scope = {"cache": cache, "backend": backend}
    cell = """
@cache(backend=backend)
def price(x):
    return x * {a} + {b}
"""
    exec(cell.format(a=1, b=23), scope)
    assert scope["price"](10) == 33

    exec(cell.format(a=12, b=3), scope)
    assert scope["price"](10) == 123  # recomputed, not the stale 33
    assert scope["price"].cache.stats().code_change_invalidations == 1
