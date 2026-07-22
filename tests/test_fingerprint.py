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


def test_positional_defaults_are_part_of_identity():
    """A default lives on the function object, not in co_consts (#50):
    editing one changes behavior without touching the bytecode."""

    def before(x, mult=2):
        return x * mult

    def after(x, mult=3):
        return x * mult

    def same(x, mult=2):
        return x * mult

    assert function_fingerprint(before) != function_fingerprint(after)
    assert function_fingerprint(before) == function_fingerprint(same)


def test_keyword_only_defaults_are_part_of_identity():
    def before(x, *, mult=2):
        return x * mult

    def after(x, *, mult=3):
        return x * mult

    assert function_fingerprint(before) != function_fingerprint(after)


def test_function_valued_defaults_recurse_into_their_fingerprint():
    def double(x):
        return x * 2

    def triple(x):
        return x * 3

    def with_double(x, fn=double):
        return fn(x)

    def with_triple(x, fn=triple):
        return fn(x)

    assert function_fingerprint(with_double) != function_fingerprint(with_triple)


def test_opaque_defaults_are_instance_stable():
    class Opaque:
        pass

    shared = Opaque()

    def make(obj):
        def f(x, o=obj):  # default evaluated at def time: lands in __defaults__
            return (x, o)

        return f

    assert function_fingerprint(make(shared)) == function_fingerprint(make(shared))
    assert function_fingerprint(make(shared)) != function_fingerprint(make(Opaque()))


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

    # The premise of the bug is version-dependent. Up to 3.13 the literals
    # live in co_consts and the bytecode is byte-identical, which is exactly
    # what made the collision reachable. 3.14 encodes small ints in the
    # instruction itself, so the two bodies no longer share bytecode there
    # (their co_code differs precisely at the operands 1/12 and 23/3) and this
    # particular collision is unreachable. Larger constants, floats and strings
    # still go through co_consts on every version — the fingerprints must
    # differ either way, which is what the assertion below actually protects.
    if sys.version_info < (3, 14):
        assert f.__code__.co_code == g.__code__.co_code
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


def test_nested_code_block_boundary_cannot_be_reframed():
    """#54: the nested-code block was emitted unframed, so a parent whose
    trailing constants continue where the nested block ends could collide
    with a parent whose nested child absorbed those constants. Only
    reachable with fabricated code objects — defense in depth."""
    import types

    def template():
        pass

    inner = (lambda: None).__code__
    inner_absorbing = inner.replace(co_consts=inner.co_consts + ("x",))
    parent_a = template.__code__.replace(co_consts=(inner, "x"))
    parent_b = template.__code__.replace(co_consts=(inner_absorbing,))
    func_a = types.FunctionType(parent_a, {})
    func_b = types.FunctionType(parent_b, {})

    assert function_fingerprint(func_a) != function_fingerprint(func_b)
    assert function_fingerprint(func_a) == function_fingerprint(
        types.FunctionType(template.__code__.replace(co_consts=(inner, "x")), {})
    )
