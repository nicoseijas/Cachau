"""Function identity: code fingerprint and namespaces (GUIDELINES.md §2-3, §7)."""

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
