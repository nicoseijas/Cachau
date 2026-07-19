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
