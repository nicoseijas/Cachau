"""cachau.code(): a helper's implementation as a declared dependency (#27).

The function fingerprint covers only the cached function's own code object
(plus closures): a module-level helper called by global lookup can change
without invalidating — a false HIT across processes. ``cachau.code(helper)``
closes the gap by folding the helper's implementation fingerprint into every
entry, and ``profile()`` flags such helpers when they are left undeclared.
"""

from json import loads as json_loads

import cachau
import pytest
from cachau import cache
from cachau.dependencies import CodeDependency, normalize_dependencies
from cachau.errors import ConfigurationError


def _double(x):
    return x * 2


def _double_clone(x):
    return x * 2


def _triple(x):
    return x * 3


def _scale(x):
    return x * 2


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #


def test_fingerprint_differs_when_implementation_differs():
    assert cachau.code(_double).fingerprint() != cachau.code(_triple).fingerprint()


def test_fingerprint_ignores_function_name_and_location():
    # Renaming or moving a helper must not invalidate — only changing what it does.
    assert cachau.code(_double).fingerprint() == cachau.code(_double_clone).fingerprint()


def test_label_is_the_stable_module_qualname():
    assert cachau.code(_scale).label() == f"code:{__name__}._scale"


def test_label_accepts_a_name_override():
    assert cachau.code(_scale, name="scaler").label() == "code:scaler"


def test_code_rejects_non_callables_at_declaration():
    with pytest.raises(ConfigurationError):
        cachau.code(42)


def test_code_rejects_unfingerprintable_callables_at_declaration():
    with pytest.raises(ConfigurationError):
        cachau.code(len)  # builtins carry no code object


def test_code_unwraps_a_cached_function_to_its_implementation():
    @cache
    def helper(x):
        return x * 2

    # Declaring a dependency on another cached function must fingerprint the
    # user's implementation, not cachau's wrapper.
    assert cachau.code(helper).fingerprint() == cachau.code(_double).fingerprint()


def test_duplicate_code_dependencies_fail_at_decoration():
    with pytest.raises(ConfigurationError, match="duplicate"):
        normalize_dependencies([cachau.code(_scale), cachau.code(_scale)])


def test_code_dependency_satisfies_the_dependency_protocol():
    dep = cachau.code(_scale)
    assert isinstance(dep, CodeDependency)
    labeled = normalize_dependencies([dep])
    assert labeled[0].label == f"code:{__name__}._scale"


# --------------------------------------------------------------------------- #
# The false-HIT scenario, closed
# --------------------------------------------------------------------------- #


def test_edited_helper_recomputes_instead_of_false_hit():
    """The twin-harness repro: helper edited (x*2 -> x*3), cache must not serve 20."""
    calls = []

    @cache(depends_on=[cachau.code(_scale)])
    def estimate(x):
        calls.append(x)
        return _scale(x)

    assert estimate(10) == 20
    assert estimate(10) == 20  # unchanged helper → served from cache
    assert calls == [10]
    original = _scale.__code__
    try:
        _scale.__code__ = _triple.__code__  # simulate the edit-and-reload
        explanation = estimate.cache.explain(10)
        assert explanation.outcome == "MISS"
        assert explanation.reason == "dependency_changed"
        assert explanation.changed_dependencies == (f"code:{__name__}._scale",)
        assert estimate(10) == 30  # recomputed with the new helper
        assert calls == [10, 10]
    finally:
        _scale.__code__ = original


# --------------------------------------------------------------------------- #
# profile() flags undeclared helpers
# --------------------------------------------------------------------------- #


def test_profile_flags_undeclared_helper_calls():
    @cache
    def estimate(x):
        return _scale(x)

    profiled = estimate.cache.profile(3, repeats=1)
    assert "_scale" in profiled.unfingerprinted_calls
    assert "cachau.code" in str(profiled)


def test_profile_does_not_flag_declared_helpers():
    @cache(depends_on=[cachau.code(_scale)])
    def estimate(x):
        return _scale(x)

    profiled = estimate.cache.profile(3, repeats=1)
    assert profiled.unfingerprinted_calls == ()
    assert "cachau.code" not in str(profiled)


def test_profile_does_not_flag_closure_captured_helpers():
    def make_estimator():
        def scale(x):
            return x * 2

        @cache
        def estimate(x):
            return scale(x)  # closure capture: the fingerprint recurses into it

        return estimate

    estimate = make_estimator()
    assert estimate.cache.profile(3, repeats=1).unfingerprinted_calls == ()


def test_profile_does_not_flag_third_party_functions():
    @cache
    def parse(text):
        return json_loads(text)  # another package's code: covered by cachau.package

    profiled = parse.cache.profile('{"a": 1}', repeats=1)
    assert profiled.unfingerprinted_calls == ()


def test_code_keeps_jit_dispatcher_compile_options():
    """A dispatcher must not unwrap to py_func: that would drop fastmath etc."""
    numba = pytest.importorskip("numba")

    @numba.njit(fastmath=True)
    def fast(x):
        return x * 2.0

    @numba.njit(fastmath=False)
    def careful(x):
        return x * 2.0

    assert cachau.code(fast).fingerprint() != cachau.code(careful).fingerprint()


def test_profile_flags_helpers_from_other_local_modules(tmp_path, monkeypatch):
    """Flat layouts too: a helper in a sibling module is still the user's code."""
    import sys
    import types as types_module

    module = types_module.ModuleType("local_helpers_for_test")
    module.__file__ = str(tmp_path / "local_helpers_for_test.py")
    exec("def sibling_scale(x):\n    return x * 2", module.__dict__)
    monkeypatch.setitem(sys.modules, "local_helpers_for_test", module)
    sibling_scale = module.sibling_scale
    globals()["_sibling_scale"] = sibling_scale
    try:

        @cache
        def estimate(x):
            return _sibling_scale(x)  # noqa: F821 - injected above

        profiled = estimate.cache.profile(3, repeats=1)
        assert "_sibling_scale" in profiled.unfingerprinted_calls
    finally:
        del globals()["_sibling_scale"]


def test_profile_does_not_flag_installed_package_functions():
    np = pytest.importorskip("numpy")
    array_equal = np.array_equal
    globals()["_np_array_equal"] = array_equal
    try:

        @cache
        def same(x):
            return bool(_np_array_equal(x, x))  # noqa: F821 - injected above

        profiled = same.cache.profile(3, repeats=1)
        assert profiled.unfingerprinted_calls == ()
    finally:
        del globals()["_np_array_equal"]


def test_profile_sees_helpers_called_from_nested_code():
    @cache
    def estimate(xs):
        return sum(map(lambda x: _scale(x), xs))

    profiled = estimate.cache.profile((1, 2), repeats=1)
    assert "_scale" in profiled.unfingerprinted_calls


# --------------------------------------------------------------------------- #
# Attribute-style calls: import mod; mod.helper() (#48)
# --------------------------------------------------------------------------- #


def _first_party_module(monkeypatch, tmp_path, name, source):
    """A real first-party module: importable, with a file outside site-packages."""
    import sys
    import types as types_module

    module = types_module.ModuleType(name)
    module.__file__ = str(tmp_path / f"{name}.py")
    exec(source, module.__dict__)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_profile_flags_attribute_style_helper_calls(tmp_path, monkeypatch):
    module = _first_party_module(
        monkeypatch, tmp_path, "attr_helpers_for_test",
        "def attr_scale(x):\n    return x * 2",
    )
    globals()["_attr_helpers"] = module
    try:

        @cache
        def estimate(x):
            return _attr_helpers.attr_scale(x)  # noqa: F821 - injected above

        profiled = estimate.cache.profile(3, repeats=1)
        assert "_attr_helpers.attr_scale" in profiled.unfingerprinted_calls
    finally:
        del globals()["_attr_helpers"]


def test_profile_flags_chained_module_attribute_calls(tmp_path, monkeypatch):
    sub = _first_party_module(
        monkeypatch, tmp_path, "attr_pkg_sub_for_test",
        "def deep_scale(x):\n    return x * 2",
    )
    pkg = _first_party_module(monkeypatch, tmp_path, "attr_pkg_for_test", "")
    pkg.sub = sub
    globals()["_attr_pkg"] = pkg
    try:

        @cache
        def estimate(x):
            return _attr_pkg.sub.deep_scale(x)  # noqa: F821 - injected above

        profiled = estimate.cache.profile(3, repeats=1)
        assert "_attr_pkg.sub.deep_scale" in profiled.unfingerprinted_calls
    finally:
        del globals()["_attr_pkg"]


def test_profile_does_not_flag_declared_attribute_helpers(tmp_path, monkeypatch):
    module = _first_party_module(
        monkeypatch, tmp_path, "attr_declared_for_test",
        "def declared_scale(x):\n    return x * 2",
    )
    globals()["_attr_declared"] = module
    try:

        @cache(depends_on=[cachau.code(module.declared_scale)])
        def estimate(x):
            return _attr_declared.declared_scale(x)  # noqa: F821 - injected above

        profiled = estimate.cache.profile(3, repeats=1)
        assert profiled.unfingerprinted_calls == ()
    finally:
        del globals()["_attr_declared"]


def test_profile_does_not_flag_stdlib_attribute_calls():
    import math

    globals()["_attr_math"] = math
    try:

        @cache
        def root(x):
            return _attr_math.sqrt(x)  # noqa: F821 - injected above

        profiled = root.cache.profile(9, repeats=1)
        assert profiled.unfingerprinted_calls == ()
    finally:
        del globals()["_attr_math"]


def test_profile_does_not_flag_installed_package_attribute_calls():
    np = pytest.importorskip("numpy")
    globals()["_attr_np"] = np
    try:

        @cache
        def zeros(n):
            return _attr_np.zeros(n).sum()  # noqa: F821 - injected above

        profiled = zeros.cache.profile(3, repeats=1)
        assert profiled.unfingerprinted_calls == ()
    finally:
        del globals()["_attr_np"]


def test_profile_sees_attribute_calls_from_nested_code(tmp_path, monkeypatch):
    module = _first_party_module(
        monkeypatch, tmp_path, "attr_nested_for_test",
        "def nested_scale(x):\n    return x * 2",
    )
    globals()["_attr_nested"] = module
    try:

        @cache
        def estimate(xs):
            return sum(_attr_nested.nested_scale(x) for x in xs)  # noqa: F821

        profiled = estimate.cache.profile((1, 2), repeats=1)
        assert "_attr_nested.nested_scale" in profiled.unfingerprinted_calls
    finally:
        del globals()["_attr_nested"]


def test_profile_flags_jit_dispatchers_reached_as_module_attributes(
    tmp_path, monkeypatch
):
    numba = pytest.importorskip("numba")
    import sys
    import types as types_module

    module = types_module.ModuleType("attr_jit_for_test")
    module.__file__ = str(tmp_path / "attr_jit_for_test.py")
    module.jit_scale = numba.njit(lambda x: x * 2.0)
    monkeypatch.setitem(sys.modules, "attr_jit_for_test", module)
    globals()["_attr_jit"] = module
    try:

        @cache
        def estimate(x):
            return _attr_jit.jit_scale(float(x))  # noqa: F821 - injected above

        profiled = estimate.cache.profile(3, repeats=1)
        assert "_attr_jit.jit_scale" in profiled.unfingerprinted_calls
    finally:
        del globals()["_attr_jit"]
