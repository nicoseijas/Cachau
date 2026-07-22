"""Numba Level A matrix: caching at the Python → dispatcher boundary
(GUIDELINES.md §14, ROADMAP Phase 1)."""

import pytest

np = pytest.importorskip("numpy")
numba = pytest.importorskip("numba")

from numba import njit, prange

from cachau import cache
from cachau.fingerprint import function_fingerprint


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Boundary basics
# ---------------------------------------------------------------------------


def test_hit_avoids_reexecution():
    @cache
    @njit
    def double(values):
        return values * 2

    arr = np.arange(10, dtype=np.float64)
    first = double(arr)
    second = double(np.arange(10, dtype=np.float64))  # equal content: HIT
    assert np.array_equal(first, np.arange(10) * 2.0)
    assert np.array_equal(first, second)
    stats = double.cache.stats()
    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.writes == 1


def test_scalar_inputs_and_outputs():
    @cache
    @njit
    def hypot2(a, b):
        return a * a + b * b

    assert hypot2(3.0, 4.0) == 25.0
    assert hypot2(3.0, 4.0) == 25.0
    assert hypot2(a=3.0, b=4.0) == 25.0  # kwargs normalize through the dispatcher
    stats = hypot2.cache.stats()
    assert stats.misses == 1
    assert stats.hits == 2


def test_ndarray_results_persist_across_restart(tmp_path):
    source = """
import numpy as np
from numba import njit
from cachau import cache

@cache(persist=PERSIST_DIR, namespace="numba.cumsum")
@njit
def cumulative(values):
    return np.cumsum(values)
"""

    def define():
        scope = {"PERSIST_DIR": str(tmp_path)}
        exec(source, scope)
        return scope["cumulative"]

    arr = np.arange(5, dtype=np.int64)
    first = define()
    result = first(arr)
    assert np.array_equal(result, np.array([0, 1, 3, 6, 10]))

    second = define()  # simulated restart: fresh dispatcher, fresh backend
    replayed = second(np.arange(5, dtype=np.int64))
    assert np.array_equal(replayed, result)
    assert second.cache.stats().hits == 1
    assert second.cache.stats().misses == 0


# ---------------------------------------------------------------------------
# Dispatcher identity: semantically relevant compile options
# ---------------------------------------------------------------------------


def test_fastmath_changes_identity():
    @njit
    def plain(x):
        return x * 2.0

    @njit(fastmath=True)
    def fast(x):
        return x * 2.0

    assert function_fingerprint(plain) != function_fingerprint(fast)


def test_parallel_changes_identity():
    @njit
    def serial(x):
        return x * 2.0

    @njit(parallel=True)
    def parallel(x):
        return x * 2.0

    assert function_fingerprint(serial) != function_fingerprint(parallel)


def test_error_model_changes_identity():
    @njit
    def default_model(x):
        return 1.0 / x

    @njit(error_model="numpy")
    def numpy_model(x):
        return 1.0 / x

    assert function_fingerprint(default_model) != function_fingerprint(numpy_model)


def test_boundscheck_changes_identity():
    @njit
    def unchecked(values):
        return values[0]

    @njit(boundscheck=True)
    def checked(values):
        return values[0]

    assert function_fingerprint(unchecked) != function_fingerprint(checked)


def test_same_options_share_identity():
    @njit(fastmath=True)
    def one(x):
        return x * 2.0

    @njit(fastmath=True)
    def two(x):
        return x * 2.0

    assert function_fingerprint(one) == function_fingerprint(two)


def test_changing_fastmath_invalidates_persisted_results(tmp_path):
    """GUIDELINES §14: changing semantically relevant configuration must
    invalidate stale persisted results."""
    template = """
from numba import njit
from cachau import cache

@cache(persist=PERSIST_DIR, namespace="numba.versioned")
@njit(fastmath=FASTMATH)
def compute(x):
    return x * 2.0
"""

    def define(fastmath):
        scope = {"PERSIST_DIR": str(tmp_path), "FASTMATH": fastmath}
        exec(template, scope)
        return scope["compute"]

    v1 = define(False)
    v1(3.0)
    v2 = define(True)  # same body, new semantics: old entry must be purged
    assert v2.cache.stats().code_change_invalidations == 1
    v2(3.0)
    assert v2.cache.stats().misses == 1  # recomputed, not served stale


def test_code_change_invalidates_dispatchers():
    @njit
    def before(x):
        return x * 2.0

    @njit
    def after(x):
        return x * 3.0

    assert function_fingerprint(before) != function_fingerprint(after)


# ---------------------------------------------------------------------------
# Composition with Numba's own machinery
# ---------------------------------------------------------------------------


def test_parallel_executes_correctly_at_the_boundary():
    @cache
    @njit(parallel=True)
    def total(values):
        acc = 0.0
        for i in prange(values.shape[0]):
            acc += values[i]
        return acc

    arr = np.ones(1000, dtype=np.float64)
    assert total(arr) == 1000.0
    assert total(np.ones(1000, dtype=np.float64)) == 1000.0
    assert total.cache.stats().hits == 1


@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaWarning")
def test_coexists_with_numba_compilation_cache(tmp_path):
    """Cachau caches results; njit(cache=True) caches machine code. A Cachau
    MISS still executes fine with Numba's compilation cache active — but the
    stacked configuration is advised against (see MachineCodeCacheWarning)."""
    from cachau import MachineCodeCacheWarning

    with pytest.warns(MachineCodeCacheWarning):

        @cache(persist=str(tmp_path))
        @njit(cache=True)
        def triple(x):
            return x * 3.0

    assert triple(2.0) == 6.0
    assert triple(2.0) == 6.0
    stats = triple.cache.stats()
    assert stats.misses == 1
    assert stats.hits == 1


# ---------------------------------------------------------------------------
# Core features against Numba workloads
# ---------------------------------------------------------------------------


def test_ttl_applies_to_numba_results():
    clock = FakeClock()

    @cache(ttl=60, clock=clock)
    @njit
    def square(x):
        return x * x

    square(3.0)
    clock.now = 61.0
    square(3.0)
    stats = square.cache.stats()
    assert stats.miss_expired == 1
    assert stats.misses == 2


def test_max_memory_budgets_array_results():
    @cache(max_memory="1MB")
    @njit
    def zeros_like_range(n):
        return np.zeros(n, dtype=np.float64)

    small = zeros_like_range(100)  # 800 B: cached
    assert small.shape == (100,)
    stats = zeros_like_range.cache.stats()
    assert stats.writes == 1
    assert stats.current_bytes >= 800

    big = zeros_like_range(1_000_000)  # ~8 MB: oversized, computed, not cached
    assert big.shape == (1_000_000,)
    assert zeros_like_range.cache.stats().skipped_oversized == 1


def test_invalidate_and_explain_through_the_dispatcher():
    @cache
    @njit
    def double(x):
        return x * 2.0

    double(5.0)
    assert double.cache.explain(5.0).outcome == "HIT"
    assert double.cache.explain(x=5.0).outcome == "HIT"  # normalized
    double.cache.invalidate(5.0)
    assert double.cache.explain(5.0).outcome == "MISS"
    assert double.cache.explain(5.0).reason == "invalidated"


# ---------------------------------------------------------------------------
# Honest metrics: cold vs warm JIT
# ---------------------------------------------------------------------------


def test_cold_jit_is_recorded_and_excluded_from_savings():
    """The one-time compilation cost must never be counted as normal
    execution cost (GUIDELINES §14)."""

    @cache
    @njit
    def work(values):
        return values.sum()

    a = np.arange(1000, dtype=np.float64)
    b = np.arange(2000, dtype=np.float64)

    work(a)  # cold: includes JIT compilation
    stats = work.cache.stats()
    assert stats.cold_compute_seconds > 0.0
    assert stats.cold_compute_seconds == stats.total_compute_seconds

    work(a)  # HIT — but no warm baseline exists yet, so savings stay honest
    assert work.cache.stats().estimated_saved_seconds == 0.0

    work(b)  # warm compute: same specialization, no compilation
    work(b)  # HIT: credited with the warm average only
    final = work.cache.stats()
    assert final.estimated_saved_seconds > 0.0
    assert final.estimated_saved_seconds < final.cold_compute_seconds


def test_factory_closures_never_collide():
    """CRITICAL regression: kernels from the same factory with different
    captured parameters must not share identity (false HIT)."""

    def make_adder(n):
        @cache
        @njit
        def adder(x):
            return x + n

        return adder

    add_one = make_adder(1)
    add_hundred = make_adder(100)
    assert add_one.cache.fingerprint != add_hundred.cache.fingerprint
    assert add_one(5.0) == 6.0
    assert add_hundred(5.0) == 105.0  # was 6.0 before the fix


def test_plain_python_factory_closures_never_collide():
    def make_scaler(factor):
        @cache
        def scale(x):
            return x * factor

        return scale

    double = make_scaler(2)
    triple = make_scaler(3)
    assert double(10) == 20
    assert triple(10) == 30


def test_locals_type_forcing_changes_identity():
    """CRITICAL regression: locals= forces intermediate precision and changes
    numeric results; it must be part of dispatcher identity."""
    from numba import float32, float64

    @njit(locals={"y": float32})
    def low(x):
        y = x / 3.0
        return y

    @njit(locals={"y": float64})
    def high(x):
        y = x / 3.0
        return y

    assert function_fingerprint(low) != function_fingerprint(high)


def test_changing_locals_invalidates_persisted_results(tmp_path):
    template = """
from numba import njit, float32, float64
from cachau import cache

@cache(persist=PERSIST_DIR, namespace="numba.locals")
@njit(locals={"y": PRECISION})
def compute(x):
    y = x / 3.0
    return y
"""

    def define(precision):
        scope = {"PERSIST_DIR": str(tmp_path), "PRECISION": precision}
        exec(template, scope)
        return scope["compute"]

    from numba import float32, float64

    v_low = define(float32)
    low_result = v_low(1.0)
    v_high = define(float64)
    assert v_high.cache.stats().code_change_invalidations == 1
    high_result = v_high(1.0)
    assert high_result != low_result  # float64 precision, not the stale float32


def test_new_specialization_compile_is_cold_not_warm():
    """HIGH regression: a later call with a new dtype triggers a fresh
    compile; it must be labeled cold and kept out of the savings baseline."""

    @cache
    @njit
    def add_one(x):
        return x + 1

    add_one(5)  # cold: int64 specialization
    first_cold = add_one.cache.stats().cold_compute_seconds
    assert first_cold > 0.0

    add_one(5.0)  # NEW float64 specialization: also cold
    stats = add_one.cache.stats()
    assert stats.cold_compute_seconds > first_cold  # accumulated both compiles

    add_one(5.0)  # HIT — no warm baseline exists, savings must stay at zero
    assert add_one.cache.stats().estimated_saved_seconds == 0.0

    add_one(6.0)  # warm: existing float64 specialization
    add_one(6.0)  # HIT: credited with warm time only
    final = add_one.cache.stats()
    assert 0.0 < final.estimated_saved_seconds < final.cold_compute_seconds


def test_njit_on_top_of_cache_fails_at_numba_not_silently():
    """Documented ordering: @cache goes BELOW @njit. The reverse tries to
    compile cachau's wrapper in nopython mode and fails loudly at Numba."""

    @njit
    @cache
    def wrong_order(x):
        return x + 1

    with pytest.raises(Exception):  # numba typing/bytecode error, never a value
        wrong_order(1.0)


def test_vectorize_dispatchers_are_rejected_clearly():
    from numba import vectorize

    from cachau.errors import ConfigurationError

    @vectorize(["float64(float64)"])
    def as_ufunc(x):
        return x * 2.0

    with pytest.raises(ConfigurationError):
        cache(as_ufunc)


def test_plain_python_functions_report_no_cold_jit():
    @cache
    def plain(x):
        return x * 2

    plain(1)
    assert plain.cache.stats().cold_compute_seconds == 0.0


# ---------------------------------------------------------------------------
# Stacked machine-code cache hazard (#32)
# ---------------------------------------------------------------------------


def test_decorating_a_cache_true_dispatcher_warns():
    from cachau import MachineCodeCacheWarning

    @njit(cache=True)
    def kernel(x):
        return x * 2.0

    with pytest.warns(MachineCodeCacheWarning):
        cache(kernel)


def test_decorating_a_plain_dispatcher_does_not_warn():
    import warnings as warnings_module

    from cachau import MachineCodeCacheWarning

    @njit
    def kernel(x):
        return x * 2.0

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error", MachineCodeCacheWarning)
        cache(kernel)


def test_decorating_a_plain_function_does_not_warn():
    import warnings as warnings_module

    from cachau import MachineCodeCacheWarning

    def plain(x):
        return x * 2.0

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error", MachineCodeCacheWarning)
        cache(plain)
