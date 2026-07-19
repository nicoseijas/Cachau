"""Integration: cachau over numba-utils decorators.

numba-utils (https://github.com/nicoseijas/numba-utils) wraps ``numba.njit``
in thin aliases with curated defaults. They return real Numba dispatchers, so
cachau's duck-typed dispatcher identity must see through them — including the
options the aliases inject (``fastmath``, ``parallel``) and the library's
global ``configure()`` overrides.

Skipped unless numba-utils is importable.
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("numba")
nu_decorators = pytest.importorskip("numba_utils.decorators")

from numba_utils.decorators import cached_njit, njit_fast, njit_parallel

from cachau import cache
from cachau.errors import UnhashableArgumentError
from cachau.fingerprint import function_fingerprint, is_jit_dispatcher


@pytest.fixture(autouse=True)
def reset_numba_utils_config():
    from numba_utils._config import config

    yield
    config.reset()


def test_njit_fast_is_seen_as_a_dispatcher():
    @njit_fast
    def kernel(x):
        return x * 2.0

    assert is_jit_dispatcher(kernel)
    assert kernel.targetoptions.get("fastmath") is True


def test_end_to_end_caching_over_njit_fast():
    @cache
    @njit_fast
    def kernel(values):
        return values * 2.0

    first = kernel(np.arange(5, dtype=np.float64))
    second = kernel(np.arange(5, dtype=np.float64))
    assert np.array_equal(first, second)
    stats = kernel.cache.stats()
    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.cold_compute_seconds > 0.0  # njit_fast compile counted as cold


def test_alias_injected_fastmath_is_part_of_identity():
    """Same body through njit_fast (fastmath) vs cached_njit (exact IEEE):
    different numeric semantics, must never share a fingerprint."""

    @njit_fast
    def fast(x):
        return x * 2.0

    @cached_njit
    def exact(x):
        return x * 2.0

    assert function_fingerprint(fast) != function_fingerprint(exact)


def test_global_configure_override_changes_identity():
    from numba_utils import _config

    @njit_fast
    def before(x):
        return x * 2.0

    _config.configure(fastmath=False)

    @njit_fast
    def after(x):
        return x * 2.0

    assert function_fingerprint(before) != function_fingerprint(after)


def test_parallel_alias_end_to_end():
    from numba import prange

    @cache
    @njit_parallel
    def total(values):
        acc = 0.0
        for i in prange(values.shape[0]):
            acc += values[i]
        return acc

    assert total(np.ones(100, dtype=np.float64)) == 100.0
    assert total(np.ones(100, dtype=np.float64)) == 100.0
    assert total.cache.stats().hits == 1
    assert total.__wrapped__.targetoptions.get("parallel") is True


def test_numba_utils_kernel_results_persist(tmp_path):
    @cache(persist=str(tmp_path), namespace="compat.persist")
    @njit_fast
    def kernel(x):
        return x * 3.0

    kernel(np.arange(3, dtype=np.float64))
    assert len(list(tmp_path.glob("*.cachau"))) == 1


def test_typed_containers_fail_loudly_as_arguments():
    """numba_utils.collections containers are Level B: unsupported arguments
    must raise, never be silently ignored. key=/ignore= are the escape."""
    from numba.typed import List as TypedList

    @cache
    def consume(items):
        return len(items)

    with pytest.raises(UnhashableArgumentError):
        consume(TypedList([1.0, 2.0]))
