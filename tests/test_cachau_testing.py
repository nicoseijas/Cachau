"""cachau.testing: certify the cache's contract from a user test suite (#33).

Two assertions, both of which must be able to FAIL — a check that cannot fail
certifies nothing: ``assert_cache_faithful`` (a HIT serves exactly what a fresh
compute produces) and ``assert_invalidates`` (a perturbation actually turns the
next check into a MISS).
"""

import pytest

import cachau
from cachau import cache
from cachau.testing import assert_cache_faithful, assert_invalidates


# --------------------------------------------------------------------------- #
# assert_cache_faithful
# --------------------------------------------------------------------------- #


def test_faithful_passes_for_a_deterministic_function():
    @cache
    def double(x):
        return x * 2

    assert_cache_faithful(double, 21)


def test_faithful_primes_a_cold_cache():
    calls = []

    @cache
    def double(x):
        calls.append(x)
        return x * 2

    assert_cache_faithful(double, 21)  # nothing cached yet: primes, then checks
    assert calls  # the function really ran


def test_faithful_fails_when_cached_and_fresh_diverge():
    counter = {"n": 0}

    @cache
    def unfaithful():
        counter["n"] += 1
        return counter["n"]

    unfaithful()  # caches 1; every fresh compute from here on differs
    with pytest.raises(AssertionError, match="faithful"):
        assert_cache_faithful(unfaithful)


def test_faithful_compares_arrays_by_content():
    np = pytest.importorskip("numpy")

    @cache
    def build():
        return np.arange(1000, dtype=np.float64)

    assert_cache_faithful(build)


def test_faithful_fails_loudly_when_nothing_caches(tmp_path):
    @cache(persist=str(tmp_path))
    def unpicklable(x):
        return lambda: x  # every write fails: there is never a HIT to certify

    with pytest.raises(AssertionError, match="not_found"):
        assert_cache_faithful(unpicklable, 1)


def test_faithful_rejects_a_plain_function():
    def plain(x):
        return x

    with pytest.raises(TypeError):
        assert_cache_faithful(plain, 1)


# --------------------------------------------------------------------------- #
# assert_invalidates
# --------------------------------------------------------------------------- #


def test_invalidates_passes_when_the_dependency_bites():
    provenance = {"v": "v1"}

    @cache(depends_on=[cachau.token(lambda: provenance["v"], name="prov")])
    def solve(x):
        return x * 2

    explanation = assert_invalidates(
        solve, lambda: provenance.update(v="v2"), 10,
        reason="dependency_changed",
    )
    assert explanation.outcome == "MISS"
    assert explanation.changed_dependencies == ("token:prov",)


def test_invalidates_fails_when_the_perturbation_does_nothing():
    provenance = {"v": "v1"}

    @cache(depends_on=[cachau.token(lambda: provenance["v"], name="prov")])
    def solve(x):
        return x * 2

    with pytest.raises(AssertionError, match="did not invalidate"):
        assert_invalidates(solve, lambda: None, 10)


def test_invalidates_checks_the_expected_reason():
    provenance = {"v": "v1"}

    @cache(depends_on=[cachau.token(lambda: provenance["v"], name="prov")])
    def solve(x):
        return x * 2

    with pytest.raises(AssertionError, match="expired"):
        assert_invalidates(
            solve, lambda: provenance.update(v="v2"), 10, reason="expired"
        )


def test_invalidates_fails_loudly_when_nothing_caches(tmp_path):
    @cache(persist=str(tmp_path))
    def unpicklable(x):
        return lambda: x

    with pytest.raises(AssertionError, match="not_found"):
        assert_invalidates(unpicklable, lambda: None, 1)


def test_invalidates_works_with_file_dependencies(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b,c")

    @cache(depends_on=[str(data)])
    def load(n):
        return n

    assert_invalidates(
        load, lambda: data.write_text("a,b,c,d"), 1, reason="dependency_changed"
    )
