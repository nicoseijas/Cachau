"""verify=: sample-recompute HITs and fail loudly on mismatch (#28).

The safety net for what no fingerprint can catch: transitive code changes that
slip past the fingerprint, and nondeterministic functions whose cached value
silently diverges from a fresh call. On a mismatch the fresh value wins
(cardinal invariant: under uncertainty, recompute), the event is warned and
counted with its own miss reason.
"""

import warnings

import pytest

from cachau import CacheVerificationWarning, cache
from cachau.errors import ConfigurationError


def test_verify_defaults_off():
    calls = []

    @cache
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(1) == 2
    assert expensive(1) == 2
    assert calls == [1]
    assert expensive.cache.stats().verifications == 0


def test_verify_zero_never_recomputes():
    calls = []

    @cache(verify=0)
    def expensive(x):
        calls.append(x)
        return x * 2

    expensive(1)
    expensive(1)
    assert calls == [1]


def test_verify_one_recomputes_every_hit():
    calls = []

    @cache(verify=1.0)
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(1) == 2  # miss: computes and caches
    assert expensive(1) == 2  # hit, selected: recomputes to compare
    assert calls == [1, 1]
    stats = expensive.cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.verifications == 1
    assert stats.verification_failures == 0


def test_verified_match_does_not_warn():
    @cache(verify=1.0)
    def expensive(x):
        return x * 2

    expensive(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CacheVerificationWarning)
        assert expensive(1) == 2  # deterministic: verification passes silently


def test_verified_mismatch_warns_and_the_fresh_value_wins():
    source = {"value": "old"}

    @cache(verify=1.0)
    def load():
        return source["value"]

    assert load() == "old"
    source["value"] = "new"  # the unfingerprinted change verify exists to catch
    with pytest.warns(CacheVerificationWarning):
        assert load() == "new"
    stats = load.cache.stats()
    assert stats.verifications == 1
    assert stats.verification_failures == 1
    assert stats.miss_verification_failed == 1
    assert stats.hits == 0
    assert stats.misses == 2  # the initial not_found plus the failed verification
    assert stats.writes == 2  # every miss commits exactly once — mismatch recommits


def test_verified_mismatch_replaces_the_cached_entry():
    source = {"value": "old"}

    @cache(verify=1.0)
    def load():
        return source["value"]

    load()
    source["value"] = "new"
    with pytest.warns(CacheVerificationWarning):
        load()
    # The fresh value was committed: the next verification passes silently.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CacheVerificationWarning)
        assert load() == "new"
    stats = load.cache.stats()
    assert stats.verifications == 2
    assert stats.verification_failures == 1


def test_verify_on_a_cold_cache_is_a_normal_miss():
    calls = []

    @cache(verify=1.0)
    def expensive(x):
        calls.append(x)
        return x * 2

    assert expensive(1) == 2
    assert calls == [1]
    stats = expensive.cache.stats()
    assert stats.miss_not_found == 1
    assert stats.verifications == 0  # nothing cached: there was nothing to verify


def test_verify_compares_arrays_by_content():
    np = pytest.importorskip("numpy")

    @cache(verify=1.0)
    def build():
        return np.arange(5)

    build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CacheVerificationWarning)
        result = build()  # fresh array, equal content: must not warn
    assert np.array_equal(result, np.arange(5))


def test_verify_hits_do_not_inflate_estimated_savings():
    @cache(verify=1.0)
    def expensive(x):
        return x * 2

    expensive(1)
    expensive(1)  # verified hit: full recompute happened, nothing was saved
    assert expensive.cache.stats().estimated_saved_seconds == 0.0


@pytest.mark.parametrize("bad", [-0.1, 1.5, "always", None])
def test_verify_rejects_invalid_probabilities(bad):
    with pytest.raises(ConfigurationError):

        @cache(verify=bad)
        def expensive(x):
            return x
