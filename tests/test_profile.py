"""profile(): measured cache economics (GUIDELINES.md §15).

Unlike explain(), profile() runs the function to measure recompute cost. The
timing-dependent behaviour is smoke-tested tolerantly (verdicts, not exact
milliseconds); the decision logic is tested deterministically through diagnose()
with synthetic numbers so it never flakes on a loaded CI box.
"""

import time

import pytest

from cachau import cache
from cachau.errors import ConfigurationError
from cachau.memory import MemoryBackend
from cachau.profile import (
    CacheProfile,
    _format_seconds,
    diagnose,
    largest_data_arg,
)


# --------------------------------------------------------------------------- #
# diagnose(): the decision logic, deterministic
# --------------------------------------------------------------------------- #


def test_diagnose_worth_it_when_compute_dominates():
    verdict, primary, rec = diagnose(
        compute_seconds=1.0, key_seconds=0.01, read_seconds=0.01, largest_data=None
    )
    assert verdict == "worth_it"
    assert "keep it" in rec.lower()


def test_diagnose_worth_it_notes_large_array_hit_cost():
    verdict, primary, rec = diagnose(
        compute_seconds=10.0,
        key_seconds=0.4,
        read_seconds=0.01,
        largest_data=("ndarray[float64, 480.0 MB]", 480_000_000),
    )
    assert verdict == "worth_it"
    assert primary == "key generation"
    assert "ndarray[float64, 480.0 MB]" in rec


def test_diagnose_not_worth_it_blames_array_hashing():
    verdict, primary, rec = diagnose(
        compute_seconds=0.004,
        key_seconds=0.016,
        read_seconds=0.0,
        largest_data=("ndarray[float64, 30.5 MB]", 32_000_000),
    )
    assert verdict == "not_worth_it"
    assert primary == "hashing ndarray[float64, 30.5 MB]"
    assert "key=" in rec


def test_diagnose_not_worth_it_when_read_beats_recompute():
    verdict, primary, rec = diagnose(
        compute_seconds=0.001, key_seconds=0.0, read_seconds=0.02, largest_data=None
    )
    assert verdict == "not_worth_it"
    assert "recompute" in rec.lower() and "not caching" in rec.lower()


def test_diagnose_marginal_near_break_even():
    verdict, _, _ = diagnose(
        compute_seconds=0.010, key_seconds=0.005, read_seconds=0.005, largest_data=None
    )
    assert verdict == "marginal"


def test_diagnose_zero_hit_cost_is_worth_it():
    verdict, _, _ = diagnose(
        compute_seconds=0.001, key_seconds=0.0, read_seconds=0.0, largest_data=None
    )
    assert verdict == "worth_it"


# --------------------------------------------------------------------------- #
# Formatting and argument inspection
# --------------------------------------------------------------------------- #


def test_format_seconds_scales():
    assert _format_seconds(0.0000004) == "0 us"
    assert _format_seconds(0.000006) == "6 us"
    assert _format_seconds(0.0431) == "43.1 ms"
    assert _format_seconds(10.2) == "10.20 s"


def test_largest_data_arg_none_without_numpy_types():
    assert largest_data_arg((("x", 1), ("y", "hello"))) is None


def test_largest_data_arg_picks_biggest_ndarray():
    np = pytest.importorskip("numpy")
    small = np.ones(10)
    big = np.ones((1000, 100))
    label, nbytes = largest_data_arg((("a", small), ("b", big), ("c", 5)))
    assert "ndarray" in label
    assert nbytes == big.nbytes


# --------------------------------------------------------------------------- #
# profile(): integration (tolerant on timing)
# --------------------------------------------------------------------------- #


def test_profile_worth_it_for_expensive_function():
    @cache
    def slow(n):
        time.sleep(0.02)
        return n * n

    p = slow.cache.profile(5)
    assert isinstance(p, CacheProfile)
    assert p.worth_it
    assert p.speedup > 1
    assert p.saved_seconds > 0
    assert p.namespace.endswith("slow")


def test_profile_not_worth_it_for_trivial_function():
    @cache
    def trivial(n):
        return n + 1

    p = trivial.cache.profile(3, repeats=5)
    assert not p.worth_it  # a bare add is not worth a key-build + lookup


def test_profile_runs_function_warmup_plus_repeats():
    calls = []

    @cache
    def f(n):
        calls.append(n)
        return n

    f.cache.profile(1, repeats=3)
    assert len(calls) == 1 + 3  # one warm-up, then `repeats` measured runs


def test_profile_does_not_touch_stats():
    @cache
    def f(n):
        return n

    before = f.cache.stats()
    f.cache.profile(1)
    assert f.cache.stats() == before  # measurement bypasses the counters


def test_profile_restores_an_absent_entry():
    backend = MemoryBackend()

    @cache(backend=backend)
    def f(n):
        return n

    f.cache.profile(1)  # entry was absent → measured against a throwaway
    assert f.cache.stats().entries == 0  # and removed again, none left behind


def test_profile_does_not_disturb_an_existing_entry():
    backend = MemoryBackend()

    @cache(backend=backend)
    def f(n):
        return n * 10

    f(2)  # populate a real entry
    entry_before = backend.get(next(k for k, _ in backend.iter_entries()))
    f.cache.profile(2)
    entries = list(backend.iter_entries())
    assert len(entries) == 1  # still there
    assert entries[0][1].value == entry_before.value == 20


def test_profile_repeats_must_be_positive():
    @cache
    def f(n):
        return n

    with pytest.raises(ConfigurationError):
        f.cache.profile(1, repeats=0)


def test_profile_works_with_custom_key():
    @cache(key=lambda dataset, version: version)
    def process(dataset, version):
        return len(dataset) + version

    p = process.cache.profile([1, 2, 3], 7)  # unhashable dataset, keyed by version
    assert isinstance(p, CacheProfile)


def test_profile_str_renders_the_report():
    @cache
    def slow(n):
        time.sleep(0.01)
        return n

    text = str(slow.cache.profile(1))
    assert "Cache economics" in text
    assert "Warm recompute" in text
    assert "Cache hit total" in text
    assert "Recommendation" in text
    assert all(ord(c) < 128 for c in text)  # printable on any console


def test_profile_is_immutable():
    import dataclasses

    @cache
    def f(n):
        return n

    p = f.cache.profile(1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.verdict = "worth_it"
