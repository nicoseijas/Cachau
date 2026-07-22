"""cachau.array_token(): hash a big immutable value once, reuse the digest (#34).

The recipe for the keygen trap profile() diagnoses: a large array argument that
is immutable for the whole run should not be content-hashed on every call. The
memo must be identity-safe — a stale digest surviving an id() reuse would key
the wrong content, a false HIT.
"""

import gc

import pytest

import cachau
from cachau import cache
from cachau.tokens import _TOKEN_MEMO, array_token


@pytest.fixture(autouse=True)
def _clean_memo():
    _TOKEN_MEMO.clear()
    yield
    _TOKEN_MEMO.clear()


def test_same_object_hashes_once():
    np = pytest.importorskip("numpy")
    arr = np.arange(1000)
    calls = []
    original = cachau.tokens._digest_value

    def counting(value):
        calls.append(1)
        return original(value)

    cachau.tokens._digest_value = counting
    try:
        first = array_token(arr)
        second = array_token(arr)
    finally:
        cachau.tokens._digest_value = original
    assert first == second
    assert len(calls) == 1


def test_equal_content_gives_equal_tokens():
    np = pytest.importorskip("numpy")
    assert array_token(np.arange(100)) == array_token(np.arange(100))


def test_different_content_gives_different_tokens():
    np = pytest.importorskip("numpy")
    assert array_token(np.arange(100)) != array_token(np.arange(101))


def test_a_stale_memo_entry_for_a_reused_id_is_ignored():
    """id() reuse must never resurrect a dead object's digest — false-HIT bug."""
    np = pytest.importorskip("numpy")

    class _DeadRef:
        def __call__(self):
            return None  # the object this entry described no longer exists

    arr = np.arange(50)
    _TOKEN_MEMO[id(arr)] = (_DeadRef(), "stale-digest-of-something-else")
    assert array_token(arr) != "stale-digest-of-something-else"
    assert array_token(arr) == array_token(np.arange(50))


def test_dead_entries_are_pruned():
    np = pytest.importorskip("numpy")
    arr = np.arange(10)
    array_token(arr)
    assert len(_TOKEN_MEMO) == 1
    del arr
    gc.collect()
    assert len(_TOKEN_MEMO) == 0


def test_non_weakrefable_values_still_tokenize():
    # ints cannot be weak-referenced: no memo, but a correct, stable digest.
    assert array_token(42) == array_token(42)
    assert array_token(42) != array_token(43)
    assert len(_TOKEN_MEMO) == 0


def test_usable_as_an_explicit_key():
    np = pytest.importorskip("numpy")
    table = np.arange(10_000, dtype=np.float64)
    calls = []

    @cache(key=lambda table, n: (array_token(table), n))
    def lookup(table, n):
        calls.append(n)
        return float(table[n])

    assert lookup(table, 7) == 7.0
    assert lookup(table, 7) == 7.0  # same token, no re-hash of the payload
    assert calls == [7]
