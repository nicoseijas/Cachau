"""Native hashing for Polars (GUIDELINES.md §2, §14; ROADMAP Phase 2).

Same bar as NumPy/pandas: deterministic content identity (equal frames share a
digest, any semantic difference forks it), no import triggered by cachau
itself, sizes estimated natively, and stability across processes.
"""

import os
import subprocess
import sys

import pytest

pl = pytest.importorskip("polars")

from cachau import cache
from cachau.errors import UnhashableArgumentError
from cachau.keys import digest_arguments
from cachau.sizes import estimate_size


def sample(a):
    return a


def digest(value):
    return digest_arguments(sample, (value,), {})


def frame():
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# --------------------------------------------------------------------------- #
# DataFrame identity
# --------------------------------------------------------------------------- #


def test_equal_frames_share_a_digest():
    assert digest(frame()) == digest(frame())


def test_cell_content_matters():
    changed = pl.DataFrame({"a": [1, 2, 4], "b": ["x", "y", "z"]})
    assert digest(frame()) != digest(changed)


def test_column_names_matter():
    renamed = frame().rename({"a": "z"})
    assert digest(frame()) != digest(renamed)


def test_column_order_matters():
    assert digest(frame()) != digest(frame().select("b", "a"))


def test_row_order_matters():
    assert digest(frame()) != digest(frame().reverse())


def test_dtype_matters():
    i64 = pl.DataFrame({"a": [1, 2, 3]})
    i32 = pl.DataFrame({"a": [1, 2, 3]}, schema={"a": pl.Int32})
    assert digest(i64) != digest(i32)


def test_null_is_distinct_from_a_value():
    with_null = pl.DataFrame({"a": [1, None, 3]})
    with_zero = pl.DataFrame({"a": [1, 0, 3]})
    assert digest(with_null) != digest(with_zero)


def test_empty_frames_differ_by_schema():
    a = pl.DataFrame(schema={"a": pl.Int64})
    b = pl.DataFrame(schema={"b": pl.Int64})
    assert digest(a) != digest(b)
    assert digest(a) == digest(pl.DataFrame(schema={"a": pl.Int64}))


def test_nested_dtypes_hash():
    nested = pl.DataFrame({"lst": [[1, 2], [3]], "st": [{"x": 1}, {"x": 2}]})
    same = pl.DataFrame({"lst": [[1, 2], [3]], "st": [{"x": 1}, {"x": 2}]})
    different = pl.DataFrame({"lst": [[1, 2], [4]], "st": [{"x": 1}, {"x": 2}]})
    assert digest(nested) == digest(same)
    assert digest(nested) != digest(different)


# --------------------------------------------------------------------------- #
# Series identity
# --------------------------------------------------------------------------- #


def test_equal_series_share_a_digest():
    assert digest(pl.Series("s", [1.0, 2.0])) == digest(pl.Series("s", [1.0, 2.0]))


def test_series_name_content_and_dtype_matter():
    base = pl.Series("s", [1, 2])
    assert digest(base) != digest(pl.Series("t", [1, 2]))
    assert digest(base) != digest(pl.Series("s", [1, 3]))
    assert digest(base) != digest(pl.Series("s", [1, 2], dtype=pl.Int32))


def test_series_never_collides_with_a_frame():
    assert digest(pl.Series("a", [1, 2, 3])) != digest(pl.DataFrame({"a": [1, 2, 3]}))


# --------------------------------------------------------------------------- #
# LazyFrame: a query plan, not data
# --------------------------------------------------------------------------- #


def test_lazyframe_fails_loudly_with_guidance():
    with pytest.raises(UnhashableArgumentError, match="collect"):
        digest(frame().lazy())


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_cache_hits_on_equal_frames_and_recomputes_on_changed():
    calls = []

    @cache
    def total(df):
        calls.append(1)
        return int(df["a"].sum())

    assert total(frame()) == 6
    assert total(frame()) == 6  # equal content, distinct object: HIT
    assert calls == [1]
    assert total(frame().with_columns(pl.col("a") * 2)) == 12
    assert calls == [1, 1]


def test_polars_results_persist_and_round_trip(tmp_path):
    @cache(persist=str(tmp_path))
    def build():
        return frame()

    first = build()
    build.cache._backend  # force nothing; the round trip happens on the HIT below
    again = build()
    assert again.equals(first)


def test_digest_is_stable_across_processes():
    """Key identity must not depend on PYTHONHASHSEED or process state —
    otherwise persist= would never find its own entries after a restart."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    script = f"""
import sys
sys.path.insert(0, {src!r})
import polars as pl
from cachau.keys import digest_arguments

def f(a):
    return a

df = pl.DataFrame({{"a": [1, 2, 3], "b": ["x", "y", "z"]}})
print(digest_arguments(f, (df,), {{}}))
"""
    digests = set()
    for seed in ("0", "424242"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1


def test_profile_names_a_polars_frame_as_the_keygen_culprit():
    from cachau.keys import normalize_call
    from cachau.profile import largest_data_arg

    big = pl.DataFrame({"a": list(range(50_000))})

    def process(df, n):
        return n

    largest = largest_data_arg(normalize_call(process, (big, 1), {}))
    assert largest is not None
    assert largest[0].startswith("DataFrame[")
    assert largest[1] >= 50_000 * 8


# --------------------------------------------------------------------------- #
# Sizes
# --------------------------------------------------------------------------- #


def test_estimate_size_uses_polars_native_accounting():
    big = pl.DataFrame({"a": list(range(100_000))})
    estimated = estimate_size(big)
    assert estimated >= big.estimated_size()
    assert estimated >= 100_000 * 8  # not the sys.getsizeof shell size


def test_estimate_size_covers_series():
    series = pl.Series("s", list(range(100_000)))
    assert estimate_size(series) >= 100_000 * 8
