"""Native hashing for NumPy and pandas (GUIDELINES.md §2, §14)."""

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cachau import cache
from cachau.keys import digest_arguments


def sample(a):
    return a


def digest(value):
    return digest_arguments(sample, (value,), {})


def test_equal_arrays_share_a_digest():
    assert digest(np.arange(100)) == digest(np.arange(100))


def test_content_matters():
    assert digest(np.array([1, 2, 3])) != digest(np.array([1, 2, 4]))


def test_dtype_matters():
    ones_i64 = np.ones(4, dtype=np.int64)
    ones_f64 = np.ones(4, dtype=np.float64)
    assert digest(ones_i64) != digest(ones_f64)
    # int32 vs int64 with identical logical values must not collide either.
    assert digest(np.ones(4, dtype=np.int32)) != digest(ones_i64)


def test_shape_matters():
    flat = np.arange(6)
    assert digest(flat) != digest(flat.reshape(2, 3))
    assert digest(flat.reshape(2, 3)) != digest(flat.reshape(3, 2))


def test_memory_layout_does_not_fragment_the_cache():
    """C-order and F-order arrays with the same semantics share a digest."""
    c_order = np.arange(12).reshape(3, 4)
    f_order = np.asfortranarray(c_order)
    assert digest(c_order) == digest(f_order)
    view = c_order[:, ::2]  # non-contiguous view
    materialized = np.ascontiguousarray(view)
    assert digest(view) == digest(materialized)


def test_numpy_scalars_hash_by_semantic_value():
    assert digest(np.int64(7)) == digest(7)
    assert digest(np.float64(1.5)) == digest(1.5)
    assert digest(np.int64(1)) != digest(np.float64(1.0))  # int vs float stays distinct


def test_arrays_nested_in_containers():
    a = {"data": np.arange(10), "tag": "x"}
    b = {"data": np.arange(10), "tag": "x"}
    c = {"data": np.arange(11), "tag": "x"}
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)


def test_equal_dataframes_share_a_digest():
    make = lambda: pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    assert digest(make()) == digest(make())


def test_dataframe_values_matter():
    base = pd.DataFrame({"a": [1, 2, 3]})
    changed = pd.DataFrame({"a": [1, 2, 4]})
    assert digest(base) != digest(changed)


def test_dataframe_column_names_matter():
    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"b": [1, 2]})
    assert digest(left) != digest(right)


def test_dataframe_index_matters():
    base = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
    reindexed = pd.DataFrame({"a": [1, 2]}, index=[5, 6])
    assert digest(base) != digest(reindexed)


def test_series_name_and_values():
    assert digest(pd.Series([1, 2], name="x")) == digest(pd.Series([1, 2], name="x"))
    assert digest(pd.Series([1, 2], name="x")) != digest(pd.Series([1, 2], name="y"))
    assert digest(pd.Series([1, 2])) != digest(pd.Series([2, 1]))


def test_dataframe_and_series_do_not_collide_with_plain_containers():
    frame = pd.DataFrame({"a": [1]})
    assert digest(frame) != digest({"a": [1]})


def test_end_to_end_caching_with_dataframe_arguments():
    calls = []

    @cache
    def features(df):
        calls.append(len(df))
        return df["a"].sum()

    df = pd.DataFrame({"a": [1, 2, 3]})
    assert features(df) == 6
    assert features(pd.DataFrame({"a": [1, 2, 3]})) == 6  # equal frame: HIT
    assert features(pd.DataFrame({"a": [9]})) == 9
    assert calls == [3, 1]


def test_object_dtype_dataframe_content_is_hashed():
    left = pd.DataFrame({"s": ["abc", "def"]})
    right = pd.DataFrame({"s": ["abc", "xyz"]})
    assert digest(left) != digest(right)
