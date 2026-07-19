"""Key building: normalization and native hashing (GUIDELINES.md §2)."""

import dataclasses
import pathlib

import pytest

from cachau.errors import UnhashableArgumentError
from cachau.keys import digest_arguments, normalize_call


def sample(a, b=2, *, c=3):
    return a + b + c


def test_kwargs_and_positional_share_a_key():
    assert digest_arguments(sample, (1, 2), {}) == digest_arguments(
        sample, (), {"a": 1, "b": 2}
    )


def test_defaults_are_applied():
    assert digest_arguments(sample, (1,), {}) == digest_arguments(
        sample, (1, 2), {"c": 3}
    )


def test_different_arguments_differ():
    assert digest_arguments(sample, (1,), {}) != digest_arguments(sample, (2,), {})


def test_normalize_call_binds_signature():
    normalized = normalize_call(sample, (1,), {"c": 9})
    assert normalized == (("a", 1), ("b", 2), ("c", 9))


def test_dict_key_order_is_irrelevant():
    assert digest_arguments(sample, ({"x": 1, "y": 2},), {}) == digest_arguments(
        sample, ({"y": 2, "x": 1},), {}
    )


def test_equal_value_different_type_differ():
    # 1 == 1.0 == True in Python; their cache identities must not collide.
    digests = {
        digest_arguments(sample, (1,), {}),
        digest_arguments(sample, (1.0,), {}),
        digest_arguments(sample, (True,), {}),
    }
    assert len(digests) == 3


def test_supported_containers_and_types():
    @dataclasses.dataclass(frozen=True)
    class Config:
        name: str
        depth: int

    value = {
        "path": pathlib.PurePosixPath("data/train.parquet"),
        "config": Config("features", 3),
        "items": [1, (2, 3), None, b"raw"],
    }
    first = digest_arguments(sample, (value,), {})
    second = digest_arguments(sample, (value,), {})
    assert first == second


def test_unhashable_argument_fails_loudly():
    class Opaque:
        pass

    with pytest.raises(UnhashableArgumentError, match="a"):
        digest_arguments(sample, (Opaque(),), {})
