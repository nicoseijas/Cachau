"""Size parsing and in-memory size estimation."""

import dataclasses

import pytest

from cachau.errors import InvalidSizeError
from cachau.sizes import estimate_size, parse_size


def test_none_means_no_limit():
    assert parse_size(None) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1024, 1024),
        ("512B", 512),
        ("2KB", 2 * 1024),
        ("100MB", 100 * 1024**2),
        ("1GB", 1024**3),
        ("1.5GB", int(1.5 * 1024**3)),
        ("2TB", 2 * 1024**4),
        ("1gb", 1024**3),
    ],
)
def test_accepted_values(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "100", "10XB", "GB", "-1GB", "0MB", 0, -5, True, [1024], 1.5],
)
def test_rejected_values_fail_fast(value):
    with pytest.raises(InvalidSizeError):
        parse_size(value)


def test_size_grows_with_content():
    assert estimate_size(b"x" * 10_000) > estimate_size(b"x" * 10)
    assert estimate_size("y" * 10_000) > estimate_size("y" * 10)
    assert estimate_size(list(range(1000))) > estimate_size(list(range(10)))


def test_containers_count_their_children():
    payload = b"z" * 100_000
    other = b"y" * 100_000
    assert estimate_size([payload]) >= estimate_size(payload)
    assert estimate_size({"k": payload}) >= estimate_size(payload)
    assert estimate_size((payload, other)) >= 2 * len(payload)


def test_shared_references_counted_once():
    payload = b"z" * 100_000
    assert estimate_size([payload, payload]) < 2 * estimate_size(payload)


def test_self_referential_container_terminates():
    loop = []
    loop.append(loop)
    assert estimate_size(loop) > 0


def test_nbytes_duck_typing_for_arrays():
    class FakeArray:
        nbytes = 4_000_000

    assert estimate_size(FakeArray()) >= 4_000_000


def test_dataclass_fields_are_counted():
    @dataclasses.dataclass(frozen=True)
    class Result:
        blob: bytes

    payload = b"z" * 100_000
    assert estimate_size(Result(payload)) >= len(payload)


def test_arbitrary_object_gets_a_nonzero_estimate():
    class Opaque:
        pass

    assert estimate_size(Opaque()) > 0
