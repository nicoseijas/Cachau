"""TTL value parsing: simple, readable, validated at the boundary."""

import pytest

from cachau.durations import parse_ttl
from cachau.errors import InvalidTTLError


def test_none_means_no_ttl():
    assert parse_ttl(None) is None


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        (60, 60.0),
        (0.5, 0.5),
        ("30s", 30.0),
        ("10m", 600.0),
        ("2h", 7200.0),
        ("7d", 604800.0),
        ("1.5h", 5400.0),
    ],
)
def test_accepted_values(value, seconds):
    assert parse_ttl(value) == seconds


@pytest.mark.parametrize(
    "value",
    ["", "10", "10x", "s", "-30s", "m10", 0, -5, "0s", True, [60]],
)
def test_rejected_values_fail_fast(value):
    with pytest.raises(InvalidTTLError):
        parse_ttl(value)
