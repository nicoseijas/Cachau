"""Cachau — delightful function caching for Python data workloads."""

from cachau.decorator import cache
from cachau.stats import CacheStats
from cachau.errors import (
    CachauError,
    ConfigurationError,
    InvalidSizeError,
    InvalidTTLError,
    UnhashableArgumentError,
)

__version__ = "0.0.1"
__all__ = [
    "cache",
    "CacheStats",
    "CachauError",
    "ConfigurationError",
    "InvalidSizeError",
    "InvalidTTLError",
    "UnhashableArgumentError",
    "__version__",
]
