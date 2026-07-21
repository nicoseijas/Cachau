"""Cachau — delightful function caching for Python data workloads."""

from cachau.decorator import cache
from cachau.dependencies import env, file, package, token
from cachau.explanation import Explanation
from cachau.profile import CacheProfile
from cachau.stats import CacheStats
from cachau.errors import (
    CachauError,
    ConfigurationError,
    InvalidSizeError,
    InvalidTTLError,
    UnhashableArgumentError,
)

__version__ = "0.3.2"
__all__ = [
    "cache",
    "file",
    "env",
    "package",
    "token",
    "CacheStats",
    "CacheProfile",
    "CachauError",
    "Explanation",
    "ConfigurationError",
    "InvalidSizeError",
    "InvalidTTLError",
    "UnhashableArgumentError",
    "__version__",
]
