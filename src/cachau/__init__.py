"""Cachau — delightful function caching for Python data workloads."""

from cachau.decorator import cache
from cachau.dependencies import code, env, file, package, token
from cachau.explanation import Explanation
from cachau.inspection import CacheEntryView, Inspection
from cachau.profile import CacheProfile
from cachau.stats import CacheStats
from cachau.errors import (
    CachauError,
    CacheVerificationWarning,
    ConfigurationError,
    InvalidSizeError,
    InvalidTTLError,
    UnhashableArgumentError,
)

__version__ = "0.4.0"
__all__ = [
    "cache",
    "file",
    "env",
    "package",
    "token",
    "code",
    "CacheStats",
    "CacheProfile",
    "CacheEntryView",
    "Inspection",
    "CachauError",
    "CacheVerificationWarning",
    "Explanation",
    "ConfigurationError",
    "InvalidSizeError",
    "InvalidTTLError",
    "UnhashableArgumentError",
    "__version__",
]
