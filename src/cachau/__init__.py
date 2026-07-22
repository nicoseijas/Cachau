"""Cachau — delightful function caching for Python data workloads."""

from cachau.decorator import cache
from cachau.dependencies import code, env, file, package, token
from cachau.explanation import Explanation
from cachau.inspection import CacheEntryView, Inspection
from cachau.profile import CacheProfile
from cachau.stats import CacheStats
from cachau.tokens import array_token
from cachau.errors import (
    CachauError,
    CacheVerificationWarning,
    ConfigurationError,
    InvalidSizeError,
    InvalidTTLError,
    MachineCodeCacheWarning,
    UnhashableArgumentError,
)

__version__ = "0.6.1"
__all__ = [
    "cache",
    "file",
    "env",
    "package",
    "token",
    "code",
    "array_token",
    "CacheStats",
    "CacheProfile",
    "CacheEntryView",
    "Inspection",
    "CachauError",
    "CacheVerificationWarning",
    "MachineCodeCacheWarning",
    "Explanation",
    "ConfigurationError",
    "InvalidSizeError",
    "InvalidTTLError",
    "UnhashableArgumentError",
    "__version__",
]
