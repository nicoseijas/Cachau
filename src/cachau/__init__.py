"""Cachau — delightful function caching for Python data workloads."""

from cachau.decorator import cache
from cachau.errors import CachauError, UnhashableArgumentError

__version__ = "0.0.1"
__all__ = ["cache", "CachauError", "UnhashableArgumentError", "__version__"]
