"""Cachau error types."""


class CachauError(Exception):
    """Base class for all cachau errors."""


class UnhashableArgumentError(CachauError, TypeError):
    """An argument has no cachau hashing support and no explicit key was given.

    Raised loudly instead of silently ignoring the argument: an ignored argument
    would make semantically different calls share a cache entry (a false HIT).
    """
