"""Cachau error types."""


class CachauError(Exception):
    """Base class for all cachau errors."""


class InvalidTTLError(CachauError, ValueError):
    """A ttl value could not be interpreted as a positive duration."""


class InvalidSizeError(CachauError, ValueError):
    """A max_memory value could not be interpreted as a positive byte size."""


class ConfigurationError(CachauError, ValueError):
    """Mutually incompatible cache options were combined."""


class UnhashableArgumentError(CachauError, TypeError):
    """An argument has no cachau hashing support and no explicit key was given.

    Raised loudly instead of silently ignoring the argument: an ignored argument
    would make semantically different calls share a cache entry (a false HIT).
    """
