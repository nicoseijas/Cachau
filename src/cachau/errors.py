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


class CacheVerificationWarning(UserWarning):
    """A verified HIT did not match a fresh recompute; the fresh value won.

    Emitted by ``verify=``: either something the fingerprint cannot see changed
    (a module-level helper, an undeclared external input) or the function is
    nondeterministic. The cache never serves the mismatched value — it is
    replaced and the call returns the fresh result.
    """


class MachineCodeCacheWarning(UserWarning):
    """The wrapped dispatcher also carries numba's on-disk machine-code cache.

    ``@cache`` over ``@njit(cache=True)`` stacks two caches. Cachau's result
    cache already survives restarts, and numba's ``.nbi``/``.nbc`` cache is a
    known cross-process crash hazard on some platforms — prefer ``persist=``
    INSTEAD of ``cache=True``, not on top of it.
    """
