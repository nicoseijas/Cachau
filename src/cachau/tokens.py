"""``array_token()``: hash a big immutable value once, reuse the digest.

``profile()`` regularly names the same culprit: key generation dominated by
content-hashing a large array argument on every call. When that argument is
immutable for the whole run (a lookup table, a canonical index map), the honest
fix is to pay the hash once. ``array_token`` does exactly that — a content
digest memoized per live object — for use inside an explicit ``key=``::

    @cache(key=lambda table, n: (cachau.array_token(table), n))
    def lookup(table, n): ...

The memo is identity-SAFE, not identity-keyed: a plain ``id()``-keyed cache
would be a false-HIT bug (the object dies, a new one reuses its id, the stale
digest describes the wrong content). Every lookup re-checks that the weakly
referenced object is still the very object being tokenized, and entries are
pruned when their object dies.

The one contract the caller owns: the value must not be MUTATED in place while
the token is in use. The memo hashes content once per object — an in-place
write afterwards leaves the token describing the old content, which is
precisely the stale-key situation ``key=`` escapes hatch out of. For values
that change, let cachau content-hash them (the default) or version them
explicitly.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

from cachau.keys import _digest_value

# id(obj) -> (weakref to obj, digest). The id key is only a fast index; the
# weakref check is what makes it correct.
_TOKEN_MEMO: dict[int, tuple[Any, str]] = {}
_MEMO_GUARD = threading.Lock()


def array_token(value: Any) -> str:
    """The content digest of ``value``, computed once per live object.

    Named for its primary use (large ndarrays), but works for any value cachau
    can hash. Values that cannot be weakly referenced (ints, strings, tuples)
    are digested on every call — correct, just unmemoized; they are cheap to
    hash anyway. Raises ``UnhashableArgumentError`` for values cachau has no
    hashing support for, same as passing them as arguments would.
    """
    try:
        memo_key = id(value)
        with _MEMO_GUARD:
            hit = _TOKEN_MEMO.get(memo_key)
            if hit is not None and hit[0]() is value:
                return hit[1]
        digest = _digest_value(value).hex()
        reference = weakref.ref(value, _make_pruner(memo_key))
    except TypeError:
        # Not weakref-able: no safe way to memoize, so hash every time.
        return _digest_value(value).hex()
    with _MEMO_GUARD:
        _TOKEN_MEMO[memo_key] = (reference, digest)
    return digest


def _make_pruner(memo_key: int) -> Any:
    def prune(dead_reference: Any) -> None:
        # Pop only our own entry: by the time this callback runs, the id may
        # already belong to a NEW object with a fresh memo entry — deleting
        # that one would only cost a re-hash, but there is no reason to.
        with _MEMO_GUARD:
            current = _TOKEN_MEMO.get(memo_key)
            if current is not None and current[0] is dead_reference:
                del _TOKEN_MEMO[memo_key]

    return prune
