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

import weakref
from typing import Any

from cachau.keys import _digest_value

# id(obj) -> (weakref to obj, digest). The id key is only a fast index; the
# weakref check on every lookup is what makes it correct.
#
# Deliberately LOCK-FREE. A weakref callback can run inside any allocation
# that triggers a cyclic GC — including one made while a lock protecting this
# dict is held, in the same thread — so a non-reentrant lock shared between
# lookups and the pruning callback is a self-deadlock waiting for the right
# allocation. Individual dict operations are atomic, and every path tolerates
# racing: a hit is only trusted after re-verifying object identity, a lost
# insert or an over-eager prune costs one re-hash, never a wrong digest.
_TOKEN_MEMO: dict[int, tuple[Any, str]] = {}


def array_token(value: Any) -> str:
    """The content digest of ``value``, computed once per live object.

    Named for its primary use (large ndarrays), but works for any value cachau
    can hash. Values that cannot be weakly referenced (ints, strings, tuples)
    are digested on every call — correct, just unmemoized; they are cheap to
    hash anyway. Raises ``UnhashableArgumentError`` for values cachau has no
    hashing support for, same as passing them as arguments would.
    """
    memo_key = id(value)
    hit = _TOKEN_MEMO.get(memo_key)
    if hit is not None and hit[0]() is value:
        return hit[1]
    digest = _digest_value(value).hex()
    try:
        reference = weakref.ref(value, _make_pruner(memo_key))
    except TypeError:
        return digest  # not weakref-able: no safe way to memoize
    _TOKEN_MEMO[memo_key] = (reference, digest)
    return digest


def _make_pruner(memo_key: int) -> Any:
    def prune(dead_reference: Any) -> None:
        # Delete only our own entry: by the time this callback runs, the id
        # may already belong to a NEW object with a fresh memo entry. The
        # get/del pair is not atomic; losing that race deletes a successor's
        # entry, which costs its next caller one re-hash and nothing else.
        current = _TOKEN_MEMO.get(memo_key)
        if current is not None and current[0] is dead_reference:
            try:
                del _TOKEN_MEMO[memo_key]
            except KeyError:
                pass

    return prune
