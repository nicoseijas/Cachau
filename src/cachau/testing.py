"""Test-suite assertions that certify a cache's contract.

``explain()`` observes; these certify. The discipline behind both helpers is
that a check that cannot fail certifies nothing: ``assert_cache_faithful``
fails when a HIT would serve anything other than a fresh compute (the "false
green" failure class), and ``assert_invalidates`` fails when a perturbation
that should kill an entry does not.

Both execute the cached function (to prime a cold entry, and to obtain the
fresh reference value), so they belong in tests, not on hot paths.
"""

from __future__ import annotations

from typing import Any, Callable

from cachau.decorator import _values_match
from cachau.explanation import Explanation


def assert_cache_faithful(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Assert that a HIT for these arguments serves exactly a fresh compute.

    Primes the entry if nothing is cached yet, reads the CACHED value straight
    from the backend (never through the call path, so ``verify=`` resampling
    cannot mask a divergence), computes once more outside the cache, and
    compares by content — the same type-tagged digest that keys arguments, so
    ndarrays and DataFrames compare bit-exactly, with pickle-bytes fallback.

    Fails when the values diverge (a nondeterministic function, a mutated
    cached object, a corrupted store) and fails loudly when there is no HIT to
    certify at all — a broken store that never caches must not pass a
    faithfulness test vacuously.
    """
    control = _control_of(func)
    explanation = control.explain(*args, **kwargs)
    if explanation.outcome != "HIT":
        func(*args, **kwargs)  # prime a cold entry
        explanation = control.explain(*args, **kwargs)
    if explanation.outcome != "HIT":
        raise AssertionError(
            f"nothing cached to certify: after priming, explain() still "
            f"reports a MISS for {control.namespace}:\n{explanation}"
        )
    peek = getattr(control._backend, "peek", control._backend.get)
    entry = peek(control._key_builder(*args, **kwargs))
    if entry is None:  # raced away between explain() and the read
        raise AssertionError(
            f"nothing cached to certify: the entry for {control.namespace} "
            f"disappeared between explain() and the backend read"
        )
    fresh = _unwrapped(func)(*args, **kwargs)
    compared, matches = _values_match(entry.value, fresh)
    if not compared:
        raise AssertionError(
            f"cannot certify {control.namespace}: values of type "
            f"{type(fresh).__qualname__!r} are neither digestible nor "
            f"picklable, so cached and fresh cannot be compared"
        )
    if not matches:
        raise AssertionError(
            f"cache is not faithful for {control.namespace}: the cached value "
            f"differs from a fresh compute. Either the function is "
            f"nondeterministic, an undeclared input changed, or the stored "
            f"value was mutated.\ncached: {_preview(entry.value)}"
            f"\nfresh:  {_preview(fresh)}"
        )


def assert_invalidates(
    func: Callable[..., Any],
    perturb: Callable[[], Any],
    *args: Any,
    reason: str | None = None,
    **kwargs: Any,
) -> Explanation:
    """Assert that ``perturb()`` turns a cached entry into a MISS.

    Primes the entry (and fails loudly if nothing caches — an invalidation
    check against an empty cache certifies nothing), applies the perturbation,
    and asserts the next ``explain()`` reports a MISS — with ``reason=`` also
    asserting WHY (``"dependency_changed"``, ``"expired"``, ...). Returns the
    resulting :class:`Explanation` for further assertions. The perturbation is
    not undone; restore state in your test's teardown if it matters.
    """
    control = _control_of(func)
    explanation = control.explain(*args, **kwargs)
    if explanation.outcome != "HIT":
        func(*args, **kwargs)
        explanation = control.explain(*args, **kwargs)
    if explanation.outcome != "HIT":
        raise AssertionError(
            f"cannot certify invalidation: nothing cached for "
            f"{control.namespace} even after priming:\n{explanation}"
        )
    perturb()
    explanation = control.explain(*args, **kwargs)
    if explanation.outcome != "MISS":
        raise AssertionError(
            f"the perturbation did not invalidate {control.namespace}: "
            f"explain() still reports a HIT, so this check certifies nothing"
        )
    if reason is not None and explanation.reason != reason:
        raise AssertionError(
            f"{control.namespace} was invalidated, but for the wrong reason: "
            f"expected {reason!r}, got {explanation.reason!r}"
        )
    return explanation


def _control_of(func: Callable[..., Any]) -> Any:
    control = getattr(func, "cache", None)
    if control is None or not hasattr(control, "explain"):
        raise TypeError(
            f"{getattr(func, '__qualname__', func)!r} is not a cachau-cached "
            f"function (no .cache control surface)"
        )
    return control


def _unwrapped(func: Callable[..., Any]) -> Callable[..., Any]:
    return getattr(func, "__wrapped__", func)


def _preview(value: Any, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
