"""``func.cache.profile(...)`` - is caching this call even worth it?

Cache economics (GUIDELINES.md §15): a cache helps only when

    T_key + T_lookup + T_deserialize  <  T_recompute

For a warm Numba kernel that runs in 50 ms, hashing a 2 GB array to build the
key can make caching *slower* than just recomputing. ``profile()`` measures both
sides of that inequality for one concrete call and says which wins, what the
dominant cost is, and what to do about it.

Unlike ``explain()``, profiling is NOT pure observation: to measure recompute
cost it must actually run the function (warmed up first, so JIT compilation is
never counted as normal execution - the same discipline as the benchmark suite).
It does not touch ``stats()`` counters, and it leaves cache state as it found it:
if the entry was absent it is measured against a throwaway copy and removed
again, so a call with a TTL or dependencies is never left with a bare entry.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from time import perf_counter as _perf_counter
from typing import Any, Callable

from cachau.explanation import _format_bytes

# Ratio of warm-recompute time to cache-hit time. Above HIGH the cache clearly
# pays; below LOW it clearly does not; between them the two are close enough
# that measurement noise dominates and the honest answer is "about break-even".
_WORTH_IT_RATIO = 1.2
_NOT_WORTH_RATIO = 0.8


@dataclass(frozen=True)
class CacheProfile:
    """A measured verdict on whether caching one call beats recomputing it."""

    namespace: str
    key: str
    repeats: int
    compute_seconds: float  # warm recompute (JIT already compiled)
    key_seconds: float  # building the cache key
    read_seconds: float  # backend lookup + deserialization of the stored value
    size_bytes: int | None
    # Filled by the diagnosis below.
    verdict: str  # "worth_it" | "marginal" | "not_worth_it"
    primary_cost: str  # human label for the dominant cache-hit cost
    recommendation: str
    # Same-package module-level functions this call reaches by global lookup
    # that neither the code fingerprint nor a declared cachau.code() covers —
    # editing one of them would NOT invalidate cached results.
    unfingerprinted_calls: tuple[str, ...] = ()

    @property
    def hit_seconds(self) -> float:
        """Total cost of serving this call from cache."""
        return self.key_seconds + self.read_seconds

    @property
    def speedup(self) -> float:
        """How many times faster a cache hit is than recomputing (``>1`` = worth it)."""
        hit = self.hit_seconds
        return self.compute_seconds / hit if hit > 0 else float("inf")

    @property
    def saved_seconds(self) -> float:
        """Wall-clock saved per hit (negative if caching is slower)."""
        return self.compute_seconds - self.hit_seconds

    @property
    def worth_it(self) -> bool:
        return self.verdict == "worth_it"

    def __str__(self) -> str:
        rows = [
            ("Warm recompute", self.compute_seconds),
            ("Key generation", self.key_seconds),
            ("Cache read", self.read_seconds),
        ]
        width = 22
        lines = [f"Cache economics: {self.namespace}", ""]
        for label, seconds in rows:
            lines.append(f"{(label + ':').ljust(width)}{_format_seconds(seconds):>10}")
        lines.append("-" * (width + 10))
        lines.append(
            f"{'Cache hit total:'.ljust(width)}{_format_seconds(self.hit_seconds):>10}"
        )
        lines.append("")
        if self.verdict == "worth_it":
            lines.append(
                f"Caching saves ~{_format_seconds(self.saved_seconds)} per hit "
                f"({self.speedup:.1f}x faster). Worth it."
            )
        elif self.verdict == "not_worth_it":
            lines.append(
                f"Caching is slower than recompute by {1 / self.speedup:.1f}x."
            )
        else:
            lines.append("Caching and recomputation are about break-even.")
        lines.append(f"{'Primary hit cost:'.ljust(width)}{self.primary_cost}")
        lines.append(f"{'Recommendation:'.ljust(width)}{self.recommendation}")
        if self.unfingerprinted_calls:
            names = ", ".join(self.unfingerprinted_calls)
            lines.append("")
            lines.append(
                f"Warning: calls {names} by global lookup; the code fingerprint "
                "does not cover them, so editing them will NOT invalidate cached "
                "results. Declare them: depends_on=[cachau.code(...)]"
            )
        return "\n".join(lines)


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def measure(fn: Callable[[], Any], repeats: int) -> float:
    """Run ``fn`` ``repeats`` times and return the fastest run.

    The minimum is the cleanest estimator of intrinsic cost: it is the run least
    disturbed by scheduler preemption, GC pauses, or a cold cache line - noise
    can only ever make a run slower, never faster.
    """
    best = float("inf")
    for _ in range(repeats):
        start = _perf_counter()
        fn()
        best = min(best, _perf_counter() - start)
    return best


def largest_data_arg(bound: tuple[tuple[str, Any], ...]) -> tuple[str, int] | None:
    """Name and byte size of the largest ndarray/DataFrame argument, if any.

    This is the usual culprit when key generation dominates: hashing a big array
    costs real time on every lookup, and the fix (an explicit ``key=`` or version
    token) sidesteps it. Only inspects libraries already imported - cachau never
    triggers an import to profile.
    """
    np = sys.modules.get("numpy")
    pd = sys.modules.get("pandas")
    best: tuple[str, int] | None = None
    for _, value in bound:
        label: str | None = None
        nbytes = 0
        if np is not None and isinstance(value, np.ndarray):
            label = f"ndarray[{value.dtype}, {_format_bytes(value.nbytes)}]"
            nbytes = int(value.nbytes)
        elif pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
            nbytes = int(value.memory_usage(deep=True).sum()) if isinstance(
                value, pd.DataFrame
            ) else int(value.memory_usage(deep=True))
            label = f"{type(value).__name__}[{_format_bytes(nbytes)}]"
        if label is not None and (best is None or nbytes > best[1]):
            best = (label, nbytes)
    return best


def diagnose(
    *,
    compute_seconds: float,
    key_seconds: float,
    read_seconds: float,
    largest_data: tuple[str, int] | None,
) -> tuple[str, str, str]:
    """Classify the trade-off and return (verdict, primary_cost, recommendation)."""
    hit = key_seconds + read_seconds
    ratio = compute_seconds / hit if hit > 0 else float("inf")
    key_dominates = key_seconds >= read_seconds
    primary_cost = "key generation" if key_dominates else "cache read (lookup + deserialize)"

    if ratio >= _WORTH_IT_RATIO:
        verdict = "worth_it"
        if key_dominates and largest_data is not None:
            recommendation = (
                f"Worth caching. Key generation dominates the hit "
                f"(hashing {largest_data[0]}); an explicit key= or dataset "
                f"version would make hits even cheaper."
            )
        else:
            recommendation = "Worth caching - keep it."
        return verdict, primary_cost, recommendation

    if ratio <= _NOT_WORTH_RATIO:
        verdict = "not_worth_it"
    else:
        verdict = "marginal"

    if key_dominates and largest_data is not None:
        primary_cost = f"hashing {largest_data[0]}"
        recommendation = (
            "Provide an explicit stable key= (e.g. a dataset version) so the "
            "payload isn't hashed on every lookup - that is the whole cost here."
        )
    elif read_seconds >= key_seconds and read_seconds > compute_seconds:
        recommendation = (
            "Recompute is faster than reading the cached value back. This "
            "function is cheap to recompute - consider not caching it."
        )
    else:
        recommendation = (
            "Recompute is nearly as cheap as a cache hit; the cache adds little. "
            "Consider dropping it, or reduce hit cost with an explicit key=."
        )
    return verdict, primary_cost, recommendation
