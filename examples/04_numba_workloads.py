"""Numba workloads: caching at the dispatcher boundary, honest JIT metrics.

Requires numba (pip install numba). @cache goes ABOVE @njit: cachau caches
results; numba's cache=True caches machine code; they compose.
"""

import sys
import time

try:
    import numpy as np
    from numba import njit
except ImportError:
    sys.exit("this example needs numba: pip install numba")

from cachau import cache


@cache(max_memory="1GB")
@njit
def pairwise_energy(points):
    total = 0.0
    n = points.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - points[j, k]
                d += diff * diff
            total += 1.0 / (d + 1e-9)
    return total


def timed(label, fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    print(f"  {label}: {time.perf_counter() - start:.3f}s")
    return result


def main() -> None:
    rng = np.random.default_rng(seed=7)
    points = rng.random((4000, 3))

    print("first call (cold: includes one-time JIT compilation):")
    timed("cold compute", pairwise_energy, points)

    print("new argument (warm compute: compiled code, no compilation):")
    warm_points = rng.random((4000, 3))
    timed("warm compute", pairwise_energy, warm_points)

    print("repeated argument (cachau HIT: no execution at all):")
    timed("cache hit   ", pairwise_energy, warm_points)

    stats = pairwise_energy.cache.stats()
    print(
        f"\ncold JIT time (reported separately, never counted as normal "
        f"execution): {stats.cold_compute_seconds:.3f}s"
    )
    print(f"total compute: {stats.total_compute_seconds:.3f}s")
    print(
        f"estimated saved by caching: {stats.estimated_saved_seconds:.3f}s "
        f"(based on WARM computes only - compile time never inflates it)"
    )

    # Changing a semantically relevant compile option (fastmath, parallel,
    # locals=...) changes the cache identity: results computed under different
    # numeric semantics are never mixed. See tests/test_numba.py for the
    # full matrix, and tests/test_numba_utils_compat.py for composition with
    # numba-utils decorator aliases.


if __name__ == "__main__":
    main()
