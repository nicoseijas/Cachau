"""Numba boundary: compile -> warm up -> benchmark; cold JIT reported apart.

The methodology GUIDELINES.md §14 mandates: never count one-time compilation
as normal execution cost. This benchmark shows all four numbers a Numba user
needs: cold JIT, warm execution, cachau HIT, and what stats() reports.
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _timing import fmt, measure, measure_once, print_environment

try:
    import numpy as np
    from numba import njit
except ImportError:
    sys.exit("this benchmark needs numba: pip install numba")

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


def main() -> None:
    print("Numba workload: cold JIT vs warm vs cachau HIT\n")
    print_environment()

    rng = np.random.default_rng(seed=7)
    first = rng.random((1500, 3))

    # 1. COLD: one-time compilation + first execution (single run, labeled).
    cold = measure_once(lambda: pairwise_energy(first))
    print(f"  cold (JIT compile + execute, ONE-TIME)  {fmt(cold)}")

    # 2. WARM: compiled code on fresh arguments (median; every call is a
    #    cachau miss because the content differs).
    warm = measure(lambda: pairwise_energy(rng.random((1500, 3))), repeats=5)
    print(f"  warm compute (median, fresh args)       {fmt(warm)}")

    # 3. HIT: same argument, no execution at all (keying an ~36 KB array
    #    + lookup).
    hit = measure(lambda: pairwise_energy(first))
    print(f"  cachau HIT (keying + lookup)            {fmt(hit)}   (~{warm / hit:,.0f}x faster than warm)")

    stats = pairwise_energy.cache.stats()
    print(
        f"\n  stats() agrees: cold_compute_seconds={stats.cold_compute_seconds:.3f}s "
        f"(excluded from savings), estimated_saved={stats.estimated_saved_seconds:.3f}s"
    )
    print(
        "\nreading: cache economics hold when hit cost << warm compute. For\n"
        "kernels FASTER than the hit row, caching loses - measure, don't assume."
    )


if __name__ == "__main__":
    main()
