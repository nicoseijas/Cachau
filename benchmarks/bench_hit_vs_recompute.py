"""Hit cost vs. recompute cost: when caching pays, and when it loses.

Measures the full hit path (keying + lookup + deserialize where relevant)
against honest recompute times, for both backends — including the case the
docs warn about: a fast function with a huge argument, where caching is a
net LOSS.
"""

import sys
import tempfile
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _timing import fmt, measure, print_environment

from cachau import cache


def busy_wait(seconds: float) -> None:
    # time.sleep undersells recompute cost at ms scale on Windows; spin.
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def main() -> None:
    print("Full HIT path vs recompute\n")
    print_environment()

    # --- 1. memory backend, scalar key, 50 ms function ----------------------
    @cache
    def expensive_scalar(n: int) -> int:
        busy_wait(0.050)
        return n * n

    expensive_scalar(7)  # commit
    hit = measure(lambda: expensive_scalar(7))
    print(f"  50 ms function, int arg,  memory HIT   {fmt(hit)}   (speedup ~{0.050 / hit:,.0f}x)")

    try:
        import numpy as np
    except ImportError:
        print("\n(numpy not installed - array cases skipped)")
        return

    # --- 2. memory backend, 8 MB array key, 50 ms function ------------------
    arr = np.arange(1_000_000, dtype=np.float64)

    @cache
    def expensive_array(values) -> float:
        busy_wait(0.050)
        return float(values.sum())

    expensive_array(arr)
    hit = measure(lambda: expensive_array(arr))
    print(f"  50 ms function, 8 MB arg, memory HIT   {fmt(hit)}   (speedup ~{0.050 / hit:.0f}x)")

    # --- 3. disk backend: hit includes read + unpickle of an 8 MB result ----
    with tempfile.TemporaryDirectory() as d:

        @cache(persist=d)
        def expensive_persisted(n: int):
            busy_wait(0.050)
            return np.arange(1_000_000, dtype=np.float64)  # 8 MB result

        expensive_persisted(1)
        hit = measure(lambda: expensive_persisted(1))
        print(f"  50 ms function, 8 MB result, disk HIT {fmt(hit)}   (speedup ~{0.050 / hit:.0f}x)")

    # --- 4. the honest loss case: fast function, 80 MB argument -------------
    big = np.arange(10_000_000, dtype=np.float64)

    def cheap(values) -> float:
        return float(values[0] + values[-1])

    recompute = measure(lambda: cheap(big))
    cached_cheap = cache(cheap)
    cached_cheap(big)
    hit = measure(lambda: cached_cheap(big))
    print(f"\n  CHEAP function (~{fmt(recompute).strip()}), 80 MB argument:")
    print(f"    recompute                            {fmt(recompute)}")
    print(f"    cache HIT (dominated by keying)      {fmt(hit)}")
    verdict = "LOSS" if hit > recompute else "win"
    print(
        f"    -> caching is a {verdict} here ({hit / recompute:,.0f}x slower). "
        f"Use key= with a cheap version tag instead."
    )


if __name__ == "__main__":
    main()
