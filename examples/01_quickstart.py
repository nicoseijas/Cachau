"""Quickstart: persistent, bounded caching in one decorator.

Run this script TWICE:

    python examples/01_quickstart.py   # computes (~2s), commits to ./.cachau
    python examples/01_quickstart.py   # instant: served from disk

Delete ./.cachau to start over.
"""

import time

from cachau import cache


@cache(persist=True, ttl="1h", max_memory="100MB")
def slow_analysis(day: str) -> dict:
    print(f"  computing analysis for {day} (expensive, ~1s) ...")
    time.sleep(1.0)
    return {"day": day, "mean": 42.0, "records": 10_000}


def main() -> None:
    started = time.perf_counter()
    print("first call:")
    print(" ", slow_analysis("2026-07-19"))
    print("second call (same arguments):")
    print(" ", slow_analysis("2026-07-19"))
    elapsed = time.perf_counter() - started

    stats = slow_analysis.cache.stats()
    print(f"\ntotal wall time: {elapsed:.2f}s")
    print(f"hits={stats.hits} misses={stats.misses} hit_rate={stats.hit_rate:.0%}")
    print(f"estimated time saved: {stats.estimated_saved_seconds:.2f}s")
    print("\nwhy was that a hit?  ->  slow_analysis.cache.explain(...)")
    print(slow_analysis.cache.explain("2026-07-19"))
    print("\nre-run this script: the first call will be a HIT from ./.cachau")


if __name__ == "__main__":
    main()
