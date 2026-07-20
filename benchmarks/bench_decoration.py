"""Decoration-time cost of a persistent cache.

Import time is not free for ``persist=``: decorating reads every entry header
in the cache directory once, to purge superseded fingerprints and — since
v0.3.0 — to rehydrate the LRU budget so ``max_memory`` still holds after a
restart. That scan is paid once per decoration, i.e. once per import, but it
is paid on the user's startup path, so it deserves a number rather than a
promise.

The question this answers: does it scale with the NUMBER of entries (headers)
or with the SIZE of the payloads (bodies)? Only the first is acceptable.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _timing import fmt, measure, print_environment

from cachau import cache


def _seed(directory: str, entries: int, payload_bytes: int) -> None:
    """Populate a cache directory with ``entries`` committed results."""
    blob = b"x" * payload_bytes

    @cache(persist=directory)
    def seeded(n: int) -> bytes:
        return blob

    for n in range(entries):
        seeded(n)


def _decorate(directory: str, max_memory: str | None) -> None:
    @cache(persist=directory, max_memory=max_memory)
    def target(n: int) -> bytes:
        return b""


def main() -> None:
    print("Decoration cost of persist= (paid once per import)\n")
    print_environment()

    print("  scaling with entry COUNT (1 KB payloads, max_memory=64mb):")
    for entries in (10, 100, 1_000, 5_000):
        with tempfile.TemporaryDirectory() as d:
            _seed(d, entries, 1_024)
            cost = measure(lambda: _decorate(d, "64mb"), repeats=5, warmup=1)
            print(f"    {entries:>5,} entries              {fmt(cost)}"
                  f"  ({cost / entries * 1e6:5.1f} µs/entry)")

    print("\n  scaling with payload SIZE (1,000 entries, max_memory=64mb):")
    for payload in (1_024, 64 * 1_024, 1_024 * 1_024):
        with tempfile.TemporaryDirectory() as d:
            _seed(d, 1_000, payload)
            cost = measure(lambda: _decorate(d, "64mb"), repeats=5, warmup=1)
            print(f"    {payload // 1024:>5,} KB x 1,000 entries  {fmt(cost)}")

    print("\n  cost attributable to max_memory rehydration (1,000 entries):")
    with tempfile.TemporaryDirectory() as d:
        _seed(d, 1_000, 1_024)
        without = measure(lambda: _decorate(d, None), repeats=5, warmup=1)
        with_budget = measure(lambda: _decorate(d, "64mb"), repeats=5, warmup=1)
        print(f"    persist only                {fmt(without)}")
        print(f"    persist + max_memory        {fmt(with_budget)}"
              f"   (+{(with_budget - without) * 1e3:.1f} ms)")

    print("\nreading: decoration must scale with the NUMBER of cached entries,")
    print("never with their size — headers are read, bodies are not.")


if __name__ == "__main__":
    main()
