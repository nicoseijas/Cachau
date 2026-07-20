"""Decoration-time cost of a persistent cache.

Import time is not free for ``persist=``: decorating scans the cache
directory once, to purge entries left by superseded versions of the function
and — since v0.3.0 — to rebuild the LRU budget so ``max_memory`` still holds
after a restart.

Decoration happens ONCE per process, on a directory the OS has not touched
yet, so the honest number is the **cold** one. Measuring it the usual way
(warm up, then take the median) reports the wrong thing by design: the warmup
run is the only one that resembles a real import, and every later run reads a
directory the OS now has cached. On this machine that is a 57x difference.
Both are reported below, labeled, because they answer different questions:
cold is what a user's startup pays, warm is what the scan costs once I/O is
free.
"""

import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _timing import fmt, measure, measure_once, print_environment

from cachau import cache


def _seed(directory: str, entries: int, payload_bytes: int) -> None:
    blob = b"x" * payload_bytes

    @cache(persist=directory, namespace="bench.seeded")
    def seeded(n: int) -> bytes:
        return blob

    for n in range(entries):
        seeded(n)


def _decorate(directory: str, max_memory: str | None) -> None:
    # A namespace of its own: sharing the seed's namespace would make the scan
    # also PURGE every entry, so the run would measure deletion, not scanning,
    # and the next repeat would find an empty directory.
    @cache(persist=directory, max_memory=max_memory, namespace="bench.target")
    def target(n: int) -> bytes:
        return b""


def cold(entries: int, payload_bytes: int, max_memory: str | None = "64mb",
         repeats: int = 3) -> float:
    """Median first-ever scan, each repeat against a freshly seeded directory."""
    samples = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as d:
            _seed(d, entries, payload_bytes)
            samples.append(measure_once(lambda: _decorate(d, max_memory)))
    return statistics.median(samples)


def warm(entries: int, payload_bytes: int, max_memory: str | None = "64mb") -> float:
    """Median re-scan of a directory the OS has already cached."""
    with tempfile.TemporaryDirectory() as d:
        _seed(d, entries, payload_bytes)
        return measure(lambda: _decorate(d, max_memory), repeats=5, warmup=1)


def main() -> None:
    print("Decoration cost of persist= (paid once per import)\n")
    print_environment()

    print("  scaling with entry COUNT (1 KB payloads, max_memory=64mb):")
    print(f"    {'entries':>8}  {'cold (real import)':>20}  {'warm re-scan':>14}")
    for entries in (100, 500, 2_000):
        c, w = cold(entries, 1_024), warm(entries, 1_024)
        print(f"    {entries:>8,}  {fmt(c):>20}  {fmt(w):>14}"
              f"   ({c / entries * 1e3:.2f} ms/entry cold)")

    print("\n  scaling with payload SIZE (300 entries) — must stay flat:")
    for payload in (1_024, 64 * 1_024, 1_024 * 1_024):
        c, w = cold(300, payload), warm(300, payload)
        print(f"    {payload // 1024:>5,} KB x 300      cold {fmt(c)}   warm {fmt(w)}")

    print("\n  cost attributable to max_memory rehydration (500 entries, cold):")
    without = cold(500, 1_024, max_memory=None)
    with_budget = cold(500, 1_024, max_memory="64mb")
    print(f"    persist only                {fmt(without)}")
    print(f"    persist + max_memory        {fmt(with_budget)}"
          f"   ({(with_budget - without) * 1e3:+.0f} ms)")

    print("\nreading: budget for the COLD column — decoration happens once, at")
    print("import, against a directory the OS has not read yet. It scales with")
    print("the NUMBER of entries, never with their size.")


if __name__ == "__main__":
    main()
