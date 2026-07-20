"""Shared measurement helpers: honest methodology only.

Warm up first, then take the MEDIAN of repeated runs (robust to scheduler
noise). One-time costs (JIT compilation, first-touch page faults) are
measured separately and labeled — never averaged into steady-state numbers.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from typing import Any, Callable


# A sample must last long enough that timer granularity and scheduler noise
# are a rounding error rather than the measurement. At ~7 µs per call, timing
# one call at a time measures the OS; batching until the sample clears this
# floor measures the code.
_MIN_SAMPLE_SECONDS = 0.005


def _calibrate(fn: Callable[[], Any]) -> int:
    """Smallest power-of-two batch whose runtime clears the noise floor."""
    batch = 1
    while batch < 1 << 20:
        start = time.perf_counter()
        for _ in range(batch):
            fn()
        if time.perf_counter() - start >= _MIN_SAMPLE_SECONDS:
            return batch
        batch *= 2
    return batch


def measure(
    fn: Callable[[], Any], *, repeats: int = 9, warmup: int = 2, batch: int | None = None
) -> float:
    """Median wall-clock seconds of ONE ``fn()`` after ``warmup`` discarded runs.

    Each sample runs ``fn`` ``batch`` times and divides, because a single
    microsecond-scale call is below the noise floor of the wall clock: timing
    one 7 µs cache hit at a time reports whatever the scheduler did, and moves
    by 2x between runs. ``batch`` is calibrated automatically unless pinned.
    """
    for _ in range(warmup):
        fn()
    if batch is None:
        batch = _calibrate(fn)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(batch):
            fn()
        samples.append((time.perf_counter() - start) / batch)
    return statistics.median(samples)


def measure_once(fn: Callable[[], Any]) -> float:
    """Wall-clock seconds of a single run — for one-time costs (cold JIT)."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def fmt(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:8.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.1f} ms"
    return f"{seconds:8.2f} s "


def print_environment() -> None:
    print(f"python {sys.version.split()[0]} | {platform.platform()}")
    try:
        import numpy

        print(f"numpy {numpy.__version__}", end="")
    except ImportError:
        print("numpy not installed", end="")
    try:
        import pandas

        print(f" | pandas {pandas.__version__}", end="")
    except ImportError:
        pass
    try:
        import numba

        print(f" | numba {numba.__version__}", end="")
    except ImportError:
        pass
    import cachau

    print(f" | cachau {cachau.__version__}")
    print("-" * 72)
