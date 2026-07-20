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


def measure(
    fn: Callable[[], Any], *, repeats: int = 9, warmup: int = 2
) -> float:
    """Median wall-clock seconds of ``fn()`` after ``warmup`` discarded runs."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
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
