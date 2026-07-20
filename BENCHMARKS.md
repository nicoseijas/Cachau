# Benchmarks

Reproduce with the scripts in [`benchmarks/`](benchmarks/):

```
python benchmarks/bench_keying.py
python benchmarks/bench_hit_vs_recompute.py
python benchmarks/bench_numba.py
```

## Methodology (non-negotiable)

- Warm up first; report the **median** of repeated runs (robust to scheduler noise).
- One-time costs — JIT compilation, first-touch faults — are measured **separately and labeled**, never averaged into steady-state numbers (GUIDELINES.md §14).
- Each script prints its environment header. Numbers below are from one machine and one run; treat them as orders of magnitude, not contracts.

Reference environment: Python 3.13.3, Windows 11, NumPy 2.3.1, pandas 2.3.0, Numba 0.66.0, cachau 0.2.0.

## Keying cost by argument type

`T_key` is the first term of the fundamental rule `T_key + T_lookup + T_deserialize < T_recompute`. It scales with argument size:

| Argument | Keying cost (median) |
|---|---:|
| `int` | 8.5 µs |
| `str` (10 chars) | 7.2 µs |
| `dict` (10 items) | 30.8 µs |
| ndarray 1 KB (128 f64) | 11.1 µs |
| ndarray 800 KB (100k f64) | 338.7 µs |
| ndarray 8 MB (1M f64) | 4.1 ms |
| ndarray 80 MB (10M f64) | 40.6 ms |
| DataFrame 3 cols × 100k rows | 4.1 ms |

**Reading:** if your function is faster than its row, caching loses. Escape hatch: `key=lambda ...: <cheap stable identity>`.

## Full HIT path vs. recompute

| Scenario | HIT cost | Speedup |
|---|---:|---:|
| 50 ms function, `int` arg, memory backend | 7.7 µs | ~6,500× |
| 50 ms function, 8 MB array arg, memory backend | 4.0 ms | ~12× |
| 50 ms function, 8 MB result, disk backend (read + unpickle) | 3.4 ms | ~15× |

### The honest loss case

A ~200 ns function taking an 80 MB array:

| | |
|---|---:|
| recompute | 200 ns |
| cache HIT (dominated by keying) | 40.7 ms |

**Caching is a ~200,000× loss here.** This is why cachau's docs insist on measuring: an "expensive-looking" numeric function may be cheaper than hashing its input. `profile()` (V1.1) will diagnose this automatically.

## Numba boundary: cold JIT vs. warm vs. HIT

Pairwise-energy kernel (1,500 × 3 points), methodology *compile → warm up → benchmark*:

| | |
|---|---:|
| cold (JIT compile + execute, **one-time**) | 333.1 ms |
| warm compute (median, fresh args) | 1.6 ms |
| cachau HIT (keying + lookup) | 26.1 µs (~62× faster than warm) |

`stats()` reports the same story: `cold_compute_seconds=0.333` is excluded from savings estimates — compile cost is never counted as normal execution cost.
