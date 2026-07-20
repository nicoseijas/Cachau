# Benchmarks

Reproduce with the scripts in [`benchmarks/`](benchmarks/):

```
python benchmarks/bench_keying.py
python benchmarks/bench_hit_vs_recompute.py
python benchmarks/bench_numba.py
python benchmarks/bench_decoration.py
```

## Methodology (non-negotiable)

- Warm up first; report the **median** of repeated runs (robust to scheduler noise).
- Microsecond-scale operations are **batched**: each sample runs the operation enough times to last at least 5 ms, then divides. Timing a single 7 µs cache hit measures the scheduler, not the code — it moved by 2× between runs before this was fixed.
- One-time costs — JIT compilation, first-touch faults — are measured **separately and labeled**, never averaged into steady-state numbers (GUIDELINES.md §14).
- Each script prints its environment header. Numbers below are from one machine and one run; treat them as orders of magnitude, not contracts.

Reference environment: Python 3.13.3, Windows 11, NumPy 2.3.1, pandas 2.3.0, Numba 0.66.0, cachau 0.3.2.

## Keying cost by argument type

`T_key` is the first term of the fundamental rule `T_key + T_lookup + T_deserialize < T_recompute`. It scales with argument size:

| Argument | Keying cost (median) |
|---|---:|
| `int` | 6.2 µs |
| `str` (10 chars) | 6.2 µs |
| `dict` (10 items) | 26.9 µs |
| ndarray 1 KB (128 f64) | 8.9 µs |
| ndarray 800 KB (100k f64) | 355.5 µs |
| ndarray 8 MB (1M f64) | 4.0 ms |
| ndarray 80 MB (10M f64) | 40.5 ms |
| DataFrame 3 cols × 100k rows | 4.3 ms |

**Reading:** if your function is faster than its row, caching loses. Escape hatch: `key=lambda ...: <cheap stable identity>`.

## Full HIT path vs. recompute

| Scenario | HIT cost | Speedup |
|---|---:|---:|
| 50 ms function, `int` arg, memory backend | 7.1 µs | ~7,000× |
| 50 ms function, 8 MB array arg, memory backend | 4.0 ms | ~12× |
| 50 ms function, 8 MB result, disk backend (read + unpickle) | 3.3 ms | ~15× |

The thread-safe stats counters added in 0.3.0 put a lock acquisition on the hit path. Measured against 0.2.0 on the same machine, over 2,000 hits per sample: **7.00 µs before, 7.02 µs after** — correct tallies cost nothing here.

### The honest loss case

A ~130 ns function taking an 80 MB array:

| | |
|---|---:|
| recompute | 129.9 ns |
| cache HIT (dominated by keying) | 44.6 ms |

**Caching is a ~340,000× loss here.** This is why cachau's docs insist on measuring: an "expensive-looking" numeric function may be cheaper than hashing its input. `profile()` (V1.1) will diagnose this automatically.

## Numba boundary: cold JIT vs. warm vs. HIT

Pairwise-energy kernel (1,500 × 3 points), methodology *compile → warm up → benchmark*:

| | |
|---|---:|
| cold (JIT compile + execute, **one-time**) | 339.7 ms |
| warm compute (median, fresh args) | 1.6 ms |
| cachau HIT (keying + lookup) | 24.0 µs (~66× faster than warm) |

`stats()` reports the same story: `cold_compute_seconds=0.340` is excluded from savings estimates — compile cost is never counted as normal execution cost.

## Decoration cost of `persist=`

Decorating a persistent cache scans the store once: it purges entries left by superseded versions of the function and — since 0.3.0 — rebuilds the LRU budget so `max_memory` still holds after a restart. This is paid at **import**, before the program does any work of its own, so it gets its own number.

| Store | Decoration cost |
|---|---:|
| 10 entries | 0.8 ms |
| 100 entries | 6.8 ms |
| 1,000 entries | 61.4 ms |
| 5,000 entries | 330.7 ms |

Roughly 65 µs per entry, linear in the **number** of entries. Rebuilding the LRU budget is not what costs: on 1,000 entries, `persist=` alone took 71.0 ms and `persist=` + `max_memory=` took 71.6 ms. The scan itself dominates, and purge and rehydration share a single pass over it.

Decoration must not scale with the **size** of what is cached, only with how many things are cached:

| 1,000 entries of | Decoration cost |
|---|---:|
| 1 KB | 65.0 ms |
| 64 KB | 58.8 ms |
| 1 MB | 60.4 ms |

That flatness is recent. Through 0.3.1 the metadata scan read each file in full and discarded the body, so the 1 MB row cost **519.8 ms** — a cache holding 1,000 × 1 MB results moved a gigabyte through memory at import. 0.3.2 reads only the two header lines (`test_decoration_does_not_read_payload_bytes` holds the line at 64 KB total for 5 MB of payloads).

**Reading:** with `persist=`, keep an eye on entry count, not cache size. A store with 100,000 entries adds several seconds to every import; that is what `max_memory=` and `ttl=` are for.
