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
- One-time costs — JIT compilation, first-touch faults, the import-time cache scan — are measured **separately and labeled**, never averaged into steady-state numbers (GUIDELINES.md §14). Where an operation only ever happens once per process, warming up measures the wrong thing: see [decoration cost](#decoration-cost-of-persist), where the discarded warmup run was 33× the median that followed it.
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

Decoration happens **once per process**, against a directory the OS has not read yet, so the honest number is the *cold* one. Warming up and taking a median — the right method everywhere else on this page — reports the wrong thing here by construction: the discarded warmup run is the only one that resembles a real import, and every later run re-reads a directory the OS now has cached. The gap is 33×, so both columns are shown.

| Store (1 KB entries) | Cold — what an import pays | Warm re-scan |
|---|---:|---:|
| 100 entries | 414.5 ms | 6.7 ms |
| 500 entries | 2.12 s | 35.2 ms |
| 2,000 entries | 8.15 s | 123.8 ms |

**≈4.1 ms per entry, cold**, linear in the number of entries — and dominated by per-file open latency on this machine (Windows 11 with Defender active), not by bytes read. Expect a different constant on Linux or with the directory excluded from AV scanning; the linearity is what transfers.

Rebuilding the LRU budget is not what costs: over 500 entries, `persist=` alone took 2.03 s and `persist=` + `max_memory=` took 1.98 s (the difference is noise). Purge and rehydration share a single pass.

Decoration must scale with how many things are cached, never with their size:

| 300 entries of | Cold | Warm re-scan |
|---|---:|---:|
| 1 KB | 1.20 s | 18.2 ms |
| 64 KB | 1.90 s | 16.8 ms |
| 1 MB | 1.16 s | 18.7 ms |

Through 0.3.1 the scan read each file in full and discarded the body, so a cache holding 1,000 × 1 MB results moved a gigabyte through memory at import. 0.3.2 reads only the two header lines; `test_decoration_does_not_read_payload_bytes` holds the line at 64 KB read for 5 MB of payloads.

How much that fix buys depends on whether opening the file is cheaper than reading it. Warm, where opens are already cached, 300 × 1 MB went from 519.8 ms to 60.4 ms. Cold on this machine, where each open costs ~4 ms regardless, the same comparison is **1392 ms → 1181 ms, about 15%**. The earlier 8.6× figure was a warm-cache measurement and overstated it. What the fix guarantees unconditionally is that the scan's cost and memory use stop tracking payload size at all.

**Reading:** with `persist=`, watch entry count, not cache size. A store with 20,000 entries can add a minute to every import on Windows. That is what `max_memory=` and `ttl=` are for.
