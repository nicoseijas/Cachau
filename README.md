# Cachau

[![PyPI](https://img.shields.io/pypi/v/cachau)](https://pypi.org/project/cachau/) [![Python](https://img.shields.io/pypi/pyversions/cachau)](https://pypi.org/project/cachau/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Delightful, observable, bounded, and persistent function caching for Python data workloads.**

Cachau is a function cache designed around the real problems of data science: large arguments, expensive computations, notebooks that restart, voluminous results, invalidation when code or data changes, and explicit memory and disk limits.

> Say ciao to recomputation.

```python
from cachau import cache

@cache(ttl="1h", persist=True, max_memory="2GB")
def expensive_analysis(df, config):
    ...
```

> **Status: v0.4.0 — the core engine plus validated Numba Level A support** (339 tests, CI on 3.10-3.13 plus free-threaded 3.13t/3.14t): normalized keys with type-tagged hashing (incl. closure captures), native NumPy/pandas identity, `key=`/`ignore=` escape hatches, code-change invalidation, TTL, LRU memory bounds that survive restarts, atomic corruption-safe persistence, same-key single-flight, `stats()` with miss reasons and cold/warm JIT accounting, `explain()` (with eviction and dependency-diff detail), `inspect()`, `depends_on=` external-dependency invalidation (files, env vars, package versions, custom tokens), and `profile()` (measured cache economics). Pre-1.0, so the API may still evolve. Next up: Polars hashing (see [ROADMAP](ROADMAP.md)).
>
> **Upgrading from 0.2.x:** v0.3.x fixes a fingerprint collision that could serve one function's result for another (a false HIT). Closing it changes how every function's identity is computed, so **existing persisted caches are invalidated once** — the first run after upgrading recomputes and reclaims the old files automatically. No action needed.

## Installation

```
pip install cachau
```

Python 3.10+. **Zero dependencies** — NumPy, pandas, and Numba integrations activate automatically when those libraries are present, without ever importing them.

## Quick start

```python
from cachau import cache

@cache(persist=True, max_memory="500MB")
def slow_square(n):
    print("computing...")
    return n * n

slow_square(12)              # computing...  → 144
slow_square(12)              # → 144 (HIT — and it survives a restart)

slow_square.cache.stats().hit_rate   # 0.5
print(slow_square.cache.explain(12))
# HIT
# Reason:      found
# Namespace:   __main__.slow_square
# Created:     2026-07-19 18:02:33 UTC
# Age:         0s
# Size:        28 B
```

## Why not just `functools.lru_cache` / joblib / diskcache?

Plenty of libraries offer TTL, persistence, or LRU. None of them combine what data workloads actually need:

| Problem | Cachau's answer |
|---|---|
| Hashing a 2 GB DataFrame just to build a key | Native hashing for NumPy and pandas (dtype + shape + content; layout-canonicalized) — plus explicit `key=` / `ignore=` escape hatches |
| Stale results after you edit the function | Code-fingerprint invalidation by default: change `x * 2` to `x * 3` — or a closure capture, or a Numba compile flag — and the old result dies |
| N threads recomputing the same missing key | Same-key single-flight: one computation, everyone else reuses it; independent keys never serialize |
| Caches that eat all your RAM or disk | First-class `max_memory` bounds with predictable LRU eviction; oversized results are returned but never cached |
| Notebook restarts throwing work away | `persist=True` — atomic, versioned, corruption-safe on-disk format that survives restarts |
| "Why was that a miss?!" | `func.cache.explain(...)` tells you exactly what happened and why — as pure observation |
| Numba treated as an afterthought | First-class support at the dispatcher boundary — `fastmath`/`parallel`/`locals=`-aware identity, honest per-specialization cold/warm JIT metrics |
| Results that outlive the data they came from | `depends_on=["data.csv", cachau.env("MODE"), cachau.package("numpy")]` — a result dies when its file, env var, package version, or custom token changes |

## A taste of the API

The common case is one decorator, zero configuration:

```python
@cache
def load_dataset(path):
    return pd.read_parquet(path)
```

Configuration is declarative and progressive — no backend objects, no config files:

```python
@cache(ttl="1h")
def build_features(df, config):
    ...

@cache(persist=True)
def train_embedding(dataset_hash, params):
    ...

@cache(max_memory="2GB")
def expensive_simulation(seed, params):
    ...

@cache(ignore=["logger", "progress_callback"])
def run(data, logger=None, progress_callback=None):
    ...

@cache(key=lambda dataset, version: version)
def process(dataset, version):
    ...
```

Declare external inputs a result depends on, and Cachau invalidates when they change:

```python
@cache(depends_on=[
    "data/train.parquet",                     # a file — content hash by default
    cachau.file("big.bin", on="mtime"),       # or cheap mtime+size, opt-in
    cachau.env("PIPELINE_MODE"),              # an environment variable
    cachau.package("scikit-learn"),           # an installed package version
    cachau.token(lambda: db.schema_version()),  # any custom token
])
def build_features(...):
    ...
```

A changed dependency is a `dependency_changed` miss: the stale entry is dropped and the function recomputes. The fingerprints ride along as small metadata in each stored entry, not in the key — so a changed dependency overwrites the same entry, and the miss is attributed to the dependency instead of vanishing as a key-not-found. `explain()` names exactly which one changed.

Files default to a **content hash** (correctness first — a same-size replacement that preserves the modification time still invalidates); `on="mtime"` opts into a cheaper mtime+size check for large files where hashing every lookup is the bottleneck. A declared dependency is assumed stable for the duration of one call — if it can change while the function runs, pass it as an argument so it enters the key.

Every cached function carries its own control surface:

```python
build_features.cache.stats()        # hits, misses, hit rate, miss reasons, bytes,
                                    # evictions, compute time, estimated time saved,
                                    # cold-JIT time — as an immutable snapshot
build_features.cache.clear()
build_features.cache.invalidate(df, config)
build_features.cache.inspect()               # browse the cached entries
build_features.cache.explain(df, config)     # pure observation, never recomputes
build_features.cache.profile(df, config)     # measures: is caching worth it?
```

### `explain()` — transparency on demand

```
MISS
Reason:      expired
Namespace:   features.build_features
Created:     2026-07-19 14:03:11 UTC
Expired:     3m 2s ago (at 2026-07-19 15:03:11 UTC)
Size:        1.2 MB
```

With `depends_on=`, a changed dependency reports as its own reason and shows exactly which one changed, before and after:

```
MISS
Reason:      dependency_changed
Changed dep: env:PIPELINE_MODE (v:fast -> v:thorough)
Namespace:   features.build_features
Created:     2026-07-19 14:03:11 UTC
Size:        1.2 MB
```

An entry the LRU budget dropped reports `evicted` rather than `not_found`, so you can tell "never cached" from "cached, then pushed out".

### `inspect()` — browse what's cached

`inspect()` lists the entries a function currently holds — newest first, read from entry headers without deserializing any values, so it stays cheap over a large persistent cache:

```
3 cached entries for features.build_features (4.6 MB)

  a1b2c3d4e5f60718  1.2 MB  age 3m   ttl 57m       deps env:PIPELINE_MODE
  9f8e7d6c5b4a3021  2.1 MB  age 11m  ttl 49m       deps env:PIPELINE_MODE
  0011223344556677  1.3 MB  age 2h   EXPIRED       deps env:PIPELINE_MODE
```

The result is a plain read-only sequence of `CacheEntryView` (indexable, iterable), each with `age_seconds`, `ttl_remaining_seconds`, `is_expired`, `size_bytes`, and `dependency_fingerprints` — natural to poke at in a notebook cell.

### `profile()` — is caching even worth it?

`profile()` measures both sides of the cache-economics inequality (`T_key + T_lookup + T_deserialize < T_recompute`) for one concrete call — running the function, warmed up, so JIT compile time is never counted — and tells you which side wins and why:

```
Cache economics: features.aggregate

Warm recompute:         3.9 ms
Key generation:        16.0 ms
Cache read:             0 us
------------------------------
Cache hit total:       16.0 ms

Caching is slower than recompute by 4.1x.
Primary hit cost:     hashing ndarray[float64, 30.5 MB]
Recommendation:       Provide an explicit stable key= (e.g. a dataset version)
                      so the payload isn't hashed on every lookup - that is the
                      whole cost here.
```

Here hashing a 30-MB array to build the key costs more than just recomputing the result, so the cache makes things *worse* — and `profile()` says so, names the culprit, and points at the fix. Unlike `explain()`, it runs the function (it must, to measure recompute cost); it doesn't touch `stats()` and restores cache state afterward. Cachau doesn't just cache — it tells you when caching is a bad decision.

## The persistent cache directory is a trust boundary

Persisted values are serialized with `pickle`, so **reading** an entry deserializes whatever is on disk. Treat the cache directory the way you treat an importable Python file:

- Keep it private to the user or service running the cache — the default `.cachau/` under your project is fine; `/tmp`, a world-writable share, or a volume mounted into a less-trusted container is not.
- Never point `persist=` at a directory another user or process can write to. Writing there is equivalent to executing code inside your process on the next read.
- Never ship or download a prepopulated cache directory as if it were data.

Cachau treats damaged entries as a MISS (bad version, corrupt metadata, undecodable payload — the file is dropped and the value recomputed), but that is corruption handling, not a defense against a hostile writer.

## First-class Numba support

```python
from numba import njit
from cachau import cache

@cache(ttl="1h", max_memory="4GB", persist=True)
@njit
def simulate(values, iterations):
    ...
```

Cachau caches **results** at the Python → dispatcher boundary (`@cache` goes below `@njit`); Numba's `cache=True` caches **machine code**. They compose: a Cachau HIT skips execution entirely, and a MISS still benefits from Numba's compilation cache. Dispatcher identity covers the Python function, closure captures, and semantically relevant compile options (`fastmath`, `parallel`, `boundscheck`, `error_model`, `locals=` type forcing) — changing any of them invalidates stale results. Metrics are honest about JIT: each specialization's first compile is reported as `cold_compute_seconds` and never counted as normal execution cost. Validated by a 26-test matrix.

### Works with [numba-utils](https://github.com/nicoseijas/numba-utils)

numba-utils' decorator aliases (`njit_fast`, `njit_parallel`, `cached_njit`, `boundscheck`) return real Numba dispatchers, so cachau composes with them out of the box — verified by an [integration suite](tests/test_numba_utils_compat.py):

```python
from numba_utils.decorators import njit_fast
from cachau import cache

@cache(persist=True)
@njit_fast          # fastmath=True lands in the cache identity automatically
def kernel(values):
    return values * 2.0
```

The options the aliases inject (`fastmath`, `parallel`) — and numba-utils' global `configure()` / `NUMBA_UTILS_*` overrides — all land in the dispatcher's compile options, so cachau fingerprints them: `njit_fast` and `cached_njit` with the same body never share an entry, and flipping a global override invalidates correctly. Its typed containers are Level B: as arguments they fail loudly (use `key=` / `ignore=`).

## Design principles

1. **Correctness before hit rate.** A false HIT is worse than a MISS. When in doubt, recompute.
2. **Safe by default.** Exceptions aren't cached; serialization failure never loses your result; corruption degrades to a miss, never a mysterious error.
3. **Observable before clever.** Every hit, miss, eviction, and skip has an inspectable reason code.
4. **No hidden magic.** Automatic detection is conservative; explicitness beats unreliable cleverness.
5. **Bounded by design.** Memory and disk limits are core features, not afterthoughts.

## What Cachau is *not*

Not Redis, not a distributed cache, not a workflow engine, not an artifact registry, not an experiment tracker, not a joblib/Dask replacement. The scope stays narrow on purpose:

> *A pleasant, robust function cache for expensive Python data workloads.*

## Cache economics, measured

Caching has a cost — keying, lookup, deserialization — and cachau refuses to pretend otherwise. [BENCHMARKS.md](BENCHMARKS.md) has the numbers (reproducible via [`benchmarks/`](benchmarks/)): a memory HIT on a 50 ms function is a ~6,500× win with scalar args and ~12× with an 8 MB array arg — while caching a 200 ns function with an 80 MB argument is a ~200,000× **loss**. Measure, don't assume.

## Documentation

- **[examples/](examples/)** — four runnable scripts: quickstart with persistence, pandas workflows (`ignore=`/`key=`), observability (miss reasons, `explain()`), and Numba workloads with honest JIT metrics
- **[BENCHMARKS.md](BENCHMARKS.md)** — measured keying costs, hit-vs-recompute economics, cold/warm JIT — with methodology
- **[VISION.md](VISION.md)** — why Cachau exists, positioning, and guiding maxims
- **[ROADMAP.md](ROADMAP.md)** — phased plan from foundations to Numba Level B
- **[GUIDELINES.md](GUIDELINES.md)** — the full design & engineering spec (API, cache identity, TTL, eviction, persistence, invalidation, observability, concurrency, Numba, testing)

## Contributing

The core engine is young and feedback is the most valuable contribution: try it on a real workload and open an issue with what surprised you. Bug reports with a failing test are gold. Before proposing features, read [GUIDELINES.md](GUIDELINES.md), especially the feature acceptance bar: every addition must preserve correctness, explainability, and the narrow mission.

## License

MIT
