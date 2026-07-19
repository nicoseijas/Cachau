# Cachau

**Delightful, observable, bounded, and persistent function caching for Python data workloads.**

Cachau is a function cache designed around the real problems of data science: large arguments, expensive computations, notebooks that restart, voluminous results, invalidation when code or data changes, and explicit memory and disk limits.

> Say ciao to recomputation.

```python
from cachau import cache

@cache(ttl="1h", persist=True, max_memory="2GB")
def expensive_analysis(df, config):
    ...
```

> **Status: v0.1.0 — the core engine is implemented and tested** (193 tests): normalized keys with type-tagged hashing, `key=`/`ignore=` escape hatches, code-change invalidation, TTL, LRU memory bounds, atomic corruption-safe persistence, same-key single-flight, `stats()` with miss reasons, and `explain()`. Pre-1.0, so the API may still evolve. Next up: validated first-class Numba support, `depends_on=` dependency invalidation, and `profile()` (see [ROADMAP](ROADMAP.md)).

## Installation

Not on PyPI yet — install from source:

```
pip install git+https://github.com/nicoseijas/Cachau
```

## Why not just `functools.lru_cache` / joblib / diskcache?

Plenty of libraries offer TTL, persistence, or LRU. None of them combine what data workloads actually need:

| Problem | Cachau's answer |
|---|---|
| Hashing a 2 GB DataFrame just to build a key | Native, optimized hashing for NumPy, pandas, and Polars — plus explicit `key=` / `ignore=` escape hatches |
| Stale results after you edit the function | Code-fingerprint invalidation by default: change `x * 2` to `x * 3`, the old result dies |
| Results that outlive the data they came from | `depends_on=["data.csv"]` — invalidate on file, env var, or package changes |
| Caches that eat all your RAM or disk | First-class `max_memory` bounds with predictable LRU eviction |
| Notebook restarts throwing work away | `persist=True` — atomic, versioned on-disk format that survives restarts |
| "Why was that a miss?!" | `func.cache.explain(...)` tells you exactly what happened and why |
| Caching that silently makes things *slower* | `func.cache.profile(...)` measures whether caching is even worth it |
| Numba treated as an afterthought | First-class support at the dispatcher boundary — `fastmath`/`parallel`-aware identity, honest cold/warm JIT metrics |

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

Coming in V1.1: `@cache(depends_on=["data/train.parquet"])` — invalidation when files, environment variables, or package versions change.

Every cached function carries its own control surface:

```python
build_features.cache.stats()        # hits, misses, hit rate, bytes, time saved...
build_features.cache.clear()
build_features.cache.invalidate(df, config)
build_features.cache.explain(df, config)
build_features.cache.profile(df, config)
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

(V1.1 adds dependency answers: *which file changed, previous vs. current fingerprint*.)

### `profile()` — is caching even worth it? *(planned — V1.1)*

```
Cache suitability

Computation (warm Numba):       182 ms
Key generation:                 347 ms
Cache lookup:                     2 ms
Deserialization:                 41 ms
--------------------------------------
Cache hit total:                390 ms

Caching is slower than recomputation by ~2.1x.
Primary cause:   hashing ndarray[float64, 480 MB]
Recommendation:  provide an explicit stable key or dataset version.
```

Cachau doesn't just cache — it tells you when caching is a bad decision.

## First-class Numba support *(in progress)*

```python
from numba import njit
from cachau import cache

@cache(ttl="1h", max_memory="4GB", persist=True)
@njit
def simulate(values, iterations):
    ...
```

Cachau caches **results** at the Python → dispatcher boundary; Numba's `cache=True` caches **machine code**. They compose: a Cachau HIT skips execution entirely, and a MISS still benefits from Numba's compilation cache. The design (dispatcher identity from semantically relevant compile options like `fastmath`/`parallel`, honest cold-JIT vs. warm-JIT metrics) is fully specified in [GUIDELINES.md](GUIDELINES.md); the dedicated test matrix that validates it is the next roadmap milestone — until then, treat Numba support as unverified.

## Design principles

1. **Correctness before hit rate.** A false HIT is worse than a MISS. When in doubt, recompute.
2. **Safe by default.** Exceptions aren't cached; serialization failure never loses your result; corruption degrades to a miss, never a mysterious error.
3. **Observable before clever.** Every hit, miss, eviction, and skip has an inspectable reason code.
4. **No hidden magic.** Automatic detection is conservative; explicitness beats unreliable cleverness.
5. **Bounded by design.** Memory and disk limits are core features, not afterthoughts.

## What Cachau is *not*

Not Redis, not a distributed cache, not a workflow engine, not an artifact registry, not an experiment tracker, not a joblib/Dask replacement. The scope stays narrow on purpose:

> *A pleasant, robust function cache for expensive Python data workloads.*

## Documentation

- **[VISION.md](VISION.md)** — why Cachau exists, positioning, and guiding maxims
- **[ROADMAP.md](ROADMAP.md)** — phased plan from foundations to Numba Level B
- **[GUIDELINES.md](GUIDELINES.md)** — the full design & engineering spec (API, cache identity, TTL, eviction, persistence, invalidation, observability, concurrency, Numba, testing)

## Contributing

The core engine is young and feedback is the most valuable contribution: try it on a real workload and open an issue with what surprised you. Bug reports with a failing test are gold. Before proposing features, read [GUIDELINES.md](GUIDELINES.md), especially the feature acceptance bar: every addition must preserve correctness, explainability, and the narrow mission.

## License

MIT
