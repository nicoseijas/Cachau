# Cachau — Vision

> **Delightful, observable, bounded, and persistent function caching for Python data workloads.**

## Why Cachau exists

Cachau is not "yet another cache." It is a function cache designed around the real problems of data science work:

- **Large arguments** — DataFrames and arrays that are expensive to hash naively.
- **Expensive computations** — feature engineering, simulations, model evaluation, ETL.
- **Notebooks** — cells re-run constantly; processes restart; useful results should survive.
- **Voluminous results** — memory and disk must be explicitly bounded.
- **Invalidation** — results must expire when the code or the underlying data changes.

The target experience is as simple as:

```python
from cachau import cache

@cache(ttl="1h", persist=True, max_memory="2GB")
def expensive(df):
    ...
```

## Core principles

1. **Decorator-first.** The common case is one decorator with zero configuration.
2. **Safe by default.** A false HIT is worse than a MISS. When in doubt, recompute.
3. **Predictable behavior.** No surprising eviction, no hidden global state.
4. **Observable and explainable.** Every hit, miss, eviction, and skip has an inspectable reason.
5. **Data-science aware.** First-class hashing and serialization for NumPy, pandas, Polars, and Numba workloads.
6. **No hidden magic.** Automatic detection is conservative; explicitness beats unreliable cleverness.
7. **Scales down and up.** From a notebook cell to a production pipeline, same API.

## Positioning

Many libraries offer TTL, persistence, or LRU. Cachau's differentiation is the *combination*:

> Pleasant API **+** correct cache identity **+** native hashing for data objects **+** invalidation on code changes **+** bounded resources **+** transparent persistence **+** observability **+** hit/miss explanation **+** analysis of whether caching is even worth it.

The conceptual promise:

> **"Cachau — delightful function caching for Python data workloads."**
>
> Or, more technically: *"Observable, bounded, persistent function caching for Python."*

(Brand identity may subtly play with *"say ciao to recomputation"* — without turning the product into a joke.)

## Target use cases

Deterministic (or mostly deterministic) computations in:

- pandas / Polars transformations
- NumPy numeric work
- Feature engineering and preprocessing
- Simulations and scientific computing
- Model evaluation
- ETL steps
- Notebook workflows
- Numba-accelerated kernels and simulations

```python
@cache
def load_dataset(path):
    return pd.read_parquet(path)

@cache(ttl="1h")
def build_features(df, config):
    ...

@cache(persist=True)
def train_embedding(dataset_hash, params):
    ...

@cache(max_memory="2GB")
def expensive_simulation(seed, params):
    ...
```

## The differentiating features

### `cache.explain()` — transparency on demand

The system should be invisible when it works and fully transparent when you need to understand it:

```
MISS
Reason:      dependency_changed
Dependency:  data/train.parquet
Previous:    mtime=...
Current:     mtime=...
```

### `cache.profile()` — is caching even worth it?

Cachau doesn't just cache — it tells you when caching is a *bad* decision:

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

The fundamental economics rule: `T_key + T_lookup + T_deserialize < T_recompute`. Never assume an "expensive-looking" numeric function benefits from caching — measure it.

## Anti-goals

Cachau deliberately is **not**:

- Redis, Memcached, or a distributed cache
- A workflow engine or DAG scheduler
- An artifact registry or database
- An experiment tracker
- A replacement for joblib or Dask
- Automatic memoization of all of Python

The scope stays narrow:

> **"A pleasant, robust function cache for expensive Python data workloads."**

## Guiding maxims

- **Correctness before hit rate.** A false HIT is worse than a MISS.
- **Observable before clever.** Before adding sophisticated policies, the system must be able to explain what happened.
- **Data is often more important than arguments.** Files, datasets, SQL, environment variables, and package versions can be part of a computation's real identity.
- **Measure, do not assume.** Especially with Numba, a cache hit can be slower than recomputing.

The ideal experience:

```python
@cache(ttl="1h", max_memory="2GB", persist=True)
def expensive_analysis(df, config):
    ...
```

And whenever in doubt:

```python
expensive_analysis.cache.explain(df, config)
expensive_analysis.cache.profile(df, config)
```
