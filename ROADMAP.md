# Cachau — Roadmap

> Scope discipline is a feature. Everything beyond V1 must justify its complexity through real use cases.

## Phase 0 — Foundations

Goal: the architectural skeleton, correct before fast.

- [ ] Layered architecture in place: Decorator / Public API → Invocation Normalization → Key Builder → Dependency Fingerprinting → Cache Policy → Storage Backend → Serializer → Metrics/Events
- [ ] Minimal `CacheBackend` interface (`get`, `set`, `delete`, `clear`, `iter_entries`)
- [ ] `MemoryBackend`
- [ ] Key building: signature binding / argument normalization (kwargs vs. positional share a key)
- [ ] Function identity + code fingerprint (implementation change ⇒ invalidation)
- [ ] Namespace isolation (`package.module.function` + fingerprint; explicit `namespace=` override)
- [ ] Native hashing for primitives, `dict`/`list`/`tuple`, dataclasses, `pathlib.Path`
- [ ] Loud failure on unhashable arguments (never silently ignored)
- [ ] Test harness covering the mandatory semantic tests

## Phase 1 — V1: the core promise

Goal: the four-keyword experience, correct and observable.

**API surface**

- [ ] `@cache` (zero-config)
- [ ] `@cache(ttl=...)` — int seconds and `"30s"/"10m"/"2h"/"7d"` strings; lazy expiration; TTL starts at commit
- [ ] `@cache(max_memory=...)` — LRU eviction; oversized entries computed, returned, not cached (`cache_skip_oversized`)
- [ ] `@cache(persist=...)` — `DiskBackend` with atomic writes (temp → sync → rename), versioned format, per-entry metadata
- [ ] `@cache(key=...)` and `@cache(ignore=[...])`
- [ ] `func.cache.clear()`, `func.cache.invalidate(...)`, `func.cache.stats()`
- [ ] Global `cache.clear()`, `cache.stats()`, `cache.configure(...)`

**Data-science identity**

- [ ] Optimized hashing for `numpy.ndarray` (dtype + shape + content)
- [ ] Optimized hashing for `pandas.DataFrame` / `Series`

**Correctness & robustness**

- [ ] Invalidation on code change (function fingerprint)
- [ ] Miss reasons: `miss_not_found`, `miss_expired`, `miss_invalidated`, `miss_code_changed`, `miss_dependency_changed`, `miss_corrupt`
- [ ] Safe fallback on corruption (degrade to miss, never mysterious errors)
- [ ] Serialization failure ⇒ return result, record `cache_write_error`
- [ ] Exceptions not cached
- [ ] Same-key single-flight (per-key locking, no global lock)

**Observability**

- [ ] Full `stats()`: hits, misses, hit rate, writes, skipped writes, expirations, invalidations, evictions, serde errors, entry count, bytes, compute time, estimated time saved
- [ ] `func.cache.explain(...)` — HIT/MISS with reason, changed dependency, size, remaining TTL

**Numba Level A (first-class)**

- [ ] `@njit` functions called from Python (cache at the dispatcher boundary)
- [ ] ndarray inputs/results, scalars
- [ ] Dispatcher identity from Python function + semantically relevant compile options (`fastmath`, `parallel`, `boundscheck`, `error_model`)
- [ ] Coexistence with Numba `cache=True` (compilation cache ⊥ result cache)
- [ ] `parallel=True` and `fastmath` supported; `fastmath` change invalidates when relevant
- [ ] Persistence, TTL, memory limits, metrics, invalidation all working for Numba workloads
- [ ] Cold/warm JIT distinction in metrics and benchmarks

**Stretch goal (V1 if it stays small, otherwise V1.1)**

- [ ] `func.cache.profile(...)` — cache-suitability analysis (keying vs. lookup vs. deserialize vs. warm recompute, with recommendation)

## Phase 2 — V1.1: explain more, depend on data

Goal: deepen the differentiators.

- [ ] `cache.profile()` complete (if it didn't ship in V1): timing breakdown, net savings, primary-cost diagnosis, actionable recommendation
- [ ] `func.cache.inspect()` — entry browsing
- [ ] `depends_on=[...]` external dependency invalidation: files (mtime / size / content hash), environment variables, package versions, user-defined tokens
- [ ] Polars hashing support
- [ ] Richer `explain()`: eviction history, dependency fingerprint diffs
- [ ] Notebook polish: cell re-runs never destroy useful persistent caches; code changes invalidate understandably
- [ ] Benchmark suite with honest methodology (compile → warm up → benchmark; cold JIT reported separately)

## Phase 3 — V2: Numba Level B and hardening

Goal: broaden supported types with the same correctness bar. Nothing is "supported" without deterministic hashing, serialization round-trip, stability tests, and benchmarks.

- [ ] `numba.typed.List` adapter
- [ ] `numba.typed.Dict` adapter
- [ ] jitclass via explicit keys / field-based adapters / custom serializers (never compiler internals)
- [ ] Extension-type adapter API
- [ ] Custom serializer registration (Arrow/Parquet, native NumPy formats)
- [ ] Mutation policy: detect-and-reject where possible, documented opt-in otherwise
- [ ] Cross-machine portability of persisted results (semantic identity, not compilation-artifact identity)

## Explicitly out of scope (until the local model is consolidated)

These are **anti-goals** for now — not "later," but "not until proven necessary by real cases":

- Calling `@cache` from nopython mode
- Distributed compilation or result caches; Redis / S3 / database backends
- CUDA / device memory
- Automatic replay of mutations
- Workflow engine, DAG scheduler, artifact registry, experiment tracker features
- Replacing joblib or Dask
- Automatic memoization of arbitrary Python

## Release criteria (every phase)

- Correctness before hit rate: no false HITs, ever. Under uncertainty, recompute.
- Mandatory semantic test suite green (see GUIDELINES.md §16), including the Numba matrix
- Every observable behavior has a reason code and shows up in `stats()` / `explain()`
- Docs updated: what "size" means, TTL semantics, persistence format version, invalidation triggers
