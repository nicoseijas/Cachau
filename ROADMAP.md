# Cachau — Roadmap

> Scope discipline is a feature. Everything beyond V1 must justify its complexity through real use cases.

## Phase 0 — Foundations ✅ (shipped in v0.1.0)

Goal: the architectural skeleton, correct before fast.

- [x] Layered architecture in place: Decorator / Public API → Invocation Normalization → Key Builder → Dependency Fingerprinting → Cache Policy → Storage Backend → Serializer → Metrics/Events
- [x] Minimal `CacheBackend` interface (`get`, `set`, `delete`, `clear`, `iter_entries`, plus `peek`/`iter_metadata` for pure observation)
- [x] `MemoryBackend`
- [x] Key building: signature binding / argument normalization (kwargs vs. positional share a key)
- [x] Function identity + code fingerprint (implementation change ⇒ invalidation, including on-disk reclamation)
- [x] Namespace isolation (`package.module.function` + fingerprint; explicit `namespace=` override)
- [x] Native hashing for primitives, `dict`/`list`/`tuple`/`set`, dataclasses, enums, `pathlib.Path` — collision-safe (length-prefixed, type-tagged encoding)
- [x] Loud failure on unhashable arguments (never silently ignored)
- [x] Test harness covering the mandatory semantic tests

## Phase 1 — V1: the core promise ✅ core shipped in v0.1.0 (Numba validation pending)

Goal: the four-keyword experience, correct and observable.

**API surface**

- [x] `@cache` (zero-config)
- [x] `@cache(ttl=...)` — int seconds and `"30s"/"10m"/"2h"/"7d"` strings; lazy expiration; TTL starts at commit; backward-clock-step clamp
- [x] `@cache(max_memory=...)` — LRU eviction; oversized entries computed, returned, not cached (`skipped_oversized`); injectable `size_of=`
- [x] `@cache(persist=...)` — `DiskBackend` with atomic writes (temp → sync → rename), versioned format, per-entry metadata, corruption-safe reads
- [x] `@cache(key=...)` and `@cache(ignore=[...])`
- [x] `func.cache.clear()`, `func.cache.invalidate(...)`, `func.cache.stats()`
- [ ] Global `cache.clear()`, `cache.stats()`, `cache.configure(...)` → moved to V1.1 (needs a control registry)

**Data-science identity**

- [x] Optimized hashing for `numpy.ndarray` (dtype + shape + content; layout-canonicalized)
- [x] Optimized hashing for `pandas.DataFrame` / `Series` / `Index`

**Correctness & robustness**

- [x] Invalidation on code change (function fingerprint, including on-disk reclamation)
- [x] Miss reasons: `miss_not_found`, `miss_expired`, `miss_invalidated` per call; code-change invalidations reported at decoration (`miss_code_changed`/`miss_dependency_changed`/`miss_corrupt` as distinct per-call reasons arrive with `depends_on` in V1.1)
- [x] Safe fallback on corruption (degrade to miss, never mysterious errors)
- [x] Serialization failure ⇒ return result, record `write_errors` (delete failures tracked separately)
- [x] Exceptions not cached
- [x] Same-key single-flight (refcounted per-key locking, no global lock, thread-safe budget)

**Observability**

- [x] Full `stats()`: hits (incl. coalesced), misses, hit rate, writes, skipped writes, expirations, invalidations, evictions, serde/delete errors, entry count, bytes, compute time, estimated time saved
- [x] `func.cache.explain(...)` — HIT/MISS with reason, size, remaining TTL; strictly pure observation (changed-dependency answers arrive with `depends_on`)

**Numba Level A (first-class)** → next milestone: the design is specified, the validating test matrix is not written yet

- [ ] `@njit` functions called from Python (cache at the dispatcher boundary)
- [ ] ndarray inputs/results, scalars
- [ ] Dispatcher identity from Python function + semantically relevant compile options (`fastmath`, `parallel`, `boundscheck`, `error_model`)
- [ ] Coexistence with Numba `cache=True` (compilation cache ⊥ result cache)
- [ ] `parallel=True` and `fastmath` supported; `fastmath` change invalidates when relevant
- [ ] Persistence, TTL, memory limits, metrics, invalidation all working for Numba workloads
- [ ] Cold/warm JIT distinction in metrics and benchmarks

**Stretch goal**

- [ ] `func.cache.profile(...)` — moved to V1.1 (Phase 2)

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
