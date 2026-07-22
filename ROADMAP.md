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
- [x] Miss reasons: `miss_not_found`, `miss_expired`, `miss_invalidated`, `miss_dependency_changed` per call; code-change invalidations reported at decoration (`miss_code_changed`/`miss_corrupt` as distinct per-call reasons still pending)
- [x] Safe fallback on corruption (degrade to miss, never mysterious errors)
- [x] Serialization failure ⇒ return result, record `write_errors` (delete failures tracked separately)
- [x] Exceptions not cached
- [x] Same-key single-flight (refcounted per-key locking, no global lock, thread-safe budget)

**Observability**

- [x] Full `stats()`: hits (incl. coalesced), misses, hit rate, writes, skipped writes, expirations, invalidations, evictions, serde/delete errors, entry count, bytes, compute time, estimated time saved
- [x] `func.cache.explain(...)` — HIT/MISS with reason, size, remaining TTL, and which declared dependency changed; strictly pure observation

**Numba Level A (first-class)** ✅ validated

- [x] `@njit` functions called from Python (cache at the dispatcher boundary; `@cache` goes below `@njit`)
- [x] ndarray inputs/results, scalars
- [x] Dispatcher identity from Python function + closure captures + semantically relevant compile options (`fastmath`, `parallel`, `boundscheck`, `error_model`, `nopython`, `forceobj`, `locals=` type forcing)
- [x] Coexistence with Numba `cache=True` (compilation cache ⊥ result cache)
- [x] `parallel=True` and `fastmath` supported; changing a semantic option invalidates persisted stale results
- [x] Persistence, TTL, memory limits, metrics, invalidation all working for Numba workloads
- [x] Cold/warm JIT distinction in metrics, per specialization (a new-dtype compile on a later call is cold, never folded into savings baselines) and in the benchmark suite

**Stretch goal**

- [x] `func.cache.profile(...)` — shipped in V1.1 (Phase 2)

## Phase 2 — V1.1: explain more, depend on data

Goal: deepen the differentiators.

- [x] `func.cache.profile()`: warm-recompute vs. cache-hit timing (key generation + backend read), net savings, verdict (worth it / marginal / not worth it), primary-cost diagnosis (names the dominant hit cost, e.g. hashing a large ndarray), and an actionable recommendation. Runs the function to measure (warmed up, JIT excluded); never touches `stats()`, restores cache state
- [x] `func.cache.inspect()` — entry browsing: newest-first listing of `CacheEntryView` (key digest, created/age, size, remaining TTL, dependency fingerprints), header-only reads, pure observation, quarantined entries omitted
- [x] `depends_on=[...]` external dependency invalidation: files (mtime / size / content hash), environment variables, package versions, user-defined tokens — declared via bare paths or `cachau.file/env/package/token`; fingerprints stored per-entry (read header-only), compared on read, surfaced as a distinct `miss_dependency_changed` reason and named in `explain()`
- [x] Polars hashing support: native content identity for `polars.DataFrame`/`Series` (schema + row hashes, order-sensitive), native size estimation via `estimated_size()`, `profile()` culprit naming, LazyFrame fails loudly with guidance; cross-process digest stability pinned by test
- [x] Richer `explain()`: `evicted` reason (LRU-dropped vs never-cached), and per-changed-dependency fingerprint diffs (`{label: (stored, current)}`, rendered `label (before -> after)`)
- [ ] Notebook polish: cell re-runs never destroy useful persistent caches; code changes invalidate understandably
- [x] Benchmark suite with honest methodology (compile → warm up → benchmark; cold JIT reported separately) — see [BENCHMARKS.md](BENCHMARKS.md) and `benchmarks/`
- [x] `explain()` surfaces `write_errors` on a `not_found` miss — a broken store (unwritable persist dir, unpicklable results) no longer looks identical to a cold cache

## Phase 3 — V2: Numba Level B and hardening

Goal: broaden supported types with the same correctness bar. Nothing is "supported" without deterministic hashing, serialization round-trip, stability tests, and benchmarks.

- [ ] `numba.typed.List` adapter
- [ ] `numba.typed.Dict` adapter
- [ ] jitclass via explicit keys / field-based adapters / custom serializers (never compiler internals)
- [ ] Extension-type adapter API
- [ ] Custom serializer registration (Arrow/Parquet, native NumPy formats)
- [ ] Mutation policy: detect-and-reject where possible, documented opt-in otherwise
- [x] Transitive code changes, declared mitigation (#27): `cachau.code(helper)` fingerprints a callable's implementation as a dependency, and `profile()` flags same-package module-level functions called by global lookup that are neither closure-captured nor declared
- [ ] Opt-in call-graph fingerprinting: automatic transitive invalidation without declaring each helper
- [x] Cross-process single-flight candidate (#35): `coalesce="processes"` — advisory lock files, one elected computer per key, holder heartbeats its lock (stale = stopped beating, decoupled from compute duration), bounded waits derived from observed compute time; every failure degrades to redundant compute, never a hang. Validated by the downstream harness's three scenarios running as real OS processes against the shipped implementation (#46: K=6 cold burst, killed holder, wedged-alive holder)
- [x] `verify=` mode (#28): sample-recompute HITs and compare by content; a mismatch warns (`CacheVerificationWarning`), counts as `miss_verification_failed`, and the fresh value replaces the entry — catches both transitive-code false HITs and nondeterminism
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
