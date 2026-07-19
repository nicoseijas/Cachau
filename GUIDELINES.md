# Cachau — Design & Engineering Guidelines

These guidelines define how Cachau must behave and how contributions should be designed. They exist so that every feature stays consistent with the project's core promise: **correct, bounded, observable, explainable caching for data workloads.**

---

## 1. API design

- The common case must be minimal:

  ```python
  @cache
  def expensive(x):
      ...
  ```

- Configuration is declarative and progressive:

  ```python
  @cache(ttl="10m", max_memory="1GB", persist=True)
  def expensive(x):
      ...
  ```

- Complexity appears only as the user needs it. Backend objects, external configuration, serializers, and config files must **never** be required for basic usage.

- Per-function secondary interface:

  ```python
  func.cache.stats()
  func.cache.clear()
  func.cache.invalidate(...)
  func.cache.inspect()
  func.cache.explain(...)
  func.cache.profile(...)
  ```

- Global operations:

  ```python
  cache.clear()
  cache.stats()
  cache.configure(...)
  ```

## 2. Cache identity and keys

A correct key conceptually represents:

```
function identity + implementation/version + normalized arguments + relevant external dependencies
```

- Semantically equivalent calls must share a key. Normalize arguments through signature binding (kwargs vs. positional must not fragment the cache).
- Optimized, native hashing for: `numpy.ndarray`, `pandas.DataFrame`/`Series`, Polars, `pathlib.Path`, dataclasses, `dict`/`list`/`tuple`, and primitives.
- **Never blindly serialize a huge object just to compute a key.**
- Explicit keying must be supported:

  ```python
  @cache(key=lambda dataset, version: version)
  def process(dataset, version):
      ...
  ```

- Explicit exclusion of irrelevant arguments:

  ```python
  @cache(ignore=["logger", "progress_callback"])
  def run(data, logger=None, progress_callback=None):
      ...
  ```

- **Never silently ignore unhashable arguments.** Fail loudly or require explicit handling.

## 3. Namespaces

- Different functions must never collide, even with identical arguments. Conceptual identity: `package.module.function + function fingerprint + arguments`.
- Stable namespaces allow refactors without unintended invalidation:

  ```python
  @cache(namespace="features.v2")
  def build_features(...):
      ...
  ```

## 4. TTL

- Optional; accepts simple readable values: `ttl=60`, `"30s"`, `"10m"`, `"2h"`, `"7d"`.
- TTL starts when the entry was **computed and committed**, not when execution began.
- Expiration is **lazy** by default: check on access; if expired, treat as miss and remove/mark stale. No background workers unless a real need is demonstrated.

## 5. Memory limits and eviction

- Bounded resources are a core feature: `@cache(max_memory="1GB")` or `cache.configure(max_memory="4GB")`.
- Document precisely what "size" means (serialized size, approximate in-memory size, deep size, or physical backend size).
- **LRU** is the initial policy — understandable and predictable. Do not add more policies before demonstrated need.
- If a single entry exceeds total capacity: compute the result, return it, **do not cache it**, and record `cache_skip_oversized`. Never flush the whole cache to fit one pathological entry.

## 6. Persistence

- Trivial to enable: `@cache(persist=True)` or `@cache(persist="./.cache")`.
- Must survive interpreter, notebook, and process restarts.
- Writes are **atomic**: serialize to temp file → sync where appropriate → atomic rename.
- Per-entry metadata: key, `created_at`, `expires_at`, `last_access`, size, serializer, function identity, function fingerprint, format version.
- The persistent format has **explicit versioning**. Incompatibility degrades to a miss or controlled invalidation — never mysterious errors.

## 7. Invalidation

Invalidation is a first-class concept:

- Per function: `expensive.cache.clear()`
- Per invocation: `expensive.cache.invalidate(x=1)`
- Global: `cache.clear()`
- **Changing a function's implementation invalidates previous results by default.** A result produced by `return x * 2` must not be reused after the code becomes `return x * 3`.
- External dependency invalidation:

  ```python
  @cache(depends_on=["data.csv"])
  def load_data():
      ...
  ```

  Possible fingerprints: mtime, size, content hash, environment variable, package version, user-defined token.

- Automatic detection is **conservative**. Explicitness is preferable to unreliable magic.

## 8. Observability and metrics

A cache that is hard to observe is hard to trust.

- `func.cache.stats()` reports: hits, misses, hit rate, writes, skipped writes, expirations, invalidations, evictions, serialization/deserialization errors, entry count, current bytes, computation time, estimated time saved.
- Misses must distinguish causes: `miss_not_found`, `miss_expired`, `miss_invalidated`, `miss_code_changed`, `miss_dependency_changed`, `miss_corrupt`.
- `func.cache.explain(args)` answers: HIT or MISS? Why? Which dependency changed? Did the code change? How big is it? How much TTL remains? What was invalidated or evicted?
- Metrics observe without changing behavior.

## 9. Serialization

- Extensible but invisible in normal cases. Priorities: Python types → NumPy → pandas → common objects.
- Strategies: pickle/cloudpickle, native NumPy formats, Arrow/Parquet, custom serializers.
- **Correctness beats micro-optimization.**
- If the function computes correctly but serialization fails: return the result, record `cache_write_error`, optionally warn. The cache is an optimization, not a correctness dependency.

## 10. Concurrency: single-flight

- Multiple concurrent callers for the same absent key should not all compute:

  ```
  Caller A → MISS → computes
  Caller B → same key → waits
  Caller C → same key → waits
  A commits the result
  B/C reuse the result
  ```

- Synchronization is **per key**. Never a global lock that serializes independent work.

## 11. Purity, exceptions, and mutability

- Exceptions are **not cached** by default.
- Correctness assumes the result is determined by the key and declared dependencies. Do not try to magically detect nondeterminism; random seeds should be explicit arguments.
- Functions that mutate inputs in-place cannot be treated as conventional memoization. Default contract: functions are pure with respect to their key. Mutation: reject when detectable, require opt-in, or document as outside the standard guarantee.
- Define what happens when a caller mutates a mutable result retrieved from the cache (especially for the in-memory backend).

## 12. Notebook experience

Notebooks are a primary target:

- `func.cache.stats() / clear() / inspect() / explain(...)` must feel natural in a cell.
- Re-running a cell must not unexpectedly destroy a useful persistent cache.
- Changing the code **should** invalidate stale results, in an understandable way.

## 13. Architecture

Conceptual layers:

```
Decorator / Public API
  → Invocation Normalization
  → Key Builder
  → Dependency Fingerprinting
  → Cache Policy
  → Storage Backend
  → Serializer
  → Metrics / Events
```

Separation of responsibilities:

- The decorator does not implement storage.
- The backend does not decide function semantics.
- The serializer does not decide invalidation.
- Metrics observe without changing behavior.

Minimal backend interface:

```python
class CacheBackend:
    def get(key): ...
    def set(key, value, metadata): ...
    def delete(key): ...
    def clear(namespace=None): ...
    def iter_entries(...): ...
```

V1 ships `MemoryBackend` + `DiskBackend`. Redis, S3, databases, and distributed caching stay out until the local model is consolidated.

## 14. Numba support

Numba is a **first-class workload**:

```python
@cache(ttl="1h", max_memory="4GB", persist=True)
@njit
def simulate(values, iterations):
    ...
```

- Cachau operates at the **Python → Numba dispatcher boundary**. The initial goal is *not* making `@cache` callable from nopython mode — it is caching Numba workloads correctly, fast, and transparently.
- **Dispatcher identity**: derived from the original Python function, semantically relevant compile options (`fastmath`, `parallel`, `boundscheck`, `error_model`), defaults, and namespace/version — never `repr()` or memory identity. Changing semantically relevant configuration invalidates stale persisted results.
- **Compilation cache vs. result cache** are independent: Numba `cache=True` caches machine code; Cachau caches computed results. They coexist — a Cachau HIT skips execution; a MISS still benefits from Numba's compilation cache.
- **Honest metrics**: distinguish cold JIT vs. warm JIT. Measure lookup, key generation, compile/dispatch, execution, serialization, and deserialization separately. Never repeatedly count one-time compilation cost as normal execution cost. Standard benchmark: compile → warm up → benchmark; report cold JIT separately.
- **ndarray identity** includes at minimum dtype, shape, and content; strides/memory order only when semantically relevant. Equal bytes with different dtype or shape must not collide. C/F/non-contiguous layout must not fragment the cache unless it affects observable semantics — identity follows semantics, not compiler internals.
- **Advanced types** (`numba.typed.List`, `typed.Dict`, jitclass, extension types): support is explicit and progressive. Do not declare support without deterministic hashing, serialization round-trip, stability tests, and benchmarks. For jitclass, prefer explicit keys or field-based adapters over compiler internals.
- `parallel=True` works at the Python boundary without Cachau altering thread configuration. Do not needlessly bind persisted results to CPU/LLVM/machine code — prefer semantic identity over compilation-artifact identity.

## 15. Cache economics

The fundamental rule:

```
T_key + T_lookup + T_deserialize < T_recompute
```

Example: a Python function takes 10 s; with Numba it takes 50 ms; hashing a 2 GB array takes 800 ms — caching makes it *worse*. Cachau should measure (or make it easy to measure): keying cost, lookup, deserialize, warm recomputation, net time saved. `cache.profile()` embodies this philosophy: not just caching, but explaining when caching is a bad decision.

## 16. Testing requirements

Mandatory semantic tests:

- HIT after first computation; MISS on different arguments
- kwargs/positional normalization
- TTL expiration; TTL starts after computation
- Memory eviction; oversized-entry skip
- Persistence across processes; corruption recovery
- Invalidation: code change, dependency change, manual
- Same-key concurrency (single-flight)
- Serialization failure fallback
- `stats()`, `clear()`, namespace isolation

Numba-specific tests:

- Warm/cold JIT distinction
- dtype/shape/layout identity
- `fastmath` and `parallel` variants
- Large arrays
- Typed containers (once supported)
- Cross-process persistence
- Keying cost vs. recomputation

## 17. Feature acceptance bar

Every new feature must justify its complexity through real use cases. When evaluating a proposal, ask:

1. Does it preserve **correctness before hit rate**?
2. Can the system still **explain** what happened?
3. Is the behavior **predictable and bounded**?
4. Is it needed for the narrow mission — *"a pleasant, robust function cache for expensive Python data workloads"* — or is it scope creep toward an anti-goal (distributed cache, workflow engine, artifact registry)?
