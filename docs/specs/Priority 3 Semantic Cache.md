You are an Elite Python Architect. Build a production-grade Semantic Cache module
for a Local AI QA Agent running on Windows 11 Native (NO Docker, NO Linux paths).
Use pathlib.Path for all filesystem operations.

════════════════════════════════════════════════════════════
KNOWLEDGE BASE CONSTRAINTS (treat as immutable spec):
════════════════════════════════════════════════════════════

1. DUAL-THRESHOLD ROUTING (from E2EGen AI spec):
   - s >= 0.95  → CACHE_HIT      : return cached playwright code immediately (zero LLM call)
   - 0.50 <= s < 0.95 → GUIDED   : inject cached entry as few-shot context, call LLM with reference
   - s < 0.50   → CACHE_MISS     : fresh LLM inference, no context injection
   After any successful LLM generation → upsert result back into LanceDB (self-healing loop)

2. VECTOR DB (from LanceDB spec):
   - Engine      : lancedb (embedded, serverless, Apache Arrow columnar, zero-copy)
   - Index type  : IVF_HNSW_SQ  (Cosine metric)
   - num_partitions: sqrt(N) where N = current row count (recompute on every index rebuild)
   - Embedding dim: 384 (all-MiniLM-L6-v2)
   - Rebuild index trigger: every 50 new upserts

3. EMBEDDING MODEL (from E2EGen AI spec):
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - Instantiate ONCE at module level (singleton), CPU-bound, thread-safe
   - Encode with normalize_embeddings=True for correct cosine similarity

4. SCHEMA (Pydantic V2 + LanceDB typed schema):
   Fields:
     - cache_id      : str   (MD5 hash of normalized query)
     - query_text    : str   (original natural language intent)
     - query_vector  : list[float]  (384-dim, stored as lancedb vector type)
     - playwright_code: str  (deterministic generated code)
     - hit_count     : int   (incremented on every CACHE_HIT)
     - confidence_score: float (cosine similarity score of last retrieval)
     - created_at    : str   (ISO 8601)
     - updated_at    : str   (ISO 8601)
     - version       : int   (incremented on every upsert/heal)
     - metadata      : str   (JSON string: source url, test_type, agent_mode)

5. COSINE SIMILARITY:
   Compute as: s = 1 - cosine_distance(query_vec, stored_vec)
   Do NOT rely on LanceDB's internal distance field blindly —
   re-derive the score from the _distance column using: s = 1 - row["_distance"]

════════════════════════════════════════════════════════════
FILE TO CREATE:
════════════════════════════════════════════════════════════
src/cache/semantic_cache.py

════════════════════════════════════════════════════════════
IMPLEMENTATION REQUIREMENTS:
════════════════════════════════════════════════════════════

CLASS: SemanticCache
  Constructor params:
    - db_path: Path  (default: Path("data/semantic_cache.lance"))
    - hit_threshold: float = 0.95
    - guided_threshold: float = 0.50
    - index_rebuild_interval: int = 50   (upserts between index rebuilds)

  ASYNC METHODS (all I/O must be async using asyncio.to_thread for sync lancedb ops):

  1. async initialize() -> None
     - Create LanceDB table if not exists using the Pydantic schema
     - Build IVF_HNSW_SQ index if row count > 10
     - Set self._upsert_counter from current row count % index_rebuild_interval

  2. async lookup(query: str) -> CacheLookupResult
     - Normalize query (lowercase, strip, collapse whitespace)
     - Encode query to 384-dim vector
     - Search LanceDB top-1 (limit=1, metric="cosine")
     - Compute s = 1 - row["_distance"]
     - Return CacheLookupResult with:
         route: Literal["CACHE_HIT", "GUIDED", "CACHE_MISS"]
         cached_entry: Optional[CacheEntry]   (None on CACHE_MISS)
         score: float
         query_vector: list[float]

  3. async upsert(
       query: str,
       playwright_code: str,
       metadata: dict,
       existing_entry: Optional[CacheEntry] = None
     ) -> CacheEntry
     - If existing_entry is provided → increment version + hit_count, update updated_at
     - If new → generate cache_id = MD5(normalized_query), version=1, hit_count=0
     - Write to LanceDB (overwrite by cache_id using delete + add pattern)
     - Increment self._upsert_counter
     - If counter >= index_rebuild_interval → call _rebuild_index()
     - Return the upserted CacheEntry

  4. async invalidate(cache_id: str) -> bool
     - Hard-delete row by cache_id from LanceDB table
     - Return True if deleted, False if not found

  5. async get_stats() -> CacheStats
     - Return total_entries, total_hits (sum of hit_count), 
       avg_confidence (mean of confidence_score), 
       index_health: Literal["INDEXED", "UNINDEXED", "NEEDS_REBUILD"]

  PRIVATE METHOD:
  6. async _rebuild_index() -> None
     - Recompute num_partitions = max(1, int(sqrt(current_row_count)))
     - Build IVF_HNSW_SQ index with metric="cosine"
     - Reset self._upsert_counter = 0
     - Log rebuild event with row count

PYDANTIC MODELS (same file, top section):
  - CacheEntry(BaseModel)   — mirrors the LanceDB schema fields
  - CacheLookupResult(BaseModel) — route, cached_entry, score, query_vector, latency_ms
  - CacheStats(BaseModel)   — total_entries, total_hits, avg_confidence, index_health

════════════════════════════════════════════════════════════
INTEGRATION STUB — also create: src/cache/__init__.py
════════════════════════════════════════════════════════════
Export: SemanticCache, CacheLookupResult, CacheEntry, CacheStats

════════════════════════════════════════════════════════════
ALSO CREATE: tests/unit/test_semantic_cache.py
════════════════════════════════════════════════════════════
Use pytest + pytest-asyncio. Write 5 tests:
  1. test_cache_miss_on_empty_db         — lookup on empty table returns CACHE_MISS
  2. test_cache_hit_above_threshold      — upsert then lookup identical query → CACHE_HIT s>=0.95
  3. test_guided_mode_similar_query      — upsert "login test", lookup "test the login flow" → GUIDED
  4. test_upsert_increments_version      — upsert same cache_id twice → version == 2
  5. test_invalidate_removes_entry       — upsert then invalidate → lookup returns CACHE_MISS

════════════════════════════════════════════════════════════
DEPENDENCY INSTALL BLOCK (output as bash):
════════════════════════════════════════════════════════════
pip install lancedb sentence-transformers pydantic pytest pytest-asyncio

════════════════════════════════════════════════════════════
HARD RULES:
════════════════════════════════════════════════════════════
- Python 3.11+, full type hints, async-first
- NO Docker, NO WSL, NO Linux paths — use pathlib.Path only
- NO OpenAI, NO vLLM — embedding is local CPU via sentence-transformers
- COMPLETE files only — zero placeholders, zero "# ... existing code ..."
- All LanceDB sync calls wrapped in asyncio.to_thread()
- Structured logging via Python logging module (not print)
- Every public method has a docstring with Args/Returns/Raises