# SPEC_CORE — Universal Local AI QA Agent (Original Spec)

You are an Elite AI & QA Automation Architect. Build a complete "Universal Local AI QA Agent"
system from scratch based on the exact specifications below. Write ALL files completely —
no placeholders, no `# ... existing code ...`. Every file must be runnable.

---

## SYSTEM CONSTRAINTS (HARD LIMITS — never violate)

- **VRAM**: 6GB (GPU) — Primary LLM must fit within 4.5GB leaving 1.5GB for KV cache
- **RAM**: 16GB system RAM
- **OS**: Windows 11 WSL2 (Ubuntu 22.04 paths and commands)
- **Language**: Python 3.11+ only, async-first everywhere (asyncio/async-await)
- **LLM Runtime**: Ollama (localhost:11434) — NO external API calls ever
- **Primary Model**: `qwen2.5-coder:7b-instruct-q4_K_M` for code/reasoning tasks
- **Embedding Model**: `nomic-embed-text` via Ollama
- **Vision Strategy**: AxTree-First — use `page.aria_snapshot()` as primary perception.
  VLM (CPU-offloaded via llama.cpp) activates ONLY as last resort when AxTree confidence < 0.4
- **Mode**: Hybrid Dual-Mode (Generative + Execution/Self-Healing both must work)
- **Fine-tune**: Synthetic dataset generation pipeline (start from zero, no existing data)

---

## PROJECT STRUCTURE (create exactly this)

```
qa-agent/
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── adapter.py          # Provider-agnostic LLM adapter (OpenAI-compatible)
│   │   ├── structured.py       # Pydantic schemas + structured output enforcement
│   │   └── prompt_templates.py # PTCF/RTF prompt templates for all agent roles
│   ├── browser/
│   │   ├── __init__.py
│   │   ├── manager.py          # Playwright async browser lifecycle manager
│   │   ├── resource_filter.py  # Block images/media/fonts to save RAM
│   │   ├── auth_manager.py     # storageState RBAC manager (multi-role)
│   │   └── ax_extractor.py     # Accessibility Tree extraction + pruning
│   ├── healing/
│   │   ├── __init__.py
│   │   ├── engine.py           # Self-healing orchestrator
│   │   ├── fuzzy_matcher.py    # Phase 1: Jaro-Winkler fuzzy matching (threshold 0.85)
│   │   └── ai_healer.py        # Phase 2: AI-powered locator recovery via AxTree
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── store.py            # LanceDB vector store (local, no server)
│   │   ├── ingestion.py        # Document ingestion + semantic chunking pipeline
│   │   └── retriever.py        # Hybrid search (semantic + BM25)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py            # LangGraph state machine (Planner→Generator→Executor→Healer)
│   │   ├── planner.py          # Planner Agent: requirement → test plan (Markdown)
│   │   ├── generator.py        # Generator Agent: test plan → Playwright Python code
│   │   └── healer.py           # Healer Agent: error + AxTree → fixed code
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_gen.py    # Synthetic QA dataset generator (JSONL ChatML format)
│   │   └── pii_masker.py       # PII masking before any LLM call (PDPA compliance)
│   ├── finetune/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Unsloth QLoRA 4-bit fine-tuning pipeline
│   │   └── export.py           # Merge LoRA → GGUF export for Ollama
│   └── reporting/
│       ├── __init__.py
│       └── allure_reporter.py  # Allure integration: attach AxTree, screenshots, AI reasoning
├── config/
│   ├── agent.yaml              # All tunable parameters (timeouts, thresholds, concurrency)
│   └── roles.yaml              # RBAC role definitions + storageState paths
├── locators/
│   └── locators.json           # External locator repository (atomic read/write with file lock)
├── tests/
│   └── example_hrm_payroll.py  # Example test: Thai HRM/Payroll system with clock emulation
├── scripts/
│   ├── setup_wsl2.sh           # WSL2 setup: Ollama, Python deps, Playwright browsers
│   └── pull_models.sh          # Pull required Ollama models
├── main.py                     # CLI entry point (argparse: --mode generate|run|finetune|heal)
├── pyproject.toml              # uv-compatible project config
└── .env.example                # Environment variables template
```

---

## MODULE SPECIFICATIONS

### 1. `src/llm/adapter.py`
- Class `OllamaAdapter` with `async def generate()`, `async def stream()`, `async def embed()`
- OpenAI-compatible client pointing to `http://localhost:11434/v1`
- Retry logic via `tenacity` (3 retries, exponential backoff)
- Temperature hardcoded to `0.1` for deterministic QA outputs
- Semaphore lock to prevent concurrent VRAM OOM (max 2 simultaneous inference calls)
- VRAM budget guard: before each call, check `nvidia-smi` free memory; if < 500MB, wait

### 2. `src/llm/structured.py`
- All Pydantic V2 schemas for structured outputs:
  - `TestPlan(steps: list[TestStep], estimated_complexity: Literal["low","medium","high"])`
  - `PlaywrightScript(reasoning: str, code: str, locators_used: list[str])` — reasoning field MUST come before code field (left-to-right generation quality)
  - `HealedLocator(original: str, healed: str, confidence: float, method: Literal["fuzzy","ai","vlm"])`
  - `SyntheticQAExample(instruction: str, input_context: str, output_code: str, domain: str)`
- Function `enforce_json_output(schema, raw_text) -> BaseModel` with fallback repair logic

#### Sprint 1 Update — Instructor Integration (2026-05-23)

As of Sprint 1 Day 1-2, the primary structured output engine is `src/llm/instructor_client.py`
(`structured_output_engine: "instructor"` in `config/agent.yaml`). The legacy 3-stage repair
pipeline in `enforce_json_output()` remains as a fallback (`structured_output_engine: "legacy_repair"`).

Canonical schema definitions live in `src/llm/schemas.py` (Pydantic V2, `extra="forbid"`).
`src/llm/structured.py` re-exports these schemas for backward compatibility and retains the
repair utilities for the legacy path.

Sprint 1 results (n=10 pilot dataset): instructor pass_rate=0.90 vs legacy 0.80; 
parse_fail_rate=0.10 vs 0.20 legacy; latency +52% (FAIL on sprint targets, no regression).

### 3. `src/browser/manager.py`
- Async context manager `BrowserManager`
- Launch args for WSL2: `--disable-dev-shm-usage`, `--no-sandbox`, `--disable-gpu`
- Always headless mode
- Default timeout: 15000ms (from config)
- Browser context pool (max 3 concurrent contexts based on RAM constraint)

### 4. `src/browser/resource_filter.py`
- Block resource types: `["image", "media", "font", "stylesheet"]` via `page.route()`
- Allow: `["document", "script", "xhr", "fetch", "websocket"]`
- Log bytes saved per session for monitoring

### 5. `src/browser/auth_manager.py`
- Load/save `storageState` per role from `config/roles.yaml`
- Roles: `admin`, `hr_manager`, `payroll_officer`, `employee`, `auditor`
- API-based login (POST to configurable auth endpoint) — never UI login
- Token expiry detection + auto-refresh
- Atomic file write for storageState (write to `.tmp` then `os.replace()`)

### 6. `src/browser/ax_extractor.py`
- `async def extract_axtree(page) -> str` using `page.aria_snapshot()`
- `async def prune_axtree(raw_yaml: str, max_nodes: int = 200) -> str` — remove decorative/hidden nodes
- `async def compute_semantic_density(axtree: str) -> float` — ratio of interactive nodes
- If semantic_density < 0.4: trigger VLM fallback signal (do NOT call VLM here, just signal)
- Token count estimation before sending to LLM (warn if > 4000 tokens)

### 7. `src/healing/engine.py`
- `async def heal(page, failed_locator: str, action: str) -> HealedLocator`
- Phase 1: call `fuzzy_matcher.py` → if confidence >= 0.85, return immediately
- Phase 2: call `ai_healer.py` with AxTree context
- Phase 3 (VLM last resort): signal to use CPU-based VLM only if Phase 2 confidence < 0.5
- State Validation Loop: after healing, verify action succeeded by checking URL change or new element appearance
- Update `locators/locators.json` atomically on successful heal
- Log all healing events with before/after AxTree diff

### 8. `src/healing/fuzzy_matcher.py`
- Implement Jaro-Winkler distance (use `jellyfish` library)
- Scan current AxTree for elements with similar `role`, `name`, `aria-label`, `data-testid`
- Return best match with confidence score
- Threshold: 0.85 for auto-accept, 0.5-0.85 for AI fallback, < 0.5 for VLM fallback

### 9. `src/rag/store.py`
- LanceDB local database at `~/.qa-agent/vector_db/`
- Table schema: `(id, content, embedding, source, doc_type, metadata_json, created_at)`
- `async def upsert(chunks: list[RAGChunk])` — batch insert with deduplication
- `async def search(query: str, top_k: int = 5) -> list[RAGChunk]`
- No external server needed (LanceDB embedded mode)

### 10. `src/rag/retriever.py`
- Hybrid search: 70% semantic (vector cosine) + 30% BM25 keyword
- Reciprocal Rank Fusion for merging results
- Context window budget: max 2000 tokens of retrieved context per LLM call

### 11. `src/agents/graph.py`
- LangGraph `StateGraph` with typed state: `QAAgentState`
- Nodes: `planner_node`, `generator_node`, `executor_node`, `healer_node`, `reporter_node`
- Edges with conditional routing:
  - `executor → healer` if test fails AND retry_count < 3
  - `healer → executor` after fix (retry loop)
  - `executor → reporter` on success or max retries exceeded
- Generative Mode entry: `planner → generator → executor`
- Execution Mode entry: `executor → (healer if needed) → reporter`
- Persist state to `~/.qa-agent/sessions/` for resume capability

### 12. `src/agents/generator.py`
- Input: `TestPlan` from planner + RAG context
- Enforce `PlaywrightScript` structured output
- Generated code rules (inject as system prompt):
  - ONLY use `get_by_role()`, `get_by_label()`, `get_by_text()`, `get_by_test_id()`
  - NEVER use CSS selectors or XPath
  - NEVER use `page.wait_for_timeout()` — use `wait_for_load_state()` or `wait_for_selector()`
  - All assertions at test level, never inside page objects
- Static validation: parse generated Python AST to check for forbidden patterns before returning
- If validation fails: retry LLM call with error feedback (max 3 retries)

### 13. `src/data/synthetic_gen.py`
- Generate synthetic QA training examples in JSONL ChatML format
- Domain seeds (hardcoded): `["hrm_login", "payroll_calculation", "leave_request", "employee_profile", "rbac_permission_check", "thai_tax_calculation", "overtime_calculation"]`
- For each domain seed, generate:
  - 10 happy path examples
  - 5 negative/edge case examples
  - 3 RBAC boundary examples
- Output format per example:
```json
  {"messages": [
    {"role": "system", "content": "You are an expert Playwright QA engineer..."},
    {"role": "user", "content": "<requirement_description>"},
    {"role": "assistant", "content": "<complete_playwright_python_code>"}
  ]}
```
- Total target: ~500 examples minimum
- Apply PII masking to all generated content before saving

### 14. `src/finetune/trainer.py`
- Unsloth + QLoRA 4-bit fine-tuning
- Base model: `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
- LoRA config: `r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"]`
- Training args: `per_device_train_batch_size=1, gradient_accumulation_steps=4, max_steps=200`
- 6GB VRAM guard: check available VRAM before starting, raise clear error if < 5.5GB free
- Save checkpoints to `~/.qa-agent/checkpoints/`

### 15. `src/finetune/export.py`
- Merge LoRA weights into base model
- Export to GGUF Q4_K_M format
- Auto-create Ollama Modelfile and register as `qa-agent-coder:latest`
- Test inference after registration (send 1 test prompt, verify response)

### 16. `tests/example_hrm_payroll.py`
- Complete example test using the agent system for Thai HRM/Payroll
- Include clock emulation: `page.clock.install()` + `page.clock.fast_forward("30:00")`
- Test scenarios:
  1. Admin login → verify dashboard (RBAC: admin role)
  2. HR Manager creates employee record
  3. Payroll Officer runs monthly payroll calculation
  4. Verify Thai SSO deduction (max 875 THB/month as of 2026)
  5. Verify progressive income tax calculation (0-35% brackets)
  6. Employee role cannot access payroll module (RBAC boundary test)
- Each test wraps actions in self-healing decorator

### 17. `main.py` CLI
```
python main.py --mode generate --requirement "test login flow" --url http://localhost:3000 --role admin
python main.py --mode run --script tests/example_hrm_payroll.py
python main.py --mode finetune --dataset ~/.qa-agent/datasets/synthetic.jsonl
python main.py --mode heal --url http://localhost:3000 --locator-file locators/locators.json
```

### 18. `config/agent.yaml`
Include all tunable parameters:
- `llm.model`, `llm.base_url`, `llm.temperature`, `llm.max_tokens`
- `llm.semaphore_limit` (default: 2)
- `llm.vram_buffer_mb` (default: 500)
- `browser.headless`, `browser.timeout_ms`, `browser.max_contexts`
- `healing.fuzzy_threshold`, `healing.max_retries`, `healing.vlm_threshold`
- `rag.top_k`, `rag.semantic_weight`, `rag.bm25_weight`
- `reporting.allure_results_dir`

### 19. `scripts/setup_wsl2.sh`
Complete setup script for WSL2 Ubuntu:
- Install Ollama
- Install Python 3.11 via deadsnakes PPA
- Install `uv` package manager
- Run `uv sync` to install all deps
- Install Playwright browsers: `playwright install chromium --with-deps`
- Create required directories: `~/.qa-agent/{vector_db,sessions,datasets,checkpoints}`
- Verify all services are running

### 20. `scripts/pull_models.sh`
- `ollama pull qwen2.5-coder:7b-instruct-q4_K_M`
- `ollama pull nomic-embed-text`
- Verify each model with a test inference call
- Print VRAM usage after each pull

---

## DEPENDENCIES (`pyproject.toml`)

```toml
[project]
name = "qa-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "playwright>=1.49.0",
    "langchain>=0.3.0",
    "langgraph>=0.2.0",
    "lancedb>=0.13.0",
    "pydantic>=2.9.0",
    "httpx>=0.27.0",
    "tenacity>=9.0.0",
    "jellyfish>=1.1.0",       # Jaro-Winkler fuzzy matching
    "rank-bm25>=0.2.2",       # BM25 keyword search
    "allure-pytest>=2.13.0",
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.2",
    "loguru>=0.7.2",           # Structured logging
    "unsloth[colab-new]>=2024.12.0",  # Fine-tuning (install separately with GPU)
    "filelock>=3.16.0",        # Atomic file writes for locators.json
    "aiofiles>=24.1.0",
    "rich>=13.9.0",            # CLI output formatting
]
```

---

## CRITICAL IMPLEMENTATION RULES

1. **Every async function** must have proper `try/except` with `loguru` logging
2. **Semaphore** on all LLM calls — max 2 concurrent to protect 6GB VRAM
3. **AxTree pruning** before every LLM call — never send raw HTML
4. **Reasoning field always first** in all Pydantic schemas before answer/code fields
5. **Atomic writes** for `locators.json` using `filelock` — prevent race conditions in parallel runs
6. **Temperature = 0.1** everywhere — deterministic outputs for QA tasks
7. **No hardcoded waits** — use Playwright's built-in auto-waiting exclusively
8. **PII masking** applied before any content enters LLM context
9. **WSL2-specific**: all paths use Linux format (`~/.qa-agent/`), browser launch includes `--disable-dev-shm-usage`
10. **Resource blocking** active by default on every page — only disable via config flag

---

## DELIVERY ORDER

Build in this exact order (each phase must be complete and importable before next):

**Phase 1**: `pyproject.toml` → `config/agent.yaml` → `config/roles.yaml` → `.env.example`

**Phase 2**: `src/llm/adapter.py` → `src/llm/structured.py` → `src/llm/prompt_templates.py`

**Phase 3**: `src/browser/resource_filter.py` → `src/browser/ax_extractor.py` → `src/browser/auth_manager.py` → `src/browser/manager.py`

**Phase 4**: `src/healing/fuzzy_matcher.py` → `src/healing/ai_healer.py` → `src/healing/engine.py`

**Phase 5**: `src/rag/store.py` → `src/rag/ingestion.py` → `src/rag/retriever.py`

**Phase 6**: `src/agents/planner.py` → `src/agents/generator.py` → `src/agents/healer.py` → `src/agents/graph.py`

**Phase 7**: `src/data/pii_masker.py` → `src/data/synthetic_gen.py`

**Phase 8**: `src/finetune/trainer.py` → `src/finetune/export.py`

**Phase 9**: `src/reporting/allure_reporter.py`

**Phase 10**: `locators/locators.json` → `tests/example_hrm_payroll.py` → `main.py`

**Phase 11**: `scripts/setup_wsl2.sh` → `scripts/pull_models.sh`

After all files are written, provide:
1. Full `bash` command to run setup end-to-end
2. Verification checklist to confirm system is working
3. Example CLI command for each of the 4 modes
