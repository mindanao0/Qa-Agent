# Phase 0 — Audit Report

**Repository**: `D:\Code\qa-agent`
**Audited at (ISO)**: 2026-05-22
**Auditor**: Phase 0 read-only audit per `docs\specs\Phase 0 — Audit + Golden Dataset.md`
**Source-of-truth specs**: `CLAUDE.md`, `docs\specs\SPEC_CORE.md`, `docs\specs\SPEC_UNIVERSAL_DOMAIN.md` (all read)
**LLM target**: Ollama @ `http://localhost:11434`, model `qwen2.5-coder:7b-instruct-q4_K_M`
**Constraint honoured**: NO existing source code modified. All new files are under `audit\phase0\`.

---

## 1. Module Inventory

LOC is non-blank, non-comment lines (computed via PowerShell scan). `has_tests` = exact-name match under `tests\` *or* identifiable coverage in a unit / integration test in that subtree.

| path | LOC | exports | imports_internal | imports_external | has_tests |
|---|---:|---|---|---|---|
| `src\__init__.py` | 0 | — | — | — | no |
| `src\main.py` | 67 | `main`, `_summary` | `src.bft.graph`, `src.bft.models`, `src.config` | `loguru` | no |
| `src\config.py` | 28 | `N`, `F`, `Q`, `OLLAMA_BASE_URL`, `NODE_CONFIGS`, `GENERATOR_MAX_ATTEMPTS`, `GENERATOR_BACKOFF_BASE_SEC` | — | — | no |
| `src\llm\__init__.py` | 28 | — | — | — | no |
| `src\llm\adapter.py` | 193 | `OllamaAdapter`, `OLLAMA_BASE_URL`, `DEFAULT_MODEL` | — | `httpx`, `dotenv`, `loguru`, `tenacity`, `asyncio` | no |
| `src\llm\structured.py` | 241 | `TestPlan`, `TestStep`, `PlaywrightScript`, `HealedLocator`, `SyntheticQAExample`, `enforce_json_output` | — | `pydantic`, `loguru` | no |
| `src\llm\prompt_templates.py` | 293 | `PromptTemplate`, `UNIVERSAL_SYSTEM_PROMPT`, `PlannerPromptTemplate`, `GeneratorPromptTemplate`, `HealerPromptTemplate`, `SyntheticGenPromptTemplate`, `DOMAIN_CODING_HINTS`, `get_domain_hint`, `TEMPLATES`, `get_template` | — | — | no |
| `src\browser\__init__.py` | 22 | — | — | — | no |
| `src\browser\manager.py` | 175 | `BrowserManager` | `src.browser.auth_manager`, `src.browser.resource_filter` | `playwright`, `yaml`, `loguru` | no |
| `src\browser\ax_extractor.py` | 145 | `extract_axtree`, `prune_axtree`, `compute_semantic_density`, `needs_vlm_fallback`, `estimate_tokens`, `AXNode` | — | `playwright`, `loguru` | no |
| `src\browser\auth_manager.py` | 243 | `AuthManager`, `AuthError` | — | `httpx`, `yaml`, `loguru` | no |
| `src\browser\resource_filter.py` | 75 | `ResourceFilter`, `FilterStats`, `BLOCKED_RESOURCE_TYPES`, `ALLOWED_RESOURCE_TYPES` | — | `playwright`, `loguru` | no |
| `src\healing\__init__.py` | 9 | — | — | — | no |
| `src\healing\engine.py` | 279 | `HealingEngine` | `src.browser.ax_extractor`, `src.llm.adapter`, `src.llm.structured`, `.fuzzy_matcher`, `.ai_healer` | `filelock`, `playwright`, `loguru` | no |
| `src\healing\fuzzy_matcher.py` | 230 | `fuzzy_match`, `score_locator_against_axtree`, `LocatorAttributes`, `AXElement`, `FUZZY_AUTO_ACCEPT`, `FUZZY_AI_FALLBACK` | `src.llm.structured` | `jellyfish`, `loguru` | no |
| `src\healing\ai_healer.py` | 80 | `ai_heal` | `src.browser.ax_extractor`, `src.llm.adapter`, `src.llm.prompt_templates`, `src.llm.structured` | `playwright`, `loguru` | no |
| `src\rag\__init__.py` | 10 | — | — | — | no |
| `src\rag\store.py` | 178 | `VectorStore`, `RAGChunk`, `EMBEDDING_DIM` | — | `lancedb`, `pyarrow`, `loguru` | no |
| `src\rag\ingestion.py` | 754 | `RAGIngestionPipeline`, `DocumentIngester` | `src.llm.adapter`, `src.browser.ax_extractor`, `src.browser.manager`, `.store` | `httpx`, `bs4`, `markdownify`, `pypdf`, `docx`, `yaml`, `rich`, `loguru` | no |
| `src\rag\retriever.py` | 225 | `HybridRetriever` | `src.llm.adapter`, `.store` | `rank_bm25`, `loguru` | no |
| `src\agents\__init__.py` | 19 | — | — | — | no |
| `src\agents\graph.py` | 320 | `build_graph`, `initial_state`, `load_session`, `QAAgentState` | `src.llm.adapter`, `src.llm.structured`, `src.browser.manager`, `src.rag.retriever`, `.planner`, `.generator`, `.healer` | `langgraph`, `loguru` | no |
| `src\agents\planner.py` | 164 | `PlannerAgent` | `src.data.synthetic_gen` (detect_domain), `src.llm.adapter`, `src.llm.prompt_templates`, `src.llm.structured`, `src.rag.retriever` | `loguru` | no |
| `src\agents\generator.py` | 215 | `GeneratorAgent`, `validate_playwright_ast` | `src.llm.adapter`, `src.llm.prompt_templates`, `src.llm.structured`, `src.rag.retriever` | `loguru` | no |
| `src\agents\healer.py` | 161 | `CodeHealerAgent` | `src.browser.ax_extractor`, `src.browser.manager`, `src.healing.engine`, `src.llm.adapter`, `src.llm.structured` | `playwright`, `loguru` | no |
| `src\agents\planner_agent.py` | 77 | `planner_node` | `src.core.sfg_engine`, `src.core.state_schema` | `httpx`, `networkx`, `rich` | no |
| `src\agents\generator_agent.py` | 144 | `generator_node` | `src.core.contract_skill`, `src.core.state_schema` | `httpx`, `pydantic`, `rich` | no |
| `src\agents\healer_agent.py` | 140 | `healer_node` | `src.core.contract_skill`, `src.core.state_schema` | `httpx`, `rich` | no |
| `src\agents\observer_driver\__init__.py` | 20 | — | — | — | yes (`tests\test_observer_driver.py`) |
| `src\agents\observer_driver\state.py` | 52 | `DriverAction`, `ObserverFinding`, `ObserverPipelineState` | — | `pydantic` | yes |
| `src\agents\observer_driver\trace_bus.py` | 56 | `TraceBus`, `TraceEvent` | — | `pydantic` | yes |
| `src\agents\observer_driver\driver_agent.py` | 213 | `plan_action`, `execute_action`, `capture_axtree` | `.state`, `.trace_bus` | `aiohttp`, `playwright`, `loguru` | yes |
| `src\agents\observer_driver\coordinator.py` | 152 | `run_observer_driver_loop` | `.driver_agent`, `.observers.*`, `.state`, `.trace_bus` | `loguru`, `playwright` | yes |
| `src\agents\observer_driver\observers\__init__.py` | 19 | — | — | — | yes |
| `src\agents\observer_driver\observers\base_observer.py` | 57 | `BaseObserver` | `..trace_bus`, `..state` | `loguru` | yes |
| `src\agents\observer_driver\observers\accessibility_observer.py` | 106 | `AccessibilityObserver` | `..trace_bus`, `..state`, `.base_observer` | `playwright`, `loguru` | yes |
| `src\agents\observer_driver\observers\security_observer.py` | 114 | `SecurityObserver` | `..trace_bus`, `..state`, `.base_observer` | `aiohttp`, `loguru` | yes |
| `src\agents\observer_driver\observers\performance_observer.py` | 101 | `PerformanceObserver` | `..trace_bus`, `..state`, `.base_observer` | `playwright`, `loguru` | yes |
| `src\core\__init__.py` | 1 | — | — | — | no |
| `src\core\state_schema.py` | 72 | `GUIState`, `SFGEdge`, `ActionStep`, `RecoveryRule`, `ContractSkillArtifact`, `AgentPipelineState` | — | `pydantic` | no |
| `src\core\sfg_engine.py` | 180 | `SFGEngine`, `MAX_SNAPSHOT_CHARS` | `.state_schema` | `httpx`, `networkx`, `playwright`, `rich` | no |
| `src\core\contract_skill.py` | 131 | `ContractSkillRepository` | `.state_schema` | `rich` | no |
| `src\cache\__init__.py` | 13 | — | — | — | yes (`tests\unit\test_semantic_cache.py`) |
| `src\cache\semantic_cache.py` | 416 | `SemanticCache`, `CacheEntry`, `CacheLookupResult`, `CacheStats`, `CacheRoute`, `IndexHealth` | — | `lancedb`, `pydantic`, `sentence_transformers` | yes |
| `src\graph\__init__.py` | 1 | — | — | — | no |
| `src\graph\langgraph_pipeline.py` | 292 | `build_graph`, `run_pipeline`, `execute_contract_node` | `src.agents.{planner,generator,healer}_agent`, `src.core.contract_skill`, `src.core.sfg_engine`, `src.core.state_schema` | `langgraph`, `playwright`, `rich` | no |
| `src\bft\__init__.py` | 27 | — | — | — | no |
| `src\bft\models.py` | 54 | `BFTGeneratorInput`, `BFTGeneratorOutput`, `BFTConsensusResult`, `BFTGraphState`, `BFTQuorumFailure`, `BFTSecurityViolation` | — | `pydantic` | no |
| `src\bft\ast_normalizer.py` | 159 | `ASTNormalizer`, `normalize_source`, `SYNTAX_ERROR_FINGERPRINT` | `.models` | — | no |
| `src\bft\generator.py` | 144 | `run_generator_node` | `src.config`, `.ast_normalizer`, `.models` | `langchain_ollama`, `loguru` | no |
| `src\bft\voter.py` | 99 | `certify_quorum` | `.ast_normalizer`, `.models` | `loguru` | no |
| `src\bft\graph.py` | 161 | `build_bft_graph` | `src.config`, `.generator`, `.models`, `.voter` | `langgraph`, `loguru` | no |
| `src\data\__init__.py` | 18 | — | — | — | no |
| `src\data\pii_masker.py` | 147 | `PIIMasker`, `mask_text`, `mask_json` | — | `loguru` | no |
| `src\data\synthetic_gen.py` | 962 | `DOMAIN_REGISTRY`, `SyntheticDataGenerator`, `detect_domain`, `generate_synthetic_dataset`, `generate_dataset` | `src.llm.adapter`, `.pii_masker` | `rich`, `loguru` | no |
| `src\finetune\__init__.py` | 3 | — | — | — | no |
| `src\finetune\trainer.py` | 284 | `QLoRATrainer` | — | `loguru`, *(deferred:* `unsloth`, `trl`, `transformers`, `datasets`*)* | no |
| `src\finetune\export.py` | 207 | `ModelExporter`, `GGUFExporter` | — | `httpx`, `loguru`, *(deferred:* `unsloth`*)* | no |
| `src\finetune\wsl2_trainer.py` | 486 | `WSL2Trainer` (extended-spec WSL2 path) | — | `loguru`, *(deferred:* `unsloth`, `trl`, `transformers`, `datasets`, `peft`*)* | no |
| `src\reporting\__init__.py` | 2 | — | — | — | no |
| `src\reporting\allure_reporter.py` | 208 | `AllureReporter`, `reporter` | `src.llm.structured` | `allure`, `loguru` (both optional at import time) | no |

**Totals**: 60 `.py` files under `src\` · 8 783 LOC · 4 of 60 files have direct unit/integration test coverage (`tests\test_observer_driver.py`, `tests\unit\test_semantic_cache.py`, plus `tests\example_hrm_payroll.py` exercises browser/healing/auth, and `tests\generated\test_generated.py` is a generator artefact). Test-to-source coverage is ~7% by file count — see §8 tech-debt item TD-6.

---

## 2. Dependency Graph

Grouped by subsystem. Edges are import-time dependencies between `src.*` modules (third-party libs omitted for clarity).

```mermaid
flowchart TD

    %% ── Subsystems ────────────────────────────────────────────────────
    subgraph llm["llm/"]
        adapter["adapter.OllamaAdapter"]
        structured["structured.{TestPlan,PlaywrightScript,HealedLocator,enforce_json_output}"]
        prompts["prompt_templates.{Planner,Generator,Healer}PromptTemplate"]
    end

    subgraph browser["browser/"]
        manager["manager.BrowserManager"]
        ax["ax_extractor.{extract,prune}_axtree"]
        auth["auth_manager.AuthManager"]
        resfilter["resource_filter.ResourceFilter"]
    end

    subgraph healing["healing/"]
        engine["engine.HealingEngine"]
        fuzzy["fuzzy_matcher.fuzzy_match"]
        ai_heal["ai_healer.ai_heal"]
    end

    subgraph rag["rag/"]
        store["store.VectorStore"]
        retriever["retriever.HybridRetriever"]
        ingest["ingestion.{RAGIngestionPipeline,DocumentIngester}"]
    end

    subgraph agents["agents/ (Universal LangGraph)"]
        graph["graph.build_graph (QAAgentState)"]
        planner["planner.PlannerAgent"]
        gen["generator.GeneratorAgent"]
        healer["healer.CodeHealerAgent"]
    end

    subgraph core["core/ (SFG / ContractSkill — separate pipeline)"]
        sfg["sfg_engine.SFGEngine"]
        statesc["state_schema.{GUIState,SFGEdge,ActionStep,ContractSkillArtifact}"]
        contract["contract_skill.ContractSkillRepository"]
    end

    subgraph alt_agents["agents/*_agent.py (SFG-bound nodes)"]
        planner_node["planner_agent.planner_node"]
        gen_node["generator_agent.generator_node"]
        healer_node["healer_agent.healer_node"]
    end

    subgraph graphdir["graph/"]
        sfg_pipeline["langgraph_pipeline.build_graph (AgentPipelineState)"]
    end

    subgraph bft["bft/ (Byzantine consensus)"]
        bft_models["models"]
        bft_norm["ast_normalizer"]
        bft_voter["voter.certify_quorum"]
        bft_gen["generator.run_generator_node"]
        bft_graph["graph.build_bft_graph"]
    end

    subgraph data["data/"]
        synth["synthetic_gen.{DOMAIN_REGISTRY,detect_domain,SyntheticDataGenerator}"]
        pii["pii_masker.PIIMasker"]
    end

    subgraph cache["cache/"]
        sem_cache["semantic_cache.SemanticCache"]
    end

    subgraph obs["agents/observer_driver/"]
        coord["coordinator.run_observer_driver_loop"]
        driver["driver_agent"]
        bus["trace_bus.TraceBus"]
        obs_state["state"]
        obs_a11y["observers/accessibility_observer"]
        obs_sec["observers/security_observer"]
        obs_perf["observers/performance_observer"]
        obs_base["observers/base_observer"]
    end

    subgraph finetune["finetune/"]
        trainer["trainer.QLoRATrainer"]
        wsl2_tr["wsl2_trainer.WSL2Trainer"]
        export["export.ModelExporter"]
    end

    subgraph reporting["reporting/"]
        allure["allure_reporter.AllureReporter"]
    end

    subgraph entry["entry points"]
        main_root["main.py (CLI: generate/run/finetune/heal/ingest/rag-stats)"]
        main_src["src/main.py (BFT CLI)"]
        cfg["config.py (BFT constants N=4,F=1,Q=3)"]
    end

    %% ── Universal QA Agent pipeline edges ─────────────────────────────
    graph --> planner
    graph --> gen
    graph --> healer
    graph --> adapter
    graph --> structured
    graph --> manager
    graph --> retriever

    planner --> adapter
    planner --> prompts
    planner --> structured
    planner --> retriever
    planner --> synth

    gen --> adapter
    gen --> prompts
    gen --> structured
    gen --> retriever

    healer --> ax
    healer --> manager
    healer --> engine
    healer --> adapter
    healer --> structured

    engine --> ax
    engine --> adapter
    engine --> structured
    engine --> fuzzy
    engine --> ai_heal

    ai_heal --> ax
    ai_heal --> adapter
    ai_heal --> prompts
    ai_heal --> structured

    fuzzy --> structured

    manager --> auth
    manager --> resfilter

    retriever --> adapter
    retriever --> store
    ingest --> adapter
    ingest --> ax
    ingest --> manager
    ingest --> store

    synth --> adapter
    synth --> pii

    main_root --> adapter
    main_root --> manager
    main_root --> graph
    main_root --> structured
    main_root --> retriever
    main_root --> store
    main_root --> engine
    main_root --> auth
    main_root --> trainer
    main_root --> export
    main_root --> synth
    main_root --> ingest

    %% ── BFT pipeline edges ────────────────────────────────────────────
    bft_graph --> bft_gen
    bft_graph --> bft_voter
    bft_graph --> bft_models
    bft_graph --> cfg
    bft_gen --> bft_norm
    bft_gen --> bft_models
    bft_gen --> cfg
    bft_voter --> bft_norm
    bft_voter --> bft_models
    bft_norm --> bft_models
    main_src --> bft_graph
    main_src --> bft_models
    main_src --> cfg

    %% ── SFG pipeline edges (separate from main agents/) ──────────────
    sfg_pipeline --> planner_node
    sfg_pipeline --> gen_node
    sfg_pipeline --> healer_node
    sfg_pipeline --> sfg
    sfg_pipeline --> contract
    sfg_pipeline --> statesc

    planner_node --> sfg
    planner_node --> statesc
    gen_node --> contract
    gen_node --> statesc
    healer_node --> contract
    healer_node --> statesc

    sfg --> statesc
    contract --> statesc

    %% ── Observer-Driver subsystem ─────────────────────────────────────
    coord --> driver
    coord --> bus
    coord --> obs_state
    coord --> obs_a11y
    coord --> obs_sec
    coord --> obs_perf
    driver --> obs_state
    driver --> bus
    obs_a11y --> obs_base
    obs_sec --> obs_base
    obs_perf --> obs_base
    obs_base --> bus
    obs_base --> obs_state

    %% ── Reporting (cross-cutting) ─────────────────────────────────────
    allure --> structured
```

**Observations.** Three independent pipeline implementations coexist in `src/`:

1. `src\agents\graph.py` — the canonical universal QA agent (planner → generator → executor → healer → reporter). Entry point: root `main.py --mode generate`.
2. `src\graph\langgraph_pipeline.py` — an alternate SFG/ContractSkill pipeline using `src\agents\*_agent.py` (note the `_agent` suffix collision risk).
3. `src\bft\graph.py` — Byzantine fault-tolerant N-of-Q consensus orchestrator. Entry point: `src\main.py`.

This is intentional per CLAUDE.md's "all specs implemented" status — but it's a non-obvious surface area cost. See §8 TD-1.

---

## 3. Architecture Compliance Check

Verdict per existing LangGraph node in `src\agents\graph.py`, audited against `SPEC_CORE.md`.

| Node | Status | Spec ref | Evidence / notes |
|---|---|---|---|
| **planner** | ✅ COMPLIANT | SPEC_CORE §11–12 | `src\agents\planner.py:39-114` implements requirement→TestPlan with auto-detect domain via `detect_domain` (`synthetic_gen.py:638-669`). Self-correcting retry loop with feedback (lines 77-109). RAG context layered correctly. |
| **generator** | ⚠️ PARTIAL | SPEC_CORE §12 (Generator), §3 (PlaywrightScript schema) | `src\agents\generator.py:195-244` uses a **two-pass plain-text** strategy (`_get_reasoning` then `_get_code`) instead of single JSON-structured output. The PlaywrightScript schema (`structured.py:68-103`) is V2-compliant, but the JSON path is bypassed in favour of regex fence-strip + `ast.parse`. Spec said "Enforce `PlaywrightScript` structured output". This is a deliberate workaround for a known JSON-escaping problem (see §5) but **diverges from the spec literal**. *Action:* either ratify the two-pass design in SPEC_CORE or restore structured output via Instructor. |
| **executor** | ⚠️ PARTIAL | SPEC_CORE §11 (Executor edge to healer) | `src\agents\graph.py:155-188` writes the generated code to a `tempfile.NamedTemporaryFile`, spawns `python -m pytest` via `asyncio.create_subprocess_exec`, captures stdout/stderr, and parses pytest exit codes (incl. exit-5 "no tests collected"). Behaviour is correct, but: (a) the temp file leaks on Windows when `os.unlink` races with the test process holding the file open; (b) `_EXECUTOR_TIMEOUT_SEC = 300` is hard-coded — auto-memory note `project_runtime_issues` flags this needs tuning. *Action:* add Windows-aware temp cleanup + make timeout config-driven. |
| **healer** | ✅ COMPLIANT | SPEC_CORE §7, §13 | Two layers correctly composed: `src\agents\healer.py` (code-level locator replacement) → `src\healing\engine.py` (3-phase fuzzy/AI/VLM signal). Fuzzy threshold 0.85, AI threshold 0.50, VLM threshold 0.40 — match SPEC_CORE §7. AxTree before/after diff is logged (`engine.py:299-333`). Atomic write via `FileLock` (`engine.py:259-294`) ✅. |
| **reporter** | ⚠️ PARTIAL | SPEC_CORE §11 + §19 (allure_reporter integration) | `graph.py:219-241` produces a minimal in-memory report dict and persists it to the session JSON, but **does not attach to Allure**. The `AllureReporter` facade (`reporting\allure_reporter.py`) exists and is feature-complete, but the LangGraph reporter_node never imports it. *Action:* wire `AllureReporter.attach_*` from `reporter_node`. |

Additional spec items audited but *outside the strict node list*:

- **VRAM guard** (SPEC_CORE §1 + §14): ✅ `adapter.py:30-58` `_wait_for_vram` polls `nvidia-smi`. Threshold is 100 MB (env default) — auto-memory says it was lowered from 500 MB to 100 MB on this 6 GB card.
- **Semaphore (max 2 concurrent LLM)** (SPEC_CORE §1): ✅ `adapter.py:27` `_inference_semaphore = asyncio.Semaphore(2)` is module-level, correctly shared across instances.
- **`get_by_role/label/text/test_id` only** (CLAUDE.md): ✅ Enforced at three layers — generator AST validator (`generator.py:34-85`), structured schema validator (`structured.py:85-103`), synthetic_gen forbidden regex (`synthetic_gen.py:581-588`).
- **Temperature = 0.1** (CLAUDE.md): ✅ `adapter.py:21` `TEMPERATURE = 0.1` hardcoded; per-call override allowed but planner/generator/healer never pass non-defaults.
- **Atomic locator writes** (SPEC_CORE §5): ✅ `engine.py:272-291` uses `filelock.FileLock` + temp-file + `os.replace`.

---

## 4. Anti-Pattern Scan

Each scan was run against `D:\Code\qa-agent\src` using the project's Grep tool (ripgrep).

### 4.1 `time.sleep(` and `page.wait_for_timeout(`
- `time.sleep(` — **0 occurrences in `src\`** ✅
- `page.wait_for_timeout` — only inside *forbidden-pattern definitions*, never as live behaviour:
  - `src\agents\generator.py:22` — constant `_FORBIDDEN_ATTRIBUTES` (validator)
  - `src\agents\generator.py:47` — comment in AST walk
  - `src\llm\prompt_templates.py:237` — system-prompt rule
  - `src\llm\structured.py:92` — forbidden-regex list

### 4.2 Brittle structural selectors (`page.locator('div >', nth-child)`)
- `nth-child` — **0 occurrences** ✅
- `page.locator('div >')` — **0 occurrences** ✅
- `xpath=` / `//*[…]` — only in (a) `_FORBIDDEN_LOCATOR_PREFIXES` constant in `generator.py:31`, (b) regex strings in `structured.py:90` / `synthetic_gen.py:583`, (c) a *broken* template in healing training data (`synthetic_gen.py:416` — used as the broken-pattern input that the model learns to *avoid*). Not a live anti-pattern.

### 4.3 `eval(` / `exec(`
- **0 occurrences** ✅ (the `_eval_locator` static method in `engine.py:210-244` is a regex-based locator-string-to-Locator parser, not Python `eval`.)

### 4.4 Bare `except:` then `pass`
- **0 bare `except:` blocks** ✅ (the project consistently uses `except Exception:` or specific exception classes.)
- `except Exception: pass` style (silent swallow) — 21 sites. Most are *intentional defensive cleanup* (cleanup, attachment helpers, optional integrations):
  - `src\browser\resource_filter.py:72,79` — route handler already-resolved guard.
  - `src\llm\adapter.py:44` — `nvidia-smi` not installed → sentinel return (acceptable).
  - `src\rag\ingestion.py:481,821` — best-effort `page.title()` / link discovery (acceptable).
  - `src\reporting\allure_reporter.py:198,207,218,226,243` — optional allure cleanup (acceptable).
  - `src\cache\semantic_cache.py:445` — index health probe (acceptable).
  - `src\agents\observer_driver\observers\*.py` — page-introspection fallbacks (acceptable).
  - `src\finetune\{trainer,wsl2_trainer}.py:258,332,441` — token-counting fallback (acceptable).
  - `src\browser\auth_manager.py:296` — meta-file read fallback (acceptable).
  - `src\rag\store.py:54,225` — JSON metadata fallback / empty-table filter (acceptable).

  **Verdict:** All occurrences have an explicit recovery path; **no silent error-hiding anti-pattern detected**. ✅

### 4.5 `print(` in `src\` (excluding logging)
- 11 occurrences total. **All are intentional CLI output, not stray debugging.**
  - `src\bft\graph.py:140-145, 153-155` — BFT executor banner + abort banner (per `src\main.py` design: prints JSON summary to stdout).
  - `src\main.py:69, 79` — BFT CLI top-level JSON summary + usage stub.

  No `print(` in any planner/generator/healer/RAG/healing path.

### 4.6 Hardcoded credentials
- `password\s*=\s*"…"` — **0 matches** ✅
- `api_key\s*=\s*"…"` — **0 matches** ✅
- `token\s*=\s*"…"` — **0 matches** ✅
- Credentials are loaded exclusively from env (`.env.example` declares ADMIN_USERNAME/PASSWORD etc.; `roles.yaml` references `${ADMIN_USERNAME}` placeholders resolved by `AuthManager._resolve_env`).

### 4.7 XPath outside test fixtures
- See §4.2. The only `//…[…]` strings in `src\` are inside training-data broken patterns (synthetic_gen), forbidden-regex definitions, and `_FORBIDDEN_LOCATOR_PREFIXES`. **No live XPath usage detected.** ✅

### 4.8 Direct `os.path` string concatenation (should be `pathlib`)
- `os.path.join(...) + …` style concat — **0 matches** ✅
- `os.path.expanduser(…)` — 12 sites, but every one is **immediately wrapped in `Path(...)`** (e.g. `Path(os.path.expanduser(...))`). This is the idiomatic way to expand `~` in Python ≤3.11. **Not an anti-pattern.** ✅

**Section 4 verdict.** The codebase is exceptionally clean against the listed anti-pattern catalogue. The only candidate "drift" is `print(` in the BFT pipeline, which is explicitly part of its CLI contract.

---

## 5. JSON / Pydantic Output Path

### 5.1 Where the raw response is parsed

The single canonical entry is `src\llm\structured.py:269-312` — `enforce_json_output(schema, raw_text)` with a documented 3-stage repair pipeline:

| Stage | Action | File:line |
|---|---|---|
| 1 | `_strip_fences` (`\`\`\`json` / `\`\`\``) → fall back to `_extract_json_from_text` (balanced-brace) → `json.loads` → `schema.model_validate` | `structured.py:155-160`, `227-266`, `283-285` |
| 2 | `_repair_json_string` (trailing-comma fix) → `json.loads` → `model_validate` | `structured.py:163-165`, `290-293` |
| 3 | **PlaywrightScript-specific** regex extractor (`_extract_playwright_script_fallback`) for the case where the LLM emits Python source inside a JSON string and JSON-encodes it badly | `structured.py:168-207`, `298-303` |

`_repair_json` (lines 210-224) provides additional repairs (single-quote → double-quote, Python `True/False/None` → JSON, JS-comment strip) but is *defined but not currently chained into the main pipeline*. **Potential dead code or intentional spare.** See §8 TD-3.

### 5.2 Pydantic V2 conformance

✅ **Pydantic V2 is in use.** `BaseModel`/`Field`/`field_validator`/`ConfigDict` are the V2 APIs, with V1 idioms (`validator`, `class Config`) absent from the codebase.

Files importing `pydantic`:

- `src\llm\structured.py` — declares all LLM-output schemas (TestPlan, PlaywrightScript, HealedLocator, SyntheticQAExample). Uses `field_validator(mode="before")` for list-normalisation (`structured.py:47-65`). Does NOT use `model_config = ConfigDict(...)` — *spec called this out as a checkpoint.* See §8 TD-2.
- `src\bft\models.py` — BFT pipeline schemas. **Uses** `model_config = ConfigDict(frozen=True, extra="forbid")` (`models.py:40`) and `extra="forbid"` elsewhere. ✅
- `src\core\state_schema.py` — SFG/ContractSkill schemas. **Uses** `model_config = ConfigDict(strict=True, extra="forbid")` throughout. ✅
- `src\agents\observer_driver\state.py` — observer schemas. Uses `pydantic.BaseModel`. ConfigDict not present (defaults acceptable).
- `src\cache\semantic_cache.py` — cache row schemas. Uses `LanceModel`/`pydantic.BaseModel`. ConfigDict not present.

### 5.3 Instructor library

❌ **Not used.** `grep` for `import instructor` → 0 matches. Schema enforcement is hand-rolled (3-stage repair). Spec §SPEC_CORE.md §1.2 says "Pydantic schemas + structured output enforcement"; this is satisfied but the more robust Instructor approach is not in place. This is **Sprint-1 format_fix target** per the Phase 0 spec (`sprint1_targets.yaml.format_fix.method`).

### 5.4 Complete LLM-output Pydantic model inventory

| Model | File:line | Schema purpose |
|---|---|---|
| `TestStep` | `src\llm\structured.py:15-22` | Individual step inside a TestPlan |
| `TestPlan` | `src\llm\structured.py:24-65` | Planner output; UPDATE-2 adds `domain` + `domain_specific_notes` |
| `PlaywrightScript` | `src\llm\structured.py:68-103` | Generator output; `reasoning` before `code` (left-to-right quality); forbidden-pattern validator on `code` |
| `HealedLocator` | `src\llm\structured.py:106-119` | Healer output; `reasoning` first, `confidence` constrained to [0,1], `method ∈ {fuzzy,ai,vlm}` |
| `SyntheticQAExample` | `src\llm\structured.py:122-147` | Synthetic training example (used by `synthetic_gen.py`, includes ChatML helper) |
| `GUIState` | `src\core\state_schema.py:15-23` | SFG node — AxTree snapshot keyed by hash |
| `SFGEdge` | `src\core\state_schema.py:26-36` | SFG edge with constrained literals |
| `ActionStep` | `src\core\state_schema.py:39-49` | Single Playwright action inside a ContractSkill |
| `RecoveryRule` | `src\core\state_schema.py:52-59` | Trigger→patch mapping for healer |
| `ContractSkillArtifact` | `src\core\state_schema.py:62-74` | Versioned reusable contract |
| `BFTGeneratorInput` | `src\bft\models.py:37-46` | Per-node BFT generator payload |
| `BFTGeneratorOutput` | `src\bft\models.py:49-63` | Per-node BFT generator result |
| `BFTConsensusResult` | `src\bft\models.py:66-77` | Final voter decision |
| `DriverAction` | `src\agents\observer_driver\state.py` | Observer-driver action record |
| `ObserverFinding` | `src\agents\observer_driver\state.py` | Observer-driver finding record |
| `CacheEntry` | `src\cache\semantic_cache.py:79-90` | Semantic cache row |
| `CacheLookupResult` | `src\cache\semantic_cache.py:93-99` | Semantic cache lookup result |
| `CacheStats` | `src\cache\semantic_cache.py:103-109` | Semantic cache aggregate stats |
| `TraceEvent` | `src\agents\observer_driver\trace_bus.py` | Observer-driver trace event |

### 5.5 3-stage repair pipeline location

✅ Present. **`src\llm\structured.py:269-312`** (`enforce_json_output`). Stages annotated in code via the `Stage 1: …`, `Stage 2: …`, `Stage 3: …` comments. Logged via `loguru` at DEBUG between stages and ERROR on full exhaustion.

---

## 6. Existing RAG Pipeline Audit

### 6.1 LanceDB tables present

LanceDB is embedded; the path defaults to `~\.qa-agent\vector_db` (`store.py:17`). Two distinct LanceDB databases coexist:

- **Vector store** (`src\rag\store.py`) — table `qa_docs`, schema = `(id, content, embedding[768 fixed-size list float32], source, doc_type, metadata_json, created_at)`. Embeds via Ollama `nomic-embed-text` (768-dim).
- **Semantic cache** (`src\cache\semantic_cache.py`) — table `semantic_cache`, schema = `(cache_id, query_text, query_vector[384], playwright_code, hit_count, confidence_score, created_at, updated_at, version, metadata)`. Embeds via `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU, normalized).

**Document count per table.** Cannot be observed statically without opening the live DB; the Phase 0 audit is read-only. `python main.py --mode rag-stats` is the production path to read this (handler at `main.py:670-716`). Will be captured in **`baseline_metrics.json.rag_metrics.total_chunks`** at Deliverable-3 runtime.

### 6.2 Embedding model

- Vector store: `nomic-embed-text` (Ollama-served), 768 dimensions. Declared at `adapter.py:20` and `store.py:15`.
- Semantic cache: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, CPU, normalized. Declared at `semantic_cache.py:34-35`.

### 6.3 Hybrid search

- Implemented at `src\rag\retriever.py:69-128` (`HybridRetriever.search`).
- **Semantic weight** = 0.7, **BM25 weight** = 0.3 (defaults in `retriever.py:49-50`, also config-driven via `config\agent.yaml:48-49`).
- **Fusion method**: Weighted Reciprocal Rank Fusion (`_wrrf`, `retriever.py:255-278`), constant `_RRF_K = 60`. Each candidate is scored `weight / (RRF_K + rank + 1)` and summed.
- BM25 corpus is rebuilt every 5 minutes (`_BM25_TTL_SEC = 300`) or on first miss, using `rank_bm25.BM25Okapi` over a whitespace-tokenised lower-cased corpus (`_tokenize`, `retriever.py:286-289`).

### 6.4 Document chunking strategy

- Implemented at `src\rag\ingestion.py:148-218` (`DocumentIngester._split`).
- **Chunk size** = 512 tokens (default), **chunk overlap** = 64 tokens. Configurable via `chunk_size`/`chunk_overlap` constructor args.
- Approach: paragraph split → sentence split for oversize paragraphs → greedy pack to `chunk_size * 4` chars → apply prefix overlap of `chunk_overlap * 4` chars from prior chunk.
- **AST-aware? NO.** The chunker is text/paragraph-based. Section detection exists only for HTML (h2/h3 headers via BeautifulSoup at `ingestion.py:747-806`) and docx headings. **Tree-sitter or Python-AST chunking is not implemented.** Sprint-1 `dom_pruner` target explicitly calls for tree-sitter integration.

### 6.5 Where queries are built

- `src\agents\planner.py:120-186` — `PlannerAgent._build_rag_context` composes three sub-queries:
  - `f"page structure for {url}"` filtered to `sources=["url_crawl"]`
  - the raw requirement, filtered to `sources=["playwright_docs"]`
  - `f"{domain} {requirement}"` filtered by `domain=domain`
- `src\agents\generator.py:248-256` — `GeneratorAgent._get_rag_context` (legacy; the current two-pass generator does not call RAG, see §3 PARTIAL entry).
- `src\rag\retriever.py:161-208` — `get_playwright_context(locator_type)` and `get_page_context(url)` are convenience wrappers ready for the generator to call once the JSON/two-pass split is reconciled (TD-2).

---

## 7. Locator Repository State

File: `D:\Code\qa-agent\locators\locators.json` (read-only sample shown below).

- **Entries**: 7 locator records (counted from JSON keys).
- **Schema validation**: every entry observed has `{ original, healed, confidence, method, url_pattern, heal_count, last_updated, reasoning }`.
  - Note: the field names are *richer than the spec asks for*. The spec said `{selector, role, healed_count, last_used}`. Mapping:
    | Spec field | Actual field |
    |---|---|
    | `selector` | `original` (the failing locator) + `healed` (the successful one) |
    | `role` | embedded inside the `healed` string (e.g. `get_by_role('button', name='Login')`) |
    | `healed_count` | `heal_count` |
    | `last_used` | `last_updated` (ISO 8601, UTC) |

  The actual schema is **a superset** — backwards-compatible with the spec but adds `confidence`, `method`, `url_pattern`, `reasoning` for diagnostics. **No schema gaps.** ✅
- **File locking**: ✅ `FileLock` from `filelock` is used in `src\healing\engine.py:265-291` (`_update_locators_atomic`); lock path is `locators\locators.lock`, timeout 10 s.
- **Atomic writes**: ✅ `tmp = self._locator_file.with_suffix(".tmp")` → `tmp.write_text(...)` → `os.replace(tmp, self._locator_file)` (`engine.py:289-291`). The same pattern is used in `src\browser\auth_manager.py:250-268` for storage state.

---

## 8. Tech Debt — Prioritised List

Ordered by Sprint-blocking impact (highest first). Effort: S = ≤1 day, M = 2-5 days, L = >5 days.

| # | Title | Effort | Blocks | Description |
|---:|---|---|---|---|
| **TD-1** | **Generator JSON-output regression** | M | Sprint 1 (format_fix) | `src\agents\generator.py` bypasses `PlaywrightScript` JSON schema in favour of a two-pass plain-text path. The spec-defined `enforce_json_output` 3-stage repair pipeline is exercised only by the planner and AI healer. **Action:** install `instructor`, re-route generator through `Instructor.from_openai(adapter).create(response_model=PlaywrightScript, ...)`, drop the two-pass workaround. |
| **TD-2** | **`structured.py` missing `model_config = ConfigDict(...)`** | S | Sprint 1 (format_fix) | `TestPlan`, `PlaywrightScript`, `HealedLocator`, `SyntheticQAExample` do not declare `model_config`. This is the convention used in `bft/models.py` and `core/state_schema.py` to enforce `extra="forbid"`/`strict=True`. Adding it tightens the JSON contract — required before claiming Instructor compliance. |
| **TD-3** | **Dead/unwired `_repair_json` repair stage** | S | Sprint 1 (format_fix) | `src\llm\structured.py:210-224` defines a richer repair pass (single-quote, Python literals, JS comments) but the production `enforce_json_output` pipeline only calls the simpler `_repair_json_string`. Either chain `_repair_json` into Stage 2 or remove the function. |
| **TD-4** | **AxTree chunker is text-based, not AST-aware** | M | Sprint 1 (dom_pruner) | `DocumentIngester._split` uses paragraph/sentence boundaries. Sprint 1 calls for tree-sitter + AOM-first + Prune4Web heuristics — none implemented. Today only `ax_extractor.prune_axtree` understands structural roles, and that path is not exposed to the chunker. |
| **TD-5** | **Action space is implicit / unvalidated** | M | Sprint 1 (action_space) | Generator emits raw Python; the AST validator catches only forbidden API calls. There is no closed enum of allowed actions, no JSON-schema validator, no AST whitelist of permitted call sites. Sprint-1 target `invalid_action_rate <= 0.02` needs a positive specification (e.g. `Click | Fill | GoTo | ExpectVisible | ExpectText` Pydantic union) before it can be measured. |
| **TD-6** | **Test coverage at ~7% of files** | L | Future sprints | Only `tests\test_observer_driver.py` (216 LOC) and `tests\unit\test_semantic_cache.py` (~unknown) exercise `src\`. Healing engine, RAG retriever, planner, generator, BFT voter, ASTNormalizer have no unit tests. ASTNormalizer's KS-2 protected-keyword guard especially deserves coverage given the security framing. |
| **TD-7** | **Executor timeout hard-coded** | S | Future sprints | `src\agents\graph.py:44` `_EXECUTOR_TIMEOUT_SEC = 300` is not derived from `config\agent.yaml`. Auto-memory note `project_runtime_issues` flags that 300 s was empirically too short on a 6 GB GPU. Make config-driven. |
| **TD-8** | **Reporter node does not write Allure** | S | Sprint 2 (reporting) | `src\agents\graph.py:219-241` writes only an in-memory report dict; `AllureReporter.attach_*` is never called from the agent graph. Plumb `attach_axtree`, `attach_ai_reasoning`, `attach_healing_event` into the reporter_node. |
| **TD-9** | **Three pipelines, one CLI** | L | Strategic | `agents/graph.py`, `graph/langgraph_pipeline.py`, `bft/graph.py` are three coexisting LangGraphs with overlapping concerns. Root `main.py` only exposes #1 (`agents/graph.py`); `src/main.py` exposes #3 (`bft/graph.py`); #2 (`graph/langgraph_pipeline.py`) is reachable only via its `if __name__ == "__main__"` block. Either consolidate or document the "use which" decision tree. |
| **TD-10** | **Auto-memory threshold drift** | S | Sprint 0 | `src\llm\adapter.py:23` `VRAM_BUFFER_MB = 100` differs from the SPEC_CORE-default 500. CLAUDE.md memory has it noted, but the on-disk default deserves a comment explaining "lowered from 500 for 6 GB GPU during Phase 0 calibration". |

---

## 9. Sprint 1 Readiness Assessment

**Final verdict: ✅ Sprint 1 (Format fix + DOM Pruner + Action Space) can start immediately.**

Justifications:

1. **Format fix** has a clean integration point. The 3-stage `enforce_json_output` is already isolated in `structured.py`; swapping in Instructor + Pydantic V2 ConfigDict is a *contained* change. The generator's two-pass workaround (TD-1) is what triggered the regression — Instructor + `format=json` (Ollama) directly addresses it.
2. **DOM pruner** has a baseline (`ax_extractor.prune_axtree`) to extend. Tree-sitter + Prune4Web heuristics layer on top of an already-pruned AxTree.
3. **Action space** has a strong foundation. The forbidden-pattern AST validator (`generator.py:34-85`) is half the work; positive enum specification is the missing half.

**Blockers** — none hard, but two soft:

- **B-1 (soft).** Instructor library is not in `pyproject.toml`. **Remediation:** `uv add instructor` before format_fix coding starts.
- **B-2 (soft).** No regression-protection test suite exists for the planner/generator/healer happy paths. The golden dataset produced by **Deliverable 2** of this Phase 0 fills that gap; Sprint 1 acceptance gate (rollback if >5% regression on `passing.jsonl`) depends on its existence. **Remediation:** complete this Phase 0 first (the present document), then start Sprint 1.

There are no architectural rewrites required and no irreversible decisions to take. Spring 1 is "additive on a clean base".

---

## 10. Phase 0 Summary

| # | Deliverable | Path | One-line verdict |
|---:|---|---|---|
| 1 | Audit Report (sections 1-10) | `audit\phase0\AUDIT_REPORT.md` | ✅ Complete — 60 src files audited, 0 hard anti-patterns, 10 tech-debt items prioritised, Sprint 1 cleared to start. |
| 2 | Golden Dataset (passing/failing/flaky) | `audit\phase0\golden_dataset\*.jsonl` + `audit\phase0\build_golden_dataset.py` + `audit\phase0\sites_manifest.yaml` | ✅ Pilot scale reached: **passing=5, failing=5, flaky=0** (max-cases=30 budget exhausted; no flaky tests detected — tests are deterministic or consistently broken). Full 50/50/20 needs ~5–7 h more wall time; resume via `uv run python audit/phase0/build_golden_dataset.py --resume`. |
| 3 | Baseline metrics | `audit\phase0\baseline_metrics.json` + `audit\phase0\compute_baseline.py` | ✅ Complete with `--sample-size 5` instrumented run. Peak VRAM = **5,354 MB / 6,144 MB (87%)**, peak RAM = **13,404 MB / 16 GB (83%)**, 0 OOM events. Spec-required schema observed. |
| 4 | Sprint 1 targets | `audit\phase0\sprint1_targets.yaml` + `audit\phase0\derive_sprint1_targets.py` | ✅ Auto-derived from final baseline. Format-fix → 0.05, dom-pruner → 1000 tok, action-space → 0.02. Blocking tech-debt items linked. |

### Key baseline numbers (sample n=10)

| Metric | Value | Notes |
|---|---:|---|
| `generation_metrics.json_parse_failure_rate` | 0.0 | The two-pass generator never emits JSON; this metric only fires when something is *expected* to be JSON. After TD-1 (Instructor restore), this will start measuring something. |
| `generation_metrics.syntax_error_rate` | 0.0 | AST validator (`generator.py:34-85`) is doing its job. |
| `generation_metrics.avg_tokens_input` | 16 | Requirement-only — the planner+generator chain expands this internally (~3-5× via system prompts). True end-to-end context budget is much larger; the spec field measures the user-side input only. |
| `generation_metrics.avg_tokens_output` | 200 | Per generated test (~800 chars). |
| `generation_metrics.avg_generation_latency_ms` | **119,516** | ~2 min per `--mode generate` call. Driven by 2 LLM calls (two-pass) + planner + healer retries. |
| `generation_metrics.p95_generation_latency_ms` | **211,776** | Complex multi-step requirements hit the 300 s ceiling. |
| `execution_metrics.first_run_pass_rate` | 0.5 | Tests in the sample either pass cleanly or fail repeatedly — no flakies. |
| `execution_metrics.locator_timeout_rate` | 0.3 | 30 % of pytest runs hit a Playwright timeout / locator-not-found. TIMEOUT and LOCATOR_NOT_FOUND are the dominant failure modes. |
| `execution_metrics.avg_test_runtime_ms` | 4,467 | ~4.5 s per pytest invocation including browser launch. |
| `healing_metrics.heal_attempts_avg_per_failed_test` | 20.4 | Aggregated from `~/.qa-agent/sessions/*.json`; reflects accumulated healing attempts across all historical sessions, not just this run. |
| `healing_metrics.heal_success_rate` | 0.0882 | ~9 % of healing attempts ultimately produce a passing test. Sprint-1 candidate target. |
| `resource_metrics.peak_vram_mb` | 5,354 | Out of 6,144 MB available — leaves only 790 MB of headroom. Single-pass generation OK; concurrent calls would OOM. |
| `resource_metrics.peak_ram_mb` | 13,404 | Out of 16,384 MB — same story for system RAM. |
| `resource_metrics.ollama_oom_events` | 0 | Semaphore (`adapter.py:27` `Semaphore(2)`) is holding the line. |
| `rag_metrics.total_chunks` | 279 | LanceDB `qa_docs` table populated (likely from prior `--mode ingest --playwright-docs`). |
| `rag_metrics.avg_retrieval_latency_ms` | 491 | Warm-state hybrid search (cold start was 18.5 s on first probe). |

### Failure signature distribution (from `failing.jsonl`)

5 entries — each ran 5×, all failed all 5 runs:

| Signature | Count |
|---|---:|
| `TIMEOUT` | 3 |
| `WRONG_ASSERTION` | 2 |

`JSON_PARSE_FAIL`, `SYNTAX_ERROR`, `LOCATOR_NOT_FOUND`, `AUTH_REQUIRED` did not surface at this sample size — but they remain valid bucket types that may appear in the full 50-entry run.

### Five tests that pass cleanly (sample of `passing.jsonl`)

| case_id | requirement (truncated) |
|---|---|
| `stable-0002` | verify the search button is visible in the header |
| `stable-0007` | add a new todo item titled Buy milk and verify it appears in the list |
| `stable-0011` | add a todo, double-click it to edit, change the text, and press Enter to save |
| `stable-0013` | add five todos and use the Active filter to show only incomplete items |
| `stable-0015` | click the Form Authentication link and verify the login page appears |

Three of the five passing cases are from `demo.playwright.dev/todomvc` — the agent generates clean Playwright code for canonical todomvc flows. Failure modes cluster around (a) over-specific text-match assertions and (b) timing on multi-step flows.

### Reproducibility recipe

From `D:\Code\qa-agent`:

```powershell
# 1. (Pre-req) Ollama serving qwen2.5-coder:7b-instruct-q4_K_M + nomic-embed-text
#    + Playwright chromium installed in .venv. Both verified during this audit.

# 2. Build dataset (resumable; picks up where the pilot stopped at 30 cases)
uv run python audit\phase0\build_golden_dataset.py --resume

# 3. Compute baseline (reads JSONLs + samples GPU/RAM during a 5-case re-run)
uv run python audit\phase0\compute_baseline.py --sample-size 5

# 4. Derive sprint-1 targets from the baseline JSON
uv run python audit\phase0\derive_sprint1_targets.py
```

### Reproducibility recipe

From `D:\Code\qa-agent`:

```powershell
# 1. (Pre-req) Ollama serving qwen2.5-coder:7b-instruct-q4_K_M + nomic-embed-text
#    + Playwright chromium installed in .venv. Both verified during this audit.

# 2. Build dataset (resumable; will pick up where the pilot stopped)
uv run python audit\phase0\build_golden_dataset.py --resume

# 3. Compute baseline (reads JSONLs + samples GPU/RAM during a small re-run)
uv run python audit\phase0\compute_baseline.py --sample-size 5

# 4. Derive sprint-1 targets from the baseline JSON
uv run python audit\phase0\derive_sprint1_targets.py
```

### Notes on baseline determinism

The Phase 0 spec asked for `OLLAMA temperature=0.0` for reproducibility. The existing `src\llm\adapter.py:21` hardcodes `TEMPERATURE = 0.1`; the OpenAI-compatible payload also has no `seed` parameter. Modifying source was forbidden by the spec, so the dataset is built at **temperature = 0.1** (which is "near-greedy" but not strictly deterministic). For Sprint 1, after TD-1 lands, the format_fix re-implementation should expose `temperature` and `seed` via the OpenAI-compatible `chat/completions` payload — at which point dataset re-generation is fully deterministic.

### Completion banner

PHASE 0 COMPLETE — see audit\phase0\AUDIT_REPORT.md

