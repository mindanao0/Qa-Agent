# QA Agent — Project Specs Index

## Status
- SPEC_CORE: ✅ Implemented
- SPEC_UNIVERSAL_DOMAIN: ✅ Implemented
- SPEC_FINETUNE_RAG: ✅ Implemented
- SPEC_WSL2_FINETUNE: ✅ Implemented
- SPEC_SEMANTIC_CACHE (Priority 3): ✅ Implemented
- SPEC_CI_CD_BIFURCATED (Priority 4): ✅ Implemented
- SPEC_OBSERVER_DRIVER (Priority 5): ✅ Implemented

## Spec Files
All specifications are in docs/specs/

## Active Task
Sprint 9 CLOSED 2026-06-02 — JS/TS Code Testing complete. All gates PASS.
Results: js_functions_parsed=12, tests_generated=12, test_pass_rate=0.8333,
metamorphic_pairs=10, otel_spans_emitted=5, regression=false.
See audit/sprint9/sprint9_results.json.
New components: JSASTParser (Babel AST via Node.js subprocess → JSFunctionSpec),
JSTestGenerator (Ollama → self-contained Vitest tests, Semaphore(1)),
JSCodeJudge (4 checks: has_expect, no_settimeout, valid_vitest_sig, metamorphic_valid),
JSTestExecutor (npx vitest run --reporter=json, OTelTracer.span wrapper).
JS targets: src/js_targets/ (12 exported functions: utils, validators, formatters).
Node.js entry: scripts/ast_walker.js (@babel/parser + @babel/traverse).
Key lesson: 7B model needs actual source verbatim in prompt + explicit 3-part structure
(implementation → vitest import → describe/it blocks) to generate reliable tests.
Avoid regex backslash sequences ([\s_]+) in source targets — use [ _]+ for portability.
Next: Sprint 10 — TBD.

Prior sprints (carry-over notes):
- Sprint 8: CLOSED 2026-06-01 — CI/CD GitHub Actions + Observability complete. All gates PASS.
  Results: pipeline_stages_defined=4, otel_spans_emitted=5, structured_log_fields=9,
  audit_trail_entries=10, audit_chain_valid=true. See audit/sprint8/sprint8_results.json.
  Components: OTelTracer, StructuredLogger, CryptoAuditTrail, AgentMetrics.
  All 5 LangGraph nodes wrapped with spans. GitHub Actions: qa_agent.yml (4 stages).

- Sprint 7: CLOSED — Shadow DOM + SPA Agent + self-heal. All gates PASS.
  Results: shadow_dom_elements_found≥5, spa_transitions_handled≥3, test_pass_rate≥0.75,
  self_heal_triggered≥1. See audit/sprint7/sprint7_results.json.

- Sprint 4: CLOSED 2026-05-30 — ContractSkill + SFG Crawler live. All gates PASS.
  Results: after_pass_rate=1.00, contract_skills_compiled=4, sfg_nodes_discovered=15,
  skills_used_in_generation=4, regression=false. See audit/sprint4/sprint4_results.json.
  Key fix: CrawlerConfig expanded to max_pages=20, max_depth=4, max_time_minutes=10.
  Active components: SFGCrawler (BFS crawler → SFGStore SQLite), ContractSkillCompiler
  (trajectory → ContractSkill), bft_generator_node CONTRACT_CACHE_HIT path (skill injection).

- Sprint 3: CLOSED 2026-05-29 — BFT deactivated (hardware constraint), Judge strengthened.
  Root cause: 7B model + Pydantic V2 strict + max_retries=1 → T>0 generators fail schema,
  quorum impossible. See audit/sprint3/SPRINT3_FINAL_LOG.md.
  Rule: Judge complexity ≤ 4 checks until model upgrade to ≥14B.
  BFT code kept dormant behind `llm.bft.enabled` — reactivate when model ≥14B or VRAM ≥12GB.

- Sprint 2: CODE-COMPLETE 2026-05-26 (FuzzyMatcher, StateValidator, EpisodicStore, AIHealer + graph integration).
- TD-16: AOMExtractor only emits checked/disabled/expanded/focused/selected — aria-busy detection in StateValidator is best-effort.
- TD-17: subprocess executor in graph.py executor_node has no live page, so post-action state validation is scaffolding only.

## TestExecutor
- Entry point: `src/codetest/executor.py` (`TestExecutor`)
- Timeout: 30s (`_TIMEOUT_SECONDS`) — kills entire process tree on timeout
- Cross-platform kill: Windows uses `taskkill /F /T /PID`, Linux uses `os.killpg` + `SIGKILL`
- Subprocess isolation: Windows `CREATE_NEW_PROCESS_GROUP`, Linux `start_new_session=True`
- TD-18 FIXED 2026-05-31 — proc.kill() only killed direct child; orphan processes burned CPU indefinitely. Fixed via process-tree kill.

## Rules (always apply)
- OS: Windows 11 + WSL2 (executor cross-platform ready for Linux migration)
- LLM: Ollama localhost:11434 only — no external APIs
- Python 3.11+ async-first
- VRAM: 6GB GPU, 16GB RAM
- Locators: get_by_role, get_by_label, get_by_text, get_by_test_id ONLY
- Never use CSS selectors or XPath
- Temperature: 0.1 for all LLM calls

## LLM
- Preferred structured output entry point: `src/llm/instructor_client.py` (`InstructorClient`)
- Legacy fallback: `src/llm/structured.py` (`enforce_json_output`) — active when `structured_output_engine: "legacy_repair"` in `config/agent.yaml`
- Feature flag: `config/agent.yaml → llm.structured_output_engine` (or `STRUCTURED_OUTPUT_ENGINE` env var)
- Canonical Pydantic V2 schemas: `src/llm/schemas.py` (all have `extra="forbid"`)
- Sprint 1 verdict: FAIL on latency/parse-rate targets; no regression; Day 3-5 DOM Pruner approved to proceed

## Perception
- PAM entry point: `src/perception/grounder.py` (`Grounder`) — orchestrates the 4-layer Universal DOM Compression Pipeline
- Pipeline: `AOMExtractor` → `DOMPruner` → `SemanticCompactor` → `Grounder`
- Feature flag: `config/agent.yaml → perception.use_grounder` (read via `get_use_grounder()` from `src/config_loader.py`)
- Token budget: `config/agent.yaml → perception.context_budget_tokens` (read via `get_context_budget_tokens()`)
- Agents receive page state via `page_state: str = ""` parameter (planner.plan(), generator.generate())
- Metrics emitted to: `audit/phase0/grounder_metrics.jsonl` (JSONL, one record per ground() call)
- Technical debt open: TD-12 (shadow DOM), TD-14 (bbox deferred) — deferred to Sprint 2
- TD-13 FIXED 2026-05-24 — tiktoken cl100k_base estimator in src/perception/token_counter.py
- TD-15a FIXED 2026-05-24 — AOMExtractor uses CDP Accessibility.getFullAXTree (page.accessibility removed in Playwright ≥1.34)
- TD-15b FIXED 2026-05-24 — DOMPruner JS getClassString() handles SVGAnimatedString.baseVal
- Emergency fallback: Grounder.ground() NEVER raises — returns CompactPAM(source='failure') with URL+title on complete failure
- CompactPAM source Literal: ["aom", "dom", "hybrid", "failure"]
