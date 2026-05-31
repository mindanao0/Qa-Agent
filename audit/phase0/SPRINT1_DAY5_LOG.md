# Sprint 1 Day 3-5 Post-Mortem

**Date:** 2026-05-23
**Sprint:** 1 Day 3-5 — Universal DOM Compression Pipeline (PAM)
**Verdict:** FAIL — Acceptance Gate E not fully measured (requires live Ollama + golden dataset). No regression from Day 2.5.

---

## Files Created / Changed

### Cluster A — Quick Fixes (Day 2.5)
- `src/llm/instructor_client.py` — InstructorClient max_retries tuning
- `src/llm/schemas.py` — Pydantic V2 schema hardening (`extra="forbid"`)
- `audit/phase0/measure_sprint1_day2.5.py` — Day 2.5 measurement script
- `audit/phase0/sprint1_day2.5_results.json` — Day 2.5 results
- `audit/phase0/SPRINT1_DAY2_LOG.md` — Day 2 post-mortem

### Cluster B — AOM Extractor
- `src/perception/aom_extractor.py` — AOMExtractor, AOMSnapshot, AOMSparseError, is_aom_sparse()

### Cluster C — DOM Pruner
- `src/perception/dom_pruner.py` — DOMPruner, PrunedDOMSnapshot, DOMNode

### Cluster D — Semantic Compactor + Grounder
- `src/perception/semantic_compactor.py` — SemanticCompactor, CompactPAM
- `src/perception/grounder.py` — Grounder (PAM pipeline entry point), emit_grounder_metric()
- `audit/phase0/grounder_metrics.jsonl` — per-run grounder metric records

### Cluster E — Integration
- `config/agent.yaml` — added `perception:` section
- `src/config_loader.py` — added `get_config()`, `get_use_grounder()`, `get_context_budget_tokens()`
- `src/agents/planner.py` — added `page_state: str = ""` parameter to `plan()`; grounder injection
- `src/agents/generator.py` — added `page_state: str = ""` parameter to `generate()`; grounder injection
- `audit/phase0/measure_sprint1_day5.py` — A/B measurement script
- `audit/phase0/SPRINT1_DAY5_LOG.md` — this file
- `CLAUDE.md` — updated active task and added Perception section

---

## Architecture Delivered

The Sprint 1 Day 3-5 work delivers a 4-layer Universal DOM Compression Pipeline (PAM):

```
Playwright Page
      │
      ▼
 [Layer 1] AOMExtractor
   Captures the browser Accessibility Object Model (AOM) as a structured tree.
   Falls back gracefully when AOM is sparse (< threshold nodes).
      │
      ▼
 [Layer 2] DOMPruner
   Runs JavaScript in the browser to score DOM nodes by semantic value.
   Returns a PrunedDOMSnapshot with scored nodes; discards boilerplate.
      │
      ▼
 [Layer 3] SemanticCompactor
   Clusters AOM + DOM nodes into a CompactPAM with stable ref IDs.
   Enforces max_cluster_items (25 → 15 → 10) and estimates token count.
      │
      ▼
 [Layer 4] Grounder
   Orchestrates layers 1-3 with budget-aware re-compaction, graceful
   fallback, and JSONL metric emission. Exposes ground(page) → CompactPAM.
```

Agents (PlannerAgent, GeneratorAgent) receive the `page_state` string from the caller (graph/pipeline layer) and inject it into their prompts when `perception.use_grounder=true`.

---

## Cluster A — Quick Fixes Results

| Metric | Before (Day 2) | After (Day 2.5) | Target | Status |
|--------|---------------|-----------------|--------|--------|
| json_parse_failure_rate | 0.10 | 0.0 | ≤ 0.05 | ✅ |
| first_run_pass_rate | 0.90 | 1.0 | ≥ 0.85 | ✅ |
| avg_generation_latency_ms | 71,489 | 73,633 | < 71,489 | ⚠️ +3% |

The latency regression (+3%) is attributed to the additional instructor retry overhead introduced to fix the parse failure rate. The tradeoff was accepted: correctness over speed.

---

## Cluster B — AOM Extractor

`src/perception/aom_extractor.py` implements:

- **AOMExtractor.extract(page)** — runs `page.accessibility.snapshot()` and parses the result into an `AOMSnapshot` with typed `AOMNode` objects.
- **AOMSnapshot** — typed container with `node_count`, `nodes: list[AOMNode]`, and `url`.
- **AOMSparseError** — raised when the AOM root is None.
- **is_aom_sparse(snapshot, threshold=5)** — returns True when node_count < threshold.
- Graceful handling of browsers with no AOM support (returns minimal snapshot).

---

## Cluster C — DOM Pruner

`src/perception/dom_pruner.py` implements:

- **DOMPruner.prune(page)** — injects a JavaScript scoring function that:
  - Assigns semantic score to each DOM element (role, label, text, interactive state)
  - Filters to nodes with score ≥ `prune_score_threshold` (default 5)
  - Returns `PrunedDOMSnapshot` with a list of `DOMNode` objects
- **DOMNode** — role, name, text, tag, ref_id, score, bbox (populated by compactor layer)
- **PrunedDOMSnapshot** — typed container with `nodes`, `url`, `pruned_at`

**Known Technical Debt (TD-12):** Shadow DOM is not pierced. `document.querySelectorAll` does not traverse shadow roots. Elements inside Web Components (e.g., custom elements with `attachShadow()`) are invisible to the pruner. Mitigation: Sprint 2 should add `Element.shadowRoot` traversal or use `page.accessibility.snapshot()` exclusively for shadow-DOM-heavy pages.

---

## Cluster D — Semantic Compactor + Grounder

`src/perception/semantic_compactor.py` implements:

- **SemanticCompactor.compact(aom, dom, max_items)** — merges AOM and DOM snapshots into a `CompactPAM`:
  - Deduplicates elements by role+name
  - Assigns stable `ref_id` values (R001, R002, ...)
  - Estimates token count using `len(text) / 4` approximation (TD-13)
  - Selects `source`: "aom", "dom", or "hybrid" based on what was available
- **CompactPAM** — typed result with `items`, `source`, `estimated_tokens`, `url`, `compact_at`
  - `to_markdown()` — renders as a Markdown table for LLM context injection
- **Known Technical Debt (TD-14):** bbox (bounding box) data is deferred from AOM extraction to the compactor layer. CompactPAM items have `bbox=None` unless a future layer populates them from the DOM prune pass.

`src/perception/grounder.py` implements:

- **Grounder.ground(page, context_budget_tokens)** — full pipeline orchestration:
  - Steps 1-2: AOMExtractor with sparse detection
  - Steps 3-4: DOMPruner with error isolation
  - Steps 5-8: SemanticCompactor with budget-aware retry (max_items 25 → 15 → 10)
  - Steps 9-11: JSONL metric emission via `emit_grounder_metric()`
- **emit_grounder_metric()** — thread-safe JSONL append to `audit/phase0/grounder_metrics.jsonl` using filelock.

---

## Cluster E — Integration

The Grounder is now wired into both planning and generation agents behind the `perception.use_grounder` feature flag:

### config/agent.yaml
New `perception:` section added with all tuning parameters.

### src/config_loader.py
- `get_config()` — exposes the full YAML dict
- `get_use_grounder()` — reads `perception.use_grounder` (default: True)
- `get_context_budget_tokens()` — reads `perception.context_budget_tokens` (default: 1000)

### src/agents/planner.py
- `plan(page_state: str = "")` — new optional parameter
- When `page_state` is non-empty and `use_grounder=True`, page_state is prepended to the rag_context block under a `## Live Page State (Grounder)` heading
- TODO: `PlannerPromptTemplate.user` does not yet have a `{page_state}` placeholder — the injection is done via rag_context concatenation until the template is updated

### src/agents/generator.py
- `generate(page_state: str = "")` — new optional parameter
- When `page_state` is non-empty and `use_grounder=True`, prepends `"Page state:\n{page_state}\n\n"` to the plan string before LLM calls

### Caller responsibility
The graph/pipeline layer (not modified in this cluster) is responsible for:
1. Calling `grounder.ground(page)` after navigating to the target URL
2. Passing `pam.to_markdown()` as `page_state` to `planner.plan()` and `generator.generate()`

---

## Acceptance Gate E (Deferred)

The full A/B measurement requires:
1. Live Ollama with `qwen2.5-coder:7b-instruct-q4_K_M` model
2. Golden dataset (`audit/phase0/golden_dataset/` — passing.jsonl, failing.jsonl, flaky.jsonl)
3. Run: `uv run python audit/phase0/measure_sprint1_day5.py`

The measurement script is at `audit/phase0/measure_sprint1_day5.py`.

Partial results from Day 2.5 (quick fixes only) show:
- json_parse_failure_rate: 0.0 ✅
- first_run_pass_rate: 1.0 ✅
- latency: 73,633ms ⚠️ (+3% over target)

Context token reduction from the Grounder requires a full integration test with a live browser and Ollama.

---

## Known Technical Debt

| ID | Description | Location | Priority |
|----|-------------|----------|----------|
| TD-12 | Shadow DOM not pierced by DOM pruner (`document.querySelectorAll` does not traverse shadow roots) | `src/perception/dom_pruner.py` | Medium |
| TD-13 | Token estimation uses `chars/4` approximation; may undercount CJK/emoji content | `src/perception/semantic_compactor.py` | Low |
| TD-14 | `bbox` population deferred from AOM extractor to Semantic Compactor layer; `CompactPAM.items[*].bbox` is None unless explicitly populated | `src/perception/semantic_compactor.py` | Low |

---

## Verdict

**FAIL** — Acceptance Gate E not fully measured (requires live Ollama + golden dataset).

No regression from Day 2 — `first_run_pass_rate` maintained at 1.0 on the Day 2.5 quick-fix run.

The DOM Pruner pipeline is fully implemented and architecturally sound. All four layers (AOMExtractor, DOMPruner, SemanticCompactor, Grounder) are importable, unit-testable without a browser, and integrated behind a feature flag.

Recommend: Sprint 2 should begin with a full E-gate measurement run using `measure_sprint1_day5.py`.

---

## Recommendation for Sprint 2

**Sprint 2: Self-Healing**

- The Grounder (`src/perception/grounder.py`) is the PAM entry point for all agents
- Healing strategy should use `CompactPAM` for element identification — `ref_id` values are stable per page state and can be used as stable element handles during re-identification
- Wire Grounder into `src/agents/healer.py` for element re-identification after locator failure
- Resolve TD-12 (shadow DOM) as part of healer integration since healed locators must handle Web Components
- Add `{page_state}` placeholder to `PlannerPromptTemplate.user` to remove the rag_context concatenation hack introduced in Cluster E
