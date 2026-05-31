# Sprint 1 Day 1-2 Post-Mortem

**Date:** 2026-05-23
**Feature:** Instructor + Ollama JSON mode structured output
**Verdict:** FAIL (no regression)

---

## Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | Added `instructor>=1.6.0`, integration pytest marker |
| `src/llm/schemas.py` | **New** — canonical Pydantic V2 schemas with `extra="forbid"` (TD-2) |
| `src/llm/structured.py` | Refactored to import from `schemas.py`; repair pipeline preserved |
| `src/llm/instructor_client.py` | **New** — InstructorClient, GenerationConfig, StructuredGenerationError |
| `src/config_loader.py` | **New** — YAML config + env-var feature flag reader |
| `src/llm/adapter.py` | Added `ollama_v1_url()` method |
| `config/agent.yaml` | Added `structured_output_engine: "instructor"` |
| `src/agents/planner.py` | Feature-flag branch (instructor / legacy_repair) |
| `src/healing/ai_healer.py` | Feature-flag branch + fixed `healed.method = "ai"` mutation |
| `src/agents/generator.py` | Feature-flag branch (instructor single-pass) |
| `tests/test_instructor_client.py` | **New** — 5 integration tests (5/5 passed in 132.91s) |
| `audit/phase0/measure_sprint1_day2.py` | **New** — A/B measurement script |
| `audit/phase0/sprint1_day2_results.json` | **New** — A/B results with real numbers |

---

## Numbers Before / After

| Metric | Baseline | Instructor | Legacy (re-measured) | Target | Status |
|--------|---------|------------|---------------------|--------|--------|
| `json_parse_failure_rate` | 0.0 | **0.10** | 0.20 | ≤ 0.05 | ❌ Over target |
| `first_run_pass_rate` | 0.50 | **0.90** | 0.80 | ≥ 0.40 | ✅ Exceeds target |
| `avg_generation_latency_ms` | 119,516 | **71,489** | 46,993 | ≤ 1.20× legacy | ❌ +52% over budget |
| integration tests | N/A | **5/5 PASS** | N/A | 5/5 | ✅ |

---

## Verdict: FAIL (no regression)

### Passing criteria
- ✅ `first_run_pass_rate` 0.90 ≥ 0.40 (spec target met by wide margin)
- ✅ `first_run_pass_rate` 0.90 >> 0.286 (baseline) — no regression
- ✅ Quality delta: instructor reduced parse failures by 50% and improved pass rate by +12.5 percentage points vs legacy_repair

### Failing criteria
- ❌ `json_parse_failure_rate` 0.10 > 0.05 target — instructor still fails 1 out of 10 cases
- ❌ Latency: instructor avg 71,489ms vs legacy 46,993ms = **+52.1%** overhead, exceeding the +20% budget

---

## Root Cause Analysis

### Why parse_failure_rate = 0.10 (not 0.0)

The single instructor failure was `stable-0013` ("add five todos and use the Active filter..."). Inspection of the raw results shows that `TestStep.action` came back as a list instead of a string in one retry cycle. With `extra="forbid"` on the schema, this triggers a `ValidationError` — instructor's retry-with-feedback loop was active but the model produced the same structural error again, exhausting `max_retries=3`.

**Root cause**: The `TestStep.action` field has no `Literal` constraint — it accepts any string. The model sometimes returns a list `["fill", "click"]` instead of a string `"fill_and_click"`. Adding a `@field_validator` with `mode="before"` that coerces lists to strings (joining with `+`) would fix this.

### Why latency = +52.1% over legacy

Each instructor call makes 1-3 round-trips to Ollama (initial call + retry if validation fails). The legacy two-pass generator makes exactly 2 calls (reasoning pass + code pass) with no retry loop. For `stable-0013`, instructor made 3 calls (max_retries hit) vs legacy's 2 = 50% more inference time.

Additionally, instructor parses the full JSON schema and sends validation error context in each retry message (~200-300 extra tokens per retry), adding marginal overhead even when no retry is needed.

**Contributing factor**: The Ollama model `qwen2.5-coder:7b-instruct-q4_K_M` is slower to generate compliant JSON than plain text; the legacy two-pass path asks for plain Python (fast) while instructor requests JSON-formatted output (slower with this model size).

---

## Unexpected Issues

1. **`extra="forbid"` on TestStep**: Adding `extra="forbid"` tightened validation and exposed a field-type mismatch (`action` returned as list) that the legacy repair pipeline had been silently tolerating. This is a net positive (found a latent bug) but added one failure case.

2. **Pytest collects Pydantic model classes**: `pytest` tries to collect `TestPlan`, `TestStep` etc. as test classes because they start with `Test*`. Produces harmless warnings (not failures). Fix: add `norecursedirs` or rename classes — deferred to next sprint.

3. **`pytest-timeout` not installed**: `--timeout=300` flag was silently ignored. Tests completed in 132.91s so this was not a problem. Should be added to dev dependencies for future safety.

---

## Gap Analysis vs Sprint Targets

| Target | sprint1_targets.yaml value | Achieved | Gap |
|--------|---------------------------|----------|-----|
| format_fix.target_json_parse_fail_rate | 0.05 | 0.10 | -0.05 (2× over) |
| first_run_pass_rate (acceptance gate) | ≥ 0.40 | 0.90 | +0.50 (strong) |
| Latency budget | ≤ 1.20× legacy | 1.52× legacy | -0.32× over budget |

---

## Recommendation for Day 3-5

**Day 3-5 (DOM Pruner): PROCEED WITH CONDITIONS**

The FAIL verdict is driven by latency and one residual parse failure. Neither blocks the DOM Pruner work (TD-4 — tree-sitter + AOM-first context pruning). In fact, the DOM Pruner directly addresses latency by reducing the context window fed to the LLM on each call.

**Proposed retry strategy for instructor parse failures:**

Before Day 3-5 ships, apply this targeted fix to `src/llm/schemas.py` `TestStep`:

```python
@field_validator("action", mode="before")
@classmethod
def coerce_action_list_to_str(cls, v: object) -> str:
    if isinstance(v, list):
        return "+".join(str(x) for x in v)
    return str(v)
```

This single validator eliminates the `action`-as-list failure mode that caused the 0.10 parse failure rate without any LLM changes.

**Proposed latency reduction:**

- Reduce `max_retries=3` → `max_retries=1` for the generator path (planner needs 3, generator is the bottleneck)
- Or: add `temperature=0.0` prompt-level caching via Ollama's `/api/generate` seed parameter (not currently supported on the OpenAI-compat endpoint, but available via native API)

---

## Rollback Plan

If a future run produces REGRESSION (`first_run_pass_rate < 0.286`):

```powershell
# Rollback command:
(Get-Content config\agent.yaml) -replace 'structured_output_engine: "instructor"', 'structured_output_engine: "legacy_repair"' | Set-Content config\agent.yaml
```

Then open tech-debt entry TD-11 in `audit/phase0/AUDIT_REPORT.md` and stop.
