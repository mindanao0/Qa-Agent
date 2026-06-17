# Test-Planner Eval Benchmark

A golden-set benchmark that measures how good the Universal test planner is, so
we can decide — with **real numbers, not vibes** — whether the 7B base model
plus prompt/retrieval interventions is good enough, or whether fine-tuning is
actually warranted.

> Motivation: the 3B+LoRA fine-tune was trained on data distilled 100% from the
> 7B, so at best it copies the 7B — it cannot fix a root-cause quality problem,
> and there was no benchmark to prove any change helps. This benchmark is that
> missing measuring stick.

## Files

| File | Purpose |
|------|---------|
| `golden_set.jsonl` | 20 real pages (saucedemo, the-internet, parabank, demoqa, orangehrm, automationexercise, realworld.habsida.net). Each: `url`, `elements`, `expected_test_count`, `expected_assertion_types`, `expected_scenarios` (bilingual keywords). |
| `_build_golden.py` | Reproducible builder for `golden_set.jsonl` (provenance documented in-file). |
| `run_eval.py` | Drives the **real** `UniversalTestPlanner.plan_from_map` over each page via live Ollama and computes the metrics. |
| `baseline.json` / `phase1_grammar.json` / … | One result file per planner config. |

## Metrics (per config)

All three are computed over the planner's **LLM-generated** tests
(`type=="functional"`: functional / negative / edge). The planner's rule-based
template tests (accessibility / security / form_validation / broken_link /
error_page / search / logout) are constant across configs and are reported
separately under `diagnostics` so the headline stays sensitive to the actual
intervention.

- **`json_valid_rate`** — fraction of the planner's LLM calls whose **first**
  generation parses + validates against the Pydantic schema, with **no repair
  retry**. This is the raw structured-output reliability Phase-1 grammar
  constraint targets ("JSON พัง / markdown leak"). Baseline/few-shot/RAG use
  instructor JSON-mode (`max_retries=1`); grammar uses Ollama
  `format=<schema>` constrained decoding.
- **`coverage_score`** — mean over pages of `covered_scenarios / scenarios`. A
  scenario counts as covered when any of its bilingual keywords appears in any
  LLM-generated test for that page. Scenario-based (not raw count) so it does
  **not** saturate and can rise when retrieval surfaces missed behaviours.
- **`assertion_quality`** — fraction of LLM-generated tests whose assertion
  checks **concrete state** (a specific URL reached, a validation/error message,
  a content/count/state change, an HTTP/WCAG check) instead of a generic "page
  responds / loads / is visible". Classifier: `run_eval.classify_assertion`
  (TH + EN patterns; `sample_tests` in each result file lets you audit it).

## Run

```bash
# baseline (Phase 0)
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  .venv/bin/python eval/run_eval.py --mode baseline --label baseline --out eval/baseline.json

# smoke (first 2 pages)
.venv/bin/python eval/run_eval.py --mode baseline --limit 2
```

Model is pinned to `qwen2.5-coder:7b-instruct-q4_K_M` (config/agent.yaml). Every
number is a live generation — nothing is mocked.

## Acceptance gates (per the plan)

| Phase | Intervention | Gate |
|-------|--------------|------|
| 0 | — | `baseline.json` exists with real 3-metric numbers |
| 1 | grammar-constrained output | `json_valid_rate ≥ 0.98` **and** no regression in coverage/assertion |
| 2 | few-shot prompting | `assertion_quality` meaningfully above baseline |
| 3 | RAG retrieval (quality=1.0 only) | `coverage_score` above baseline |

If 7B + grammar + few-shot + RAG clears the bar, fine-tuning is unnecessary. The
3B is **not** wired to production unless it beats baseline here.
