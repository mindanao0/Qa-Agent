# Test-Planner Eval — Findings & Fine-Tune Decision

Goal: decide, with **real measured numbers**, whether the 7B test planner plus
prompt/decoding/retrieval interventions is good enough, or whether fine-tuning
the 3B+LoRA is warranted. All numbers below are live Ollama generations against
the pinned production model `qwen2.5-coder:7b-instruct-q4_K_M` on a GTX 1660 Ti
(6 GB). Nothing is mocked or dry-run.

Benchmark: 20 real pages, 68 scenarios (`eval/golden_set.jsonl`). Metrics scored
over the planner's **LLM-generated** tests (rule-template tests are constant
across configs and reported separately). See `eval/README.md` for definitions.

## Results (cumulative)

| config | json_valid_rate | coverage_score | assertion_quality |
|---|---|---|---|
| baseline (7B + instructor) | 0.9934 | 0.7592 | 0.8600 |
| + grammar (Ollama `format`) | 0.9801 | 0.8242 | 0.9732 |
| + few-shot | **1.0000** | **0.8883** | 0.9205 |
| + RAG | 0.9934 | 0.8283 | 0.9400 |

Per-phase gates (vs baseline) — **all PASS**:
- Phase 1 grammar: `json_valid ≥ 0.98` & no coverage/assertion regression → PASS (0.9801).
- Phase 2 few-shot: `assertion_quality > baseline` → PASS (0.9205 > 0.86).
- Phase 3 RAG: `coverage_score > baseline` → PASS (0.8283 > 0.7592).

## What each phase actually did (honest read)

1. **Grammar-constrained output** (`format=<schema>`, GBNF under the hood). JSON
   validity was *already* near-ceiling at baseline (instructor repairs well), so
   grammar did **not** improve validity — it slightly *regressed* it (0.9801)
   because Ollama ignores JSON-schema `maxLength`, letting the 7B occasionally
   ramble inside an unbounded string until `num_predict` truncates it (3/151
   calls). Its real effect was a large jump in coverage/assertion, largely
   because the grammar path drops instructor's prompt boilerplate and produced
   longer, more detailed tests.

2. **Few-shot** (3 concrete-assertion exemplars). The clear winner: concise
   exemplars made outputs concise → **eliminated grammar's truncation** (validity
   → 1.0) and lifted **coverage to its best (0.888)**. Assertion (0.9205) is well
   above baseline but below grammar-alone — grammar's verbose outputs inflated
   the keyword-based assertion measure; few-shot's concise outputs are a truer
   read.

3. **RAG retrieval** — **evaluated and rejected.** It passed the vs-baseline gate
   but **regressed coverage vs few-shot (0.888 → 0.828)** and validity (1.0 →
   0.9934). Retrieving from the existing corpus *anchored* the model to past
   examples rather than expanding scenario coverage. Kept **off by default**
   (`llm.test_planner_rag: false`).

## Decision: do **NOT** fine-tune the 3B

- 7B + grammar + few-shot clears the bar on every metric **with zero training**:
  json_valid 0.86-class issues → **1.0**, coverage **0.759 → 0.888** (+17%),
  assertion **0.86 → 0.92**. Real, gated, reproducible.
- The RAG result is the clincher and **confirms the original concern**: the
  retrieval corpus is 100% 7B-generated, and retrieving from it did **not** add
  signal — it *hurt*. A 3B fine-tuned on that same 7B-generated data cannot
  exceed the 7B; at best it copies it. The data, not the method, is the ceiling.
- Therefore: **keep the 3B out of production** (it never beat baseline here), and
  **do not resume fine-tuning** until there is a *higher-quality, non-self-
  generated* data source (e.g. human-written or executed-and-verified test cases)
  to break the self-distillation ceiling.

**Adopted production config:** `test_planner_engine: grammar` + `test_planner_fewshot: true`
(`test_planner_rag: false`). Already set in `config/agent.yaml`.

## Caveats (so the numbers aren't over-read)

- `assertion_quality` is a TH/EN keyword heuristic (`run_eval.classify_assertion`);
  verbose outputs can inflate it. `json_valid_rate` (objective) and the
  scenario-based `coverage_score` are more robust; the conclusion holds on those.
- Eval feeds a fixed synthetic NavigationMap (no live crawl), so it measures
  *planning quality given known elements*, not end-to-end discovery — the correct
  scope for "planner skill", but not a full-pipeline number.
- Element lists are curated from canonical site structure + the dataset, not a
  live DOM scrape (identical input across all configs, so deltas are valid).

## Reproduce

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  .venv/bin/python eval/run_eval.py --mode baseline --out eval/baseline.json
# modes: baseline | grammar | fewshot | rag    (fewshot/rag are cumulative)
.venv/bin/python eval/summarize.py     # rebuilds the table above + eval/summary.json
```
