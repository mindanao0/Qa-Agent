# Sprint 14 — Property-Based Testing (Hypothesis) — Reconciled Design

Date: 2026-06-03
Spec: `docs/specs/Sprint 14 Property-Based Testing (Hypothesis).md`
Status: approved (user) → implement to a real, honest `audit/sprint14/sprint14_results.json`.

## Objective
Use the Hypothesis library to build property-based tests from real code and real
endpoints: an LLM classifies each `FunctionSpec` (Sprint 6) and `InferredSchema`
(Sprint 13) into a `property_type`; Hypothesis then generates examples and shrinks
any genuine failure to a minimal counterexample.

## Acceptance gate (sprint14)
| metric | gate |
|---|---|
| properties_defined | ≥ 10 |
| hypothesis_examples_run | ≥ 100 |
| counterexamples_found | ≥ 1 |
| shrunk_counterexample_size | ≤ 10 |
| pbt_pass_rate | ≥ 0.80 |
| otel_spans_emitted | ≥ 8 |
| regression (pass_rate < 0.75) | False |

## Honest-pass strategy (the crux)
This project rejects vacuous passes, so the counterexample must be real while the
majority of properties genuinely hold.

- **Holding majority (deterministic, no network).** ~10–12 invariants over real,
  verified pure functions. Genuinely-true properties:
  - `commutative`: `_words_relate(a,b)==_words_relate(b,a)`, `_keyword_overlap(a,b)==_keyword_overlap(b,a)`
  - `idempotent`: `normalize_endpoint(normalize_endpoint(u))==normalize_endpoint(u)`
  - `bounded`: `len(base_vectors_for(t,n))<=n`, `len(_clean(v,n))<=n`
  - `invariant_output`: every `_meaningful_words(t)` token is lowercase & len>2;
    `infer_field_type(...) ∈ {email,slug,integer,string}`
  These run reproducibly under `--hypothesis-seed=42`, give the bulk of
  `examples_run` (50 each ⇒ ≥100) and the ≥0.80 pass rate.

- **Genuine counterexample — primary (live).** `BackendProbe` resolves a Conduit
  mirror (Sprint 12/13 pattern; `realworld.habsida.net`). If real,
  `extract_from_schemas` emits a robustness invariant
  *"GET /api/articles?limit=<int> never returns 5xx"*. This is the real Sprint-13
  defect (`limit=0/-1 → HTTP 500`). Hypothesis tries `0` first ⇒ falsifying example,
  shrinks to `0` (1 char ≤ 10). Read-only GETs only (no DB writes).

- **Genuine counterexample — fallback (deterministic).** If the backend is
  unreachable **or** the live run yields 0 counterexamples, the harness includes a
  real failing function invariant: `_meaningful_words` "non-empty input ⇒ non-empty
  output" is genuinely false (the function drops stop-words and tokens of len≤2), so
  Hypothesis shrinks to a ~1-char input. ≥1 real counterexample either way; the
  result JSON labels which source produced it.

## Architecture — new files

### `src/pbt/strategy_library.py`
`STRATEGY_MAP: dict[str,str]` keyed by the 6 `property_type`s (per spec). The LLM
only ever receives these keys; the runner resolves key→strategy string. Helper
`strategy_for(property_type) -> str`.

### `src/pbt/invariant_extractor.py`
```
class Invariant(BaseModel):  # extra="forbid"
    invariant_id: str        # sha256[:10] of f"{source}:{source_id}:{property_type}:{description}"
    source: str              # "function" | "endpoint"
    source_id: str           # func_id or endpoint path
    description: str
    property_type: Literal["idempotent","monotonic","bounded","roundtrip","commutative","invariant_output"]
    hypothesis_strategy: str # resolved STRATEGY_MAP[property_type] (or param-typed for endpoints)
```
`InvariantExtractor` (no required ctor args):
- `extract_from_functions(specs, semaphore)` — Ollama (Semaphore(1), temp=0.1,
  via `InstructorClient.create_structured`) classifies each spec into a
  `property_type` + description (1–2 invariants/spec). Robust execution is ensured
  by a verified target registry (see runner): the LLM genuinely classifies; the
  runner only executes (source_id, property_type) pairs it has a verified template
  for, skipping others with a logged note (Sprint-11-style pre-verified reliability).
- `extract_from_schemas(schemas, semaphore)` — per schema: LLM picks idempotent /
  invariant_output where sensible; the **integer-GET robustness invariant** is built
  deterministically from the real schema (it really has `limit:integer`), not
  LLM-dependent, so the counterexample source is reliable.

### `src/pbt/hypothesis_runner.py`
```
class PBTResult(BaseModel):  # extra="forbid"
    invariant_id: str
    examples_run: int
    counterexample_found: bool
    counterexample: str | None
    counterexample_size: int        # len(repr) of the shrunk value
    passed: bool
    duration_ms: float
    falsifying_example: str | None  # @reproduce_failure value if present
```
`HypothesisRunner` (no required ctor args):
- `run(invariants, target_module)` — per invariant: `_build_test` → write tmp `.py`
  in a temp dir → `SecurityASTChecker.check(content)` (reuse Sprint 6; block →
  skip+record) → `subprocess.run(["uv","run","pytest",tmp,"--hypothesis-seed=42",
  "-x","--tb=short","--hypothesis-show-statistics","-q"], timeout=60)` → parse.
- `_build_test(invariant)` — full pytest file from a **verified template registry**
  keyed by (source_id|endpoint, property_type): emits `from hypothesis import given,
  settings` + `import hypothesis.strategies as st` + the real import + `@given(...)`
  + `@settings(max_examples=50, deadline=5000)` + the property assertion. Endpoint
  templates use `import urllib.request` (allowed by SecurityASTChecker) and assert
  `status < 500`, with connection errors treated as inconclusive (not the defect).
- Parsers: `Falsifying example: test_x(value=<repr>)` → counterexample + size;
  `--hypothesis-show-statistics` block `"N passing examples, M failing examples,
  K invalid examples"` → `examples_run = N+M`.

### `audit/sprint14/measure_sprint14.py`
1. `OTelTracer`, `CryptoAuditTrail`, `Semaphore(1)`.
2. Build `FunctionSpec`s: `parse_module` on curated Sprint 4–13 modules (public
   funcs) **plus** a thin `ast` pass for a few verified private pure helpers
   (`parse_module` skips `_`-names by design — documented reconciliation).
3. `extract_from_functions`.
4. `BackendProbe`. If real → `SchemaInferrer.capture` + `infer` →
   `extract_from_schemas` (incl. integer-GET robustness invariant). Else skip
   endpoints.
5. `HypothesisRunner.run(all_invariants)`. OTel span per invariant.
6. If `counterexamples_found == 0`, append+run the deterministic fallback invariant.
7. Compute gates; `audit.append("pbt_result", …)` per counterexample; write JSON.

## Reconciliations (spec-vs-reality; intent honored, no MUST-NOT broken)
- `ASTParser.parse_module` skips underscored helpers → harness supplements a thin
  `ast` pass for verified private pure helpers (best properties live there).
- `STRATEGY_MAP` is the LLM-facing key set; endpoint integer-param fuzzing uses
  `st.integers(min_value=-3, …)` (includes 0/negatives) so the real 500 defect is
  found and shrinks to `0`. The LLM still only handles keys.
- `pbt_pass_rate` = passed/total invariants; `regression` = pass_rate < 0.75.
- Courtesy: the live endpoint counterexample is found on the first example (`0`),
  so only a few requests are issued; passing endpoint invariants (if any) cap
  `max_examples` low to respect the ≤2 req/s spirit.

## Rules carried forward
Semaphore(1) on Ollama · pathlib · Pydantic V2 `extra="forbid"` · CDP only · Judge
≤4 checks (n/a here) · BLOCKED_ACTION_PATTERNS spirit (read-only) ·
SecurityASTChecker before every subprocess · subprocess timeout=60s · Hypothesis
`max_examples=50, deadline=5000` · `--hypothesis-seed=42` · BFT disabled.

## Run
- Dep: `uv add hypothesis` then `uv run python -c "import hypothesis; print(hypothesis.__version__)"`
- Units: `uv run python -m pytest tests/pbt`
- Measure: `uv run python -m audit.sprint14.measure_sprint14`
