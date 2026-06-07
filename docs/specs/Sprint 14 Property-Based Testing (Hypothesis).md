## CONTEXT
PROJECT: D:\Code\qa-agent
BASELINE: Sprint 13 PASS — SchemaInferrer + InferredSchema available,
          AdvancedVectorGenerator producing typed vectors per field
STACK: Python 3.13 + uv, Ollama, LangGraph async, Playwright CDP
NEW DEP: hypothesis (uv add hypothesis)

## OBJECTIVE
GOAL: ใช้ Hypothesis library สร้าง property-based tests จาก
      InferredSchema (Sprint 13) และ FunctionSpec (Sprint 6)
      โดยให้ LLM นิยาม invariants แล้วให้ Hypothesis หา counterexamples

## ACCEPTANCE GATE (sprint14)
  properties_defined          ≥ 10    (invariants across functions + endpoints)
  hypothesis_examples_run     ≥ 100   (Hypothesis generates ≥100 examples total)
  counterexamples_found       ≥ 1     (Hypothesis shrinks to minimal failing case)
  shrunk_counterexample_size  ≤ 10    (minimal repr ≤ 10 chars/elements)
  pbt_pass_rate               ≥ 0.80  (properties that hold under all examples)
  otel_spans_emitted          ≥ 8
  REGRESSION if pass_rate     < 0.75

## ARCHITECTURE — New files:

### src/pbt/invariant_extractor.py  (NEW)
# InvariantExtractor — LLM extracts testable invariants from FunctionSpec + InferredSchema
#
# class Invariant(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     invariant_id:   str          # sha256[:10]
#     source:         str          # "function" | "endpoint"
#     source_id:      str          # func_id or endpoint path
#     description:    str          # natural language e.g. "output length ≤ input length"
#     property_type:  Literal[
#         "idempotent",            # f(f(x)) == f(x)
#         "monotonic",             # x > y → f(x) > f(y)
#         "bounded",               # len(f(x)) ≤ MAX
#         "roundtrip",             # decode(encode(x)) == x
#         "commutative",           # f(a,b) == f(b,a)
#         "invariant_output",      # certain inputs always produce same output shape
#     ]
#     hypothesis_strategy: str     # Hypothesis strategy code as string
#                                  # e.g. "st.text(min_size=1, max_size=100)"
#
# class InvariantExtractor:
#     """No required constructor args."""
#
#     async def extract_from_functions(
#         self,
#         specs: list[FunctionSpec],
#         semaphore: asyncio.Semaphore,
#     ) -> list[Invariant]:
#         """
#         Ollama (Semaphore(1), temp=0.1, format="json"):
#         For each FunctionSpec → infer 1-2 invariants based on:
#           - return_type + args → suggest property_type
#           - docstring → extract explicit contracts
#         """
#
#     async def extract_from_schemas(
#         self,
#         schemas: list[InferredSchema],
#         semaphore: asyncio.Semaphore,
#     ) -> list[Invariant]:
#         """
#         For each InferredSchema → infer 1-2 invariants:
#           - GET idempotent → repeated calls return same structure
#           - POST with same body → 201 then 422 (duplicate) or 201 (idempotent)
#           - response fields always present → invariant_output
#         """

### src/pbt/hypothesis_runner.py  (NEW)
# HypothesisRunner — executes Hypothesis tests from Invariant list
#
# class PBTResult(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     invariant_id:        str
#     examples_run:        int
#     counterexample_found: bool
#     counterexample:      str | None   # Hypothesis shrunk repr
#     counterexample_size: int          # len(repr(counterexample))
#     passed:              bool
#     duration_ms:         float
#     falsifying_example:  str | None   # Hypothesis @reproduce_failure decorator value
#
# class HypothesisRunner:
#     """No required constructor args."""
#
#     async def run(
#         self,
#         invariants: list[Invariant],
#         target_module: pathlib.Path,
#     ) -> list[PBTResult]:
#         """
#         For each invariant:
#           1. Generate pytest + @given() test from invariant.hypothesis_strategy
#           2. Write to tmp file
#           3. subprocess.run(["uv","run","pytest",tmp_file,
#              "--hypothesis-seed=42",   ← reproducible
#              "-x",                      ← stop on first failure
#              "--tb=short"], timeout=60)
#           4. Parse output:
#              - "Falsifying example:" → counterexample_found=True, extract repr
#              - "passed N examples"   → examples_run=N
#              - shrunk repr size      → counterexample_size=len(repr)
#          SecurityASTChecker before subprocess (reuse Sprint 6 pattern)
#         """
#
#     def _build_test(self, invariant: Invariant) -> str:
#         """
#         Returns complete pytest file content:
#
#         from hypothesis import given, settings
#         import hypothesis.strategies as st
#
#         @given({invariant.hypothesis_strategy})
#         @settings(max_examples=50, deadline=5000)
#         def test_{invariant.invariant_id}(value):
#             # property assertion derived from invariant.property_type
#             ...
#         """

### src/pbt/strategy_library.py  (NEW)
# Pre-built Hypothesis strategies mapped to property_type
# (avoids LLM needing to know Hypothesis API details)
#
# STRATEGY_MAP: dict[str, str] = {
#     "idempotent":       "st.text(min_size=0, max_size=50)",
#     "monotonic":        "st.integers(min_value=0, max_value=1000)",
#     "bounded":          "st.lists(st.integers(), max_size=100)",
#     "roundtrip":        "st.text(alphabet=st.characters(blacklist_categories=['Cs']))",
#     "commutative":      "st.tuples(st.integers(), st.integers())",
#     "invariant_output": "st.fixed_dictionaries({'key': st.text(min_size=1)})",
# }
# LLM receives STRATEGY_MAP keys only — picks appropriate key
# HypothesisRunner substitutes actual strategy string from map

### audit/sprint14/measure_sprint14.py
# Targets:
#   Functions: reuse FunctionSpecs from Sprint 6 (src/contractskill/*.py)
#   Endpoints: reuse InferredSchemas from Sprint 13 (realworld.habsida.net)
#
# Flow:
#   1. Load FunctionSpecs (ASTParser on Sprint 4–6 source files)
#   2. Load InferredSchemas (re-run SchemaInferrer or load cached)
#   3. InvariantExtractor.extract_from_functions() + extract_from_schemas()
#   4. HypothesisRunner.run(invariants) → list[PBTResult]
#   5. Compute gates:
#      properties_defined         = len(invariants)
#      hypothesis_examples_run    = sum(r.examples_run for r in results)
#      counterexamples_found      = sum(1 for r in results if r.counterexample_found)
#      shrunk_counterexample_size = min(r.counterexample_size
#                                       for r in results if r.counterexample_found)
#                                   or 0
#      pbt_pass_rate              = passed / len(results)
#   6. OTelTracer.span("pbt.invariant") per invariant
#   7. CryptoAuditTrail.append("pbt_result") per counterexample found

## OUTPUT audit/sprint14/sprint14_results.json:
{
  "properties_defined":          <int>,
  "hypothesis_examples_run":     <int>,
  "counterexamples_found":       <int>,
  "shrunk_counterexample_size":  <int>,
  "pbt_pass_rate":               <float>,
  "otel_spans_emitted":          <int>,
  "regression":                  <bool>,
  "sprint14_status":             "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS enforced
# - SecurityASTChecker before every subprocess (reuse Sprint 6)
# - subprocess timeout=60s per Hypothesis test
# - hypothesis settings: max_examples=50, deadline=5000ms
#   (conservative — 6GB VRAM constraint, avoid OOM from parallel examples)
# - --hypothesis-seed=42 for reproducibility
# - STRATEGY_MAP keys only passed to LLM (not raw Hypothesis API)
# - BFT: disabled (feature flag preserved)
# - uv add hypothesis before running
#   verify: uv run python -c "import hypothesis; print(hypothesis.__version__)"