Read these files FIRST in this exact order, do not skip:
  1. audit\phase0\AUDIT_REPORT.md         (focus: sections 3, 5, 8, TD-1)
  2. audit\phase0\baseline_metrics.json   (the numbers we beat)
  3. audit\phase0\sprint1_targets.yaml    (the target numbers)
  4. src\llm\structured.py                (current 3-stage repair, line 269-312)
  5. src\llm\adapter.py                   (Ollama client, temperature handling)
  6. src\core\agent.py + src\graph\workflow.py (where structured output is called)
  7. CLAUDE.md and docs\specs\SPEC_CORE.md

================================================================================
TASK: Replace the 3-stage repair pipeline with Instructor + Ollama native JSON
mode. This is Sprint 1, Day 1-2 — the highest-ROI fix in the entire roadmap.
================================================================================

HARD CONSTRAINTS (non-negotiable)
================================================================================
- Windows 11 native, Python 3.13 + uv. Use pathlib.Path everywhere.
- Ollama at http://localhost:11434, model qwen2.5-coder:7b-instruct-q4_K_M.
- Async-first: every I/O function uses asyncio. No blocking calls in hot paths.
- Pydantic V2 only. Use model_config = ConfigDict(...) syntax, NOT V1 Config class.
- Temperature 0.0 (deterministic) for structured generation. The baseline used 0.1 
  which violated the spec — fix this in adapter.py while you are there.
- The existing asyncio.Semaphore(1) on Ollama calls must remain — never remove it.
- DO NOT delete src\llm\structured.py. Leave the 3-stage repair as a fallback path 
  callable via a feature flag for A/B comparison during Sprint 1.

================================================================================
STEP 1 — INSTALL DEPENDENCIES
================================================================================
Add to pyproject.toml under [project.dependencies]:
  - instructor>=1.6.0          (the structured output library)
  - tenacity>=8.5.0             (Instructor's retry dep, pin explicitly)

Run: uv add instructor tenacity

Verify Pydantic V2 is already >=2.6.0. If not, bump it.

================================================================================
STEP 2 — CREATE src\llm\instructor_client.py
================================================================================
This is the NEW preferred entry point for structured LLM output. Architecture:

  class InstructorClient:
    - __init__(base_url, model, semaphore, max_retries=3)
    - async create_structured(prompt, response_model, temperature=0.0) -> T
        Uses instructor.from_openai() pointed at Ollama's OpenAI-compatible 
        endpoint http://localhost:11434/v1
        Passes response_format={"type": "json_object"} via Ollama's native 
        JSON mode (separate from Instructor's schema enforcement — both layered).
        Wraps the call in the shared LLM_SEMAPHORE.
        Returns a validated Pydantic V2 model instance — NEVER a dict, NEVER a string.

  class GenerationConfig (Pydantic V2 BaseModel):
    - temperature: float = 0.0
    - top_p: float = 1.0
    - num_predict: int = 2048
    - num_ctx: int = 8192
    - model_config = ConfigDict(frozen=True, extra="forbid")

Behavior contract:
  - On ValidationError: Instructor retries up to max_retries, each retry feeds 
    the previous error back to the LLM as a correction prompt.
  - On JSONDecodeError after Ollama JSON mode (should be near zero): same retry.
  - On exhausted retries: raise StructuredGenerationError(prompt_hash, last_error)
    — do NOT silently fall back to 3-stage repair here. Let the caller decide.
  - Log at INFO level: model, prompt token count, output token count, latency_ms, 
    retry_count, validation_passed.

================================================================================
STEP 3 — UPDATE src\llm\adapter.py
================================================================================
Two surgical changes:

3a. Add a temperature parameter to the Ollama call signature. Currently it is 
    likely hardcoded to 0.1. Make it explicit and default it to 0.0 for any 
    structured-output path. Keep 0.7 as the default for free-form generation 
    paths only.

3b. Add a new method ollama_v1_url() that returns base_url.rstrip("/") + "/v1"
    so the Instructor client can use Ollama's OpenAI-compatible endpoint.

Do not break the existing API — add the new parameter as keyword-only with a 
default value. Existing callers continue to work.

================================================================================
STEP 4 — DEFINE Pydantic V2 OUTPUT SCHEMAS
================================================================================
Create src\llm\schemas.py with the canonical output models. Look at the existing 
structured.py to discover what the agent currently expects, then formalize:

  class PlanStep(BaseModel):
      step_number: int
      description: str
      action_type: Literal["navigate", "click", "fill", "wait", "assert", "select"]
      target_hint: str
      expected_outcome: str
      model_config = ConfigDict(extra="forbid", frozen=True)

  class TestPlan(BaseModel):
      requirement_summary: str
      steps: list[PlanStep] = Field(min_length=1, max_length=30)
      preconditions: list[str] = Field(default_factory=list)
      postconditions: list[str] = Field(default_factory=list)
      model_config = ConfigDict(extra="forbid", frozen=True)

  class GeneratedCode(BaseModel):
      python_code: str = Field(min_length=20)
      imports: list[str]
      uses_async: bool
      estimated_runtime_ms: int = Field(ge=0, le=300_000)
      model_config = ConfigDict(extra="forbid", frozen=True)

  class HealAction(BaseModel):
      action_type: Literal["selector_replace", "wait_added", "action_changed"]
      original_locator: str
      new_locator: str
      reasoning: str = Field(max_length=500)
      confidence: float = Field(ge=0.0, le=1.0)
      model_config = ConfigDict(extra="forbid", frozen=True)

Mine the actual fields from src\llm\structured.py and src\core\* — these are 
illustrative. Use the REAL field names the codebase already expects so callers 
do not have to change.

================================================================================
STEP 5 — SWITCH CALLERS WITH A FEATURE FLAG
================================================================================
In config\agent.yaml add:
  llm:
    structured_output_engine: "instructor"   # "instructor" | "legacy_repair"
    
In every call site that currently uses structured.py (find them all — there 
should be 2-4 places: planner_node, generator_node, healer_node, maybe 
finetune dataset gen), wrap the call:

  if config.llm.structured_output_engine == "instructor":
      result = await instructor_client.create_structured(prompt, TestPlan)
  else:
      result = await legacy_structured_call(prompt)  # existing path

Default the config to "instructor" so the new path is live, but keep the escape 
hatch for one sprint.

================================================================================
STEP 6 — CREATE tests\test_instructor_client.py
================================================================================
Live integration tests — they hit real Ollama. Mark with @pytest.mark.integration:

  test_simple_plan_generation_returns_valid_pydantic_model
    Given a short requirement, assert returns TestPlan instance (not dict).

  test_invalid_schema_triggers_retry_then_succeeds
    Use a deliberately tricky prompt, monkeypatch Instructor's max_retries=2,
    assert retry_count > 0 in logs and final result is valid.

  test_semaphore_blocks_concurrent_calls
    Launch 3 create_structured() coroutines with asyncio.gather, instrument 
    timing, assert serial execution (gap >= shortest call duration).

  test_temperature_zero_is_deterministic
    Same prompt twice, assert byte-identical output (or near-identical — Ollama 
    has tiny non-determinism even at T=0; allow Levenshtein distance ≤ 5).

  test_legacy_path_still_works_when_flag_set
    Set config flag to "legacy_repair", assert old path still produces output.

================================================================================
STEP 7 — A/B MEASUREMENT SCRIPT
================================================================================
Create audit\phase0\measure_sprint1_day2.py — runs the golden dataset twice 
(once per engine) and writes audit\phase0\sprint1_day2_results.json:

  Schema:
  {
    "measured_at_iso": "...",
    "instructor": {
      "json_parse_failure_rate": 0.0,
      "first_run_pass_rate": 0.0,
      "avg_generation_latency_ms": 0,
      "avg_retry_count": 0.0,
      "validation_errors_total": 0
    },
    "legacy_repair": {
      "json_parse_failure_rate": 0.0,
      "first_run_pass_rate": 0.0,
      "avg_generation_latency_ms": 0,
      "repair_stage_hits": {"stage1": 0, "stage2": 0, "stage3": 0}
    },
    "delta": {
      "json_parse_failure_rate_reduction_pct": 0.0,
      "first_run_pass_rate_improvement_pct": 0.0,
      "latency_change_pct": 0.0
    },
    "verdict": "PASS" | "FAIL" | "REGRESSION"
  }

verdict = "PASS" iff:
  instructor.json_parse_failure_rate <= 0.05  AND
  instructor.first_run_pass_rate >= 0.40       AND
  instructor.avg_generation_latency_ms <= 1.20 * legacy.avg_generation_latency_ms

verdict = "REGRESSION" iff first_run_pass_rate dropped below baseline 0.286.

Run on the pilot golden dataset (2 passing + 5 failing) first to validate the 
measurement script itself works end-to-end. Background the resume on the larger 
build_golden_dataset.py job in parallel.

================================================================================
STEP 8 — UPDATE DOCUMENTATION
================================================================================
8a. CLAUDE.md — add a one-line entry under the LLM section pointing to 
    src\llm\instructor_client.py as the preferred entry point.

8b. docs\specs\SPEC_CORE.md — update the "Structured Output" section to 
    document Instructor as primary, legacy 3-stage repair as fallback.

8c. Add audit\phase0\SPRINT1_DAY2_LOG.md — a short post-mortem with:
    - Files changed (list)
    - Numbers before / after from sprint1_day2_results.json
    - Whether verdict was PASS / FAIL / REGRESSION
    - Any unexpected issues (Ollama JSON mode quirks, schema mismatches, etc.)
    - Recommendation for Day 3-5 (DOM Pruner) — proceed or pivot?

================================================================================
ROLLBACK PLAN (mandatory section in SPRINT1_DAY2_LOG.md)
================================================================================
If verdict == "REGRESSION":
  Set config\agent.yaml structured_output_engine back to "legacy_repair"
  Do NOT proceed to Day 3-5
  Open a tech-debt entry TD-11 in AUDIT_REPORT.md explaining the failure mode
  Stop and surface the issue

================================================================================
OUTPUT REQUIREMENTS
================================================================================
- Complete, runnable code. No placeholders. No "# existing code …".
- Specify exact file path as a header comment at top of every file you create.
- Use the existing logger pattern (do not introduce a new logging setup).
- After all changes, run these commands and capture output:
    uv run pytest tests\test_instructor_client.py -v -m integration
    uv run python audit\phase0\measure_sprint1_day2.py
    type audit\phase0\sprint1_day2_results.json
- Print final summary line:
    "SPRINT 1 DAY 1-2 COMPLETE — verdict: <PASS|FAIL|REGRESSION>"
    "Next: Day 3-5 DOM Pruner — see audit\phase0\SPRINT1_DAY2_LOG.md"