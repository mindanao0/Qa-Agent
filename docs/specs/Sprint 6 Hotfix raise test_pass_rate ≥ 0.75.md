# Sprint 6 Hotfix: fix failing generated tests to clear regression flag
# Project: D:\Code\qa-agent
# Current: test_pass_rate=0.706 (5 tests failing out of 17)
# Target:  test_pass_rate≥0.75 → regression=false

## STEP 1 — identify which tests are failing

cd D:\Code\qa-agent
uv run python audit\sprint6\measure_sprint6.py --live --verbose

# --verbose flag must print per-test result:
# PASS test_sfg_upsert_node | FAIL test_grounder_empty_tree | etc.
# If --verbose not implemented, add this to measure_sprint6.py:
#   for result in test_results:
#       print(f"{'PASS' if result.passed else 'FAIL'} {result.test_id} | {result.error or ''}")

## STEP 2 — common failure patterns in generated tests, fix in generator.py

# Pattern A: fixture/import errors
#   GeneratedTest imports module with wrong relative path
#   Fix in PytestGenerator system prompt — add rule:
#   "Always use absolute imports: from src.contractskill.sfg import SFGStore"
#   "Never use relative imports (from .sfg import ...)"

# Pattern B: async test missing @pytest.mark.asyncio
#   Fix in CodeJudge — add to check 1 (has_assert):
#   if "async def test_" in test_code and "@pytest.mark.asyncio" not in test_code:
#       grade = "needs_revision"
#       feedback = "async test requires @pytest.mark.asyncio decorator"

# Pattern C: SFGStore requires pathlib.Path(tmp) not string
#   Fix in PytestGenerator prompt:
#   "SFGStore constructor requires pathlib.Path, use tmp_path fixture:
#    def test_x(tmp_path): store = SFGStore(tmp_path / 'test.db')"

# Pattern D: metamorphic test asserts wrong direction
#   CodeJudge check 4 must verify relation string is falsifiable
#   If relation contains "always" or "never" without bound → needs_revision

## STEP 3 — after fixing generator.py + judge.py, regenerate only failed tests

# In measure_sprint6.py add --rerun-failed flag:
#   Load sprint6_results.json → find test_pass_rate < 1.0
#   Re-run ASTParser only on modules whose tests failed
#   Re-run PytestGenerator + CodeJudge + TestExecutor on those specs only
#   Merge results: passed_before + passed_now / total

## STEP 4 — rerun gate

uv run python audit\sprint6\measure_sprint6.py --live

## TARGET sprint6_results.json:
{
  "ast_functions_parsed": 17,
  "tests_generated":      17,
  "test_pass_rate":       ≥ 0.75,
  "metamorphic_pairs":    ≥ 3,
  "regression":           false,
  "sprint6_status":       "PASS"
}

## RULES (carry-forward):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - Judge: 4 checks only (patch check 1 for async, do NOT add 5th check)
# - SecurityASTChecker before subprocess
# - subprocess timeout=30s