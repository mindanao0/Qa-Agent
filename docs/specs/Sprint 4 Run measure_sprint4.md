# Sprint 4: Run measure_sprint4.py --live + fix acceptance gate gaps
# Project: D:\Code\qa-agent
# Current state:
#   sfg_nodes=5 (gate requires ≥10), edges=186 ✅
#   contract_skills_compiled=? (gate requires ≥4)
#   pass_rate=? (gate requires ≥0.75, baseline 0.60)

## STEP 1 — Run the measurement first (do this before any code changes)

cd D:\Code\qa-agent
uv run python audit\sprint4\measure_sprint4.py --live

## STEP 2 — If sfg_nodes < 10, expand CrawlerConfig in the smoke/measure script

# In audit\sprint4\measure_sprint4.py, find the CrawlerConfig call and update:
# BEFORE: CrawlerConfig(max_pages=5, max_depth=3, max_time_minutes=5)
# AFTER:  CrawlerConfig(max_pages=20, max_depth=4, max_time_minutes=10)
# Then re-run STEP 1.

## STEP 3 — Check measure_sprint4.py output against all 5 gate conditions:

# Gate checklist (print these values explicitly):
#   after_sprint4_pass_rate   ≥ 0.75  ← compare vs 0.60 baseline
#   contract_skills_compiled  ≥ 4
#   sfg_nodes_discovered      ≥ 10
#   skills_used_in_generation ≥ 2
#   REGRESSION if             < 0.55

## STEP 4 — If contract_skills_compiled < 4, run the compiler manually:

# In measure_sprint4.py or a separate script:
from src.contractskill.compiler import ContractSkillCompiler
from src.contractskill.sfg import SFGStore
import pathlib, asyncio

async def compile_skills():
    store = SFGStore(pathlib.Path("audit/sprint4/smoke_sfg.db"))
    compiler = ContractSkillCompiler(store)
    skills = await compiler.compile_all()
    print(f"contract_skills_compiled={len(skills)}")
    for s in skills:
        print(f"  skill_id={s.skill_id}  goal={s.goal}")

asyncio.run(compile_skills())

## STEP 5 — If pass_rate still ≤ 0.60 after skills are compiled,
# check if AdaptiveRouter is loading skills from ContractSkillCompiler output.
# The router must call:
#   store = ContractSkillStore(pathlib.Path("audit/sprint4/smoke_sfg.db"))
#   skills = store.list_skills()
#   if skills: inject top-matching skill steps into generation prompt

## STEP 6 — Report final sprint4_results.json values:

{
  "after_sprint4_pass_rate":   <float>,
  "contract_skills_compiled":  <int>,
  "sfg_nodes_discovered":      <int>,
  "skills_used_in_generation": <int>,
  "regression":                <bool>,
  "sprint4_status":            "PASS" | "FAIL"
}

## Rules (carry-forward):
# - asyncio.Semaphore(1) on all Ollama calls
# - pathlib.Path everywhere (no os.path string concat)
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP/AOMExtractor only (no page.accessibility)
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS enforced in execution layer