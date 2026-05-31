# Sprint 5: Exploratory Mode
# Project: D:\Code\qa-agent
# Baseline: pass_rate=1.00, sfg_nodes=15, contract_skills=4
# Goal: Agent receives a URL → autonomously discovers flows → generates test hypotheses

## ACCEPTANCE GATE (sprint5)
#   exploration_coverage    ≥ 0.70  (unique states / total reachable states estimated)
#   hypotheses_generated    ≥ 5
#   hypothesis_pass_rate    ≥ 0.70
#   skills_reused           ≥ 2     (from Sprint 4 ContractSkill store)
#   REGRESSION if pass_rate < 0.75

## ARCHITECTURE — New files to create:

### src/explorer/planner.py
# ExplorationPlanner — LangGraph sub-graph
# Input:  start_url: str, sfg: SFGStore, skills: list[ContractSkill]
# Output: list[TestHypothesis]
#
# Nodes (async):
#   1. crawl_node        → SFGCrawler(config from agent.yaml exploration section)
#   2. cluster_node      → cluster SFG nodes by ax_summary similarity (cosine, lancedb)
#   3. gap_analysis_node → compare clusters vs existing ContractSkills → find uncovered flows
#   4. hypothesis_node   → Ollama qwen2.5-coder:7b (Semaphore(1), temp=0.1, format="json")
#                          → generate TestHypothesis list from gap analysis
#   5. judge_node        → 4-check judge (existing Judge class, reuse from Sprint 3)
#
# LangGraph edges:
#   START → crawl → cluster → gap_analysis → hypothesis → judge → END
#   judge routes back to hypothesis if grade="needs_revision" (max 2 retries)

### src/explorer/hypothesis.py
# Pydantic V2 schemas
#
# class TestHypothesis(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     hypothesis_id: str          # sha256[:12] of goal+url
#     goal:          str          # natural language: "User can add and complete a todo"
#     start_url:     str
#     preconditions: list[str]    # e.g. ["page loaded", "no existing todos"]
#     steps:         list[str]    # semantic action steps (not Playwright code yet)
#     expected_outcome: str
#     source_skill_id:  str | None  # if derived from existing ContractSkill
#     confidence:       float        # 0.0–1.0 from judge
#
# class ExplorationReport(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     run_id:               str
#     start_url:            str
#     sfg_nodes_found:      int
#     hypotheses:           list[TestHypothesis]
#     exploration_coverage: float
#     skills_reused:        int
#     generated_at:         float

### src/explorer/executor.py
# HypothesisExecutor — converts TestHypothesis → Playwright script → executes → pass/fail
# Reuses: ContractSkillCompiler, RepairEngine, AdaptiveRouter from Sprint 4
#
# async def execute(hypothesis: TestHypothesis, page: Page) -> HypothesisResult
# Rules:
#   - BLOCKED_ACTION_PATTERNS enforced before any action
#   - asyncio.Semaphore(1) on Ollama generation call
#   - RepairEngine on first failure (SelReplace/PreInsert/ArgCorrect)
#   - max 1 repair attempt per hypothesis (prevent OOM loops)
#   - CDP/AOMExtractor only (no page.accessibility)

### audit/sprint5/measure_sprint5.py
# Gate measurement script
# 1. Run ExplorationPlanner on https://demo.playwright.dev/todomvc/#/
# 2. Run HypothesisExecutor on each hypothesis
# 3. Write audit/sprint5/sprint5_results.json:
{
  "exploration_coverage":   <float>,
  "hypotheses_generated":   <int>,
  "hypothesis_pass_rate":   <float>,
  "skills_reused":          <int>,
  "regression":             <bool>,
  "sprint5_status":         "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP/AOMExtractor (no page.accessibility)
# - Judge: 4 checks only (no expansion for 7B)
# - BLOCKED_ACTION_PATTERNS: delete/remove/transfer/payment/password
# - EpisodicStore + ContractSkill tables = append-only
# - BFT: disabled (feature flag), code preserved
# - SFGCrawler limits from config/agent.yaml exploration section
# - ContractSkill store from Sprint 4 = input to gap_analysis_node