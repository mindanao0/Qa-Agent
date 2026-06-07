## CONTEXT
PROJECT: D:\Code\qa-agent
BASELINE: Sprints 1–10 all PASS (verified), housekeeping clean
STACK: Python 3.13 + uv, Ollama qwen2.5-coder:7b-instruct-q4_K_M,
       LangGraph async, LanceDB, Playwright CDP, Pydantic V2

## OBJECTIVE
GOAL: Agent รับ URL เดียว → loop อัตโนมัติไม่หยุด →
      discover flows → generate tests → execute → self-heal →
      update SFG → loop ใหม่ จนกว่าจะถึง stop condition

## ACCEPTANCE GATE (sprint11)
  continuous_loop_cycles     ≥ 3     (agent วน loop ครบ 3 รอบโดยไม่ crash)
  new_states_per_cycle       ≥ 1     (แต่ละรอบเจอ state ใหม่อย่างน้อย 1)
  cumulative_tests_generated ≥ 15    (รวมทุก cycle)
  self_heal_rate             ≥ 0.50  (ถ้า test fail → repair สำเร็จ ≥ 50%)
  memory_mb_stable           = True  (RSS ไม่เพิ่มเกิน 200MB ระหว่าง cycle 1→3)
  REGRESSION if pass_rate    < 0.75

## ARCHITECTURE — New files:

### src/continuous/loop_controller.py
# ContinuousLoopController — LangGraph top-level graph ที่ wrap ทุก sub-graph
#
# class LoopState(TypedDict):
#     run_id:          str
#     start_url:       str
#     cycle:           int
#     max_cycles:      int          # stop condition (default 10, 0 = infinite)
#     stop_reason:     str | None   # "max_cycles" | "coverage_plateau" | "manual"
#     sfg_node_ids:    Annotated[list[str], operator.add]
#     tests_generated: Annotated[list[str], operator.add]
#     tests_passed:    Annotated[list[str], operator.add]
#     tests_failed:    Annotated[list[str], operator.add]
#     heal_attempts:   Annotated[list[str], operator.add]
#     memory_rss_mb:   float
#
# LangGraph nodes:
#   1. init_cycle      → increment cycle, snapshot RSS via psutil.Process().memory_info().rss
#   2. crawl           → SFGCrawler (CrawlerConfig from agent.yaml)
#                        → skip states already in SFGStore (dedup by node_id)
#   3. explore         → ExplorationPlanner sub-graph (Sprint 5)
#                        → generates TestHypotheses for NEW states only
#   4. generate        → HypothesisExecutor (Sprint 5) + PytestGenerator (Sprint 6)
#                        → Ollama Semaphore(1)
#   5. execute         → TestExecutor (Sprint 6) subprocess
#   6. heal            → RepairEngine (Sprint 4) on failed tests
#                        → max 1 repair attempt per test per cycle
#   7. store           → SFGStore.upsert_node/edge, EpisodicStore append
#   8. check_stop      → evaluate stop conditions:
#                        (a) cycle >= max_cycles
#                        (b) coverage_plateau: new_states == 0 for 2 consecutive cycles
#                        (c) memory_rss_mb > baseline_rss + 200
#                        → if any True: route to END
#                        → else: route back to init_cycle
#
# Edges:
#   START → init_cycle → crawl → explore → generate → execute
#         → heal → store → check_stop
#   check_stop → init_cycle (loop) | END (stop)

### src/continuous/memory_guard.py
# MemoryGuard — RSS monitor ที่ป้องกัน memory leak ระหว่าง loop
#
# class MemoryGuard:
#     """No required constructor args."""
#
#     def __init__(self, max_growth_mb: float = 200.0) -> None:
#         self._baseline_rss: float | None = None
#         self._max_growth   = max_growth_mb
#         self._proc         = psutil.Process()
#
#     def snapshot_baseline(self) -> float:
#         """Call once before cycle 1 starts. Returns baseline RSS in MB."""
#         self._baseline_rss = self._proc.memory_info().rss / 1024 / 1024
#         return self._baseline_rss
#
#     def current_rss_mb(self) -> float:
#         return self._proc.memory_info().rss / 1024 / 1024
#
#     def is_stable(self) -> bool:
#         """True if current RSS - baseline < max_growth_mb."""
#         if self._baseline_rss is None:
#             return True
#         return (self.current_rss_mb() - self._baseline_rss) < self._max_growth
#
#     def growth_mb(self) -> float:
#         if self._baseline_rss is None:
#             return 0.0
#         return self.current_rss_mb() - self._baseline_rss

### src/continuous/coverage_tracker.py
# CoveragePlateau detector
#
# class CoverageTracker:
#     """No required constructor args."""
#
#     def __init__(self, plateau_threshold: int = 2) -> None:
#         self._seen_nodes:     set[str] = set()
#         self._new_per_cycle:  list[int] = []
#         self._plateau_thresh  = plateau_threshold
#
#     def update(self, node_ids: list[str]) -> int:
#         """Returns count of NEW node_ids this cycle."""
#         new = [n for n in node_ids if n not in self._seen_nodes]
#         self._seen_nodes.update(node_ids)
#         self._new_per_cycle.append(len(new))
#         return len(new)
#
#     def is_plateau(self) -> bool:
#         """True if last N cycles all had 0 new states."""
#         if len(self._new_per_cycle) < self._plateau_thresh:
#             return False
#         return all(n == 0 for n in self._new_per_cycle[-self._plateau_thresh:])

### src/continuous/stop_conditions.py
# StopConditionEvaluator — pure function, no LLM
#
# class StopReason(str, Enum):
#     MAX_CYCLES       = "max_cycles"
#     COVERAGE_PLATEAU = "coverage_plateau"
#     MEMORY_LIMIT     = "memory_limit"
#     MANUAL           = "manual"
#
# def evaluate(
#     cycle:          int,
#     max_cycles:     int,
#     coverage:       CoverageTracker,
#     memory:         MemoryGuard,
# ) -> StopReason | None:
#     if max_cycles > 0 and cycle >= max_cycles:
#         return StopReason.MAX_CYCLES
#     if coverage.is_plateau():
#         return StopReason.COVERAGE_PLATEAU
#     if not memory.is_stable():
#         return StopReason.MEMORY_LIMIT
#     return None

### audit/sprint11/measure_sprint11.py
# Run ContinuousLoopController on https://demo.playwright.dev/todomvc/#/
# Config: max_cycles=3, max_pages=10, max_depth=3
#
# Measure after run completes:
#   continuous_loop_cycles     = controller.state["cycle"]
#   new_states_per_cycle       = min(coverage_tracker._new_per_cycle)
#   cumulative_tests_generated = len(controller.state["tests_generated"])
#   self_heal_rate             = healed / max(1, failed)
#   memory_mb_stable           = memory_guard.is_stable()
#   stop_reason                = StopReason value
#
# Also emit:
#   OTelTracer.span("continuous.cycle") per cycle
#   CryptoAuditTrail.append("cycle_complete", {...}) per cycle

## OUTPUT audit/sprint11/sprint11_results.json:
{
  "continuous_loop_cycles":     <int>,
  "new_states_per_cycle":       <int>,    // minimum across cycles
  "cumulative_tests_generated": <int>,
  "self_heal_rate":             <float>,
  "memory_mb_stable":           <bool>,
  "stop_reason":                <str>,
  "regression":                 <bool>,
  "sprint11_status":            "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS enforced
# - EpisodicStore + ContractSkill + SFGStore = append-only
# - BFT: disabled (feature flag preserved)
# - psutil for RSS (uv add psutil if not in pyproject)
# - LangGraph checkpointing: AsyncSqliteSaver
#   path: pathlib.Path("audit/sprint11/checkpoints.db")
# - MemoryGuard.snapshot_baseline() called ONCE before cycle 1
# - CoverageTracker.update() called ONCE per cycle after crawl
# - max 1 repair attempt per test per cycle (prevent OOM loop)
# - Browser: single persistent BrowserContext across cycles
#   (ใช้ context เดิม ไม่ new_context() ทุก cycle — ประหยัด memory)