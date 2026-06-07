## CONTEXT
PROJECT: D:\Code\qa-agent
BASELINE: Sprint 11 PASS — ContinuousLoopController working,
          memory_growth=41MB stable, checkpointing via AsyncSqliteSaver
STACK: Python 3.13 + uv, Ollama, LangGraph async, Playwright CDP

## OBJECTIVE
GOAL: ทดสอบ race condition จริงบน real stateful backend
      (ไม่ใช่ localStorage-only app) โดยใช้ RaceConditionSwarm
      ที่แก้ไขแล้ว (semantic_hash_comparison) จาก Sprint 10

## TARGET
Primary:  https://conduit.realworld.how  (RealWorld app — real REST API + DB)
Fallback: https://jsonplaceholder.typicode.com (mock, expect race=0)

## ACCEPTANCE GATE (sprint12)
  race_scenarios_tested       ≥ 5
  real_backend_confirmed      = True   (conduit API responds with real data)
  race_conditions_detected    ≥ 1      (real conflict on shared backend state)
  interleaving_patterns_found ≥ 1
  semantic_hash_method        = True   (CDP nodeId excluded from hash)
  otel_spans_emitted          ≥ 8
  REGRESSION if pass_rate     < 0.75

## ARCHITECTURE — New + Modified files:

### src/race/backend_probe.py  (NEW)
# BackendProbe — verify target has real shared backend before running swarm
#
# class BackendProbe:
#     """No required constructor args."""
#
#     async def probe(self, url: str, page: Page) -> BackendProbeResult:
#         """
#         1. page.goto(url)
#         2. Capture ≥1 API call via page.on("request") that returns JSON
#         3. Check response has real data (not empty array/object)
#         4. Return BackendProbeResult
#         """
#
# class BackendProbeResult(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     url:              str
#     has_real_backend: bool
#     api_endpoints:    list[str]   # captured during probe
#     probe_note:       str | None

### src/race/swarm.py  (MODIFY — add conduit scenarios)
# Add RealWorldScenarios for conduit.realworld.how:
#
# CONDUIT_SCENARIOS = [
#   RaceScenario(
#     scenario_id="s1_concurrent_follow",
#     description="2 agents follow same author simultaneously",
#     agents=2, action="POST /api/profiles/{username}/follow",
#     target_url="https://conduit.realworld.how",
#     overlap_ms=50, expected_safe=False,
#     # conflict = duplicate follow entries or 409 response
#   ),
#   RaceScenario(
#     scenario_id="s2_concurrent_favorite",
#     description="3 agents favorite same article simultaneously",
#     agents=3, action="POST /api/articles/{slug}/favorite",
#     target_url="https://conduit.realworld.how",
#     overlap_ms=50, expected_safe=False,
#   ),
#   RaceScenario(
#     scenario_id="s3_read_write_article",
#     description="1 agent edits article while 2 agents read it",
#     agents=3, action="GET+PUT /api/articles/{slug}",
#     target_url="https://conduit.realworld.how",
#     overlap_ms=100, expected_safe=True,
#   ),
#   RaceScenario(
#     scenario_id="s4_concurrent_comment",
#     description="3 agents post comment on same article simultaneously",
#     agents=3, action="POST /api/articles/{slug}/comments",
#     target_url="https://conduit.realworld.how",
#     overlap_ms=50, expected_safe=True,
#   ),
#   RaceScenario(
#     scenario_id="s5_concurrent_register",
#     description="2 agents register with same username simultaneously",
#     agents=2, action="POST /api/users",
#     target_url="https://conduit.realworld.how",
#     overlap_ms=50, expected_safe=False,
#     # conflict = one must get 422 Unprocessable Entity
#   ),
# ]
#
# conflict_found logic (keep semantic_hash from Sprint 10 fix):
#   conflict_found = (
#       any(status not in {200, 201, 204} for status in agent_statuses)
#       or len(set(semantic_hashes)) > 1
#       or bool(errors)
#   )
# NOTE: conduit may return 422 for duplicate register → conflict_found=True = correct

### src/race/interleaving_recorder.py  (NEW)
# InterleavingRecorder — records per-agent action timestamps + response
#
# class AgentEvent(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     agent_id:     str
#     action:       str
#     started_at:   float    # time.monotonic() ms
#     ended_at:     float
#     status_code:  int | None
#     response_hash: str     # sha256[:8] of response body
#
# class InterleavingRecorder:
#     def record(self, event: AgentEvent) -> None: ...
#     def pattern(self) -> str:
#         """e.g. 'A-start, B-start, A-end(201), B-end(422)'"""
#     def to_dict(self) -> dict: ...

### audit/sprint12/measure_sprint12.py
#
# Flow:
#   1. BackendProbe.probe("https://conduit.realworld.how") → confirm real backend
#   2. If has_real_backend=False → use fallback + set race_conditions_detected=0
#   3. RaceConditionSwarm.run() × 5 CONDUIT_SCENARIOS
#   4. ConflictDetector.analyze(results)
#   5. InterleavingRecorder.pattern() per scenario
#   6. OTelTracer.span("race.scenario") per scenario
#   7. CryptoAuditTrail.append("race_result", {...}) per scenario

## OUTPUT audit/sprint12/sprint12_results.json:
{
  "race_scenarios_tested":       <int>,
  "real_backend_confirmed":      <bool>,
  "race_conditions_detected":    <int>,
  "interleaving_patterns_found": <int>,
  "semantic_hash_method":        <bool>,
  "otel_spans_emitted":          <int>,
  "regression":                  <bool>,
  "sprint12_status":             "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - BLOCKED_ACTION_PATTERNS: delete/remove/transfer/payment/password
#   → skip scenario if action matches (conduit delete article = skip)
# - asyncio.Barrier for swarm sync (NOT asyncio.sleep)
# - SynchronizationDriftError if timing drift > overlap_ms * 2
# - semantic_hash: exclude CDP nodeId/backendDOMNodeId/childIds
# - Each agent: isolated BrowserContext (never shared)
# - Timeout per agent action: 5s
# - OTelTracer + CryptoAuditTrail on every scenario
# - BFT: disabled (feature flag preserved)
# - InterleavingRecorder: record EVERY agent event (start + end)
# - If conduit unreachable → document in sprint12_results.json,
#   use jsonplaceholder fallback, expect race_conditions_detected=0