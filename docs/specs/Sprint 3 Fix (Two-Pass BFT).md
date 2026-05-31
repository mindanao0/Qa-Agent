Sprint 3 REGRESSION: consensus_reached_rate=0.20, root cause = BFT votes on 
Python code which diverges at temperature > 0.0. Fix: move BFT consensus to 
TestPlan (structured JSON), then single-pass code generation from winning plan.

Read FIRST:
  1. audit/phase0/AUDIT_REPORT.md     §3 (generator ⚠️ "two-pass workaround")
  2. src/core/bft_consensus.py        (BFTVoter — preserve, change input)
  3. src/agents/bft_generator.py      (bft_generator_node — rewrite logic)
  4. src/llm/schemas.py               (TestPlan, GeneratedCode — both exist)
  5. src/agents/generator.py          (existing two-pass path — re-use)

================================================================================
WHAT CHANGED — ARCHITECTURE ONLY, NOT BFT MATH
================================================================================

BEFORE (broken):
  3 generators → 3 Python code strings → AST normalize → vote
  Problem: Python code diverges by temperature → rarely 2/3 agree

AFTER (fix):
  3 generators → 3 TestPlan JSON objects → JSON field vote → 1 winning plan
  1 code generator (T=0.0) → Python code from winning plan
  LLM-as-Judge validates the final code

BFTVoter logic changes:
  - Input: list[TestPlan | None] instead of list[str | None]
  - Comparison: vote on plan.steps serialized as canonical JSON
    (sort keys, strip whitespace → deterministic string)
  - Quorum: same Q ≥ 2/3 = 2 votes
  - Output: winning TestPlan object (not code string)

ASTNormalizer: UNCHANGED — still used in a new role:
  After code generation (Pass 2), normalize the single output as a 
  security check only (detect protected name remapping).
  BFTVoter no longer receives code strings.

================================================================================
SURGICAL CHANGES — 3 files only
================================================================================

CHANGE 1: src/core/bft_consensus.py
  Add new class PlanVoter alongside BFTVoter:

    class PlanVoter:
        def vote(self, candidates: list[TestPlan | None]) -> TestPlan | None:
            valid = [(p, self._canonical(p)) for p in candidates if p is not None]
            if not valid:
                return None
            groups: dict[str, list[TestPlan]] = {}
            for plan, canon in valid:
                groups.setdefault(canon, []).append(plan)
            best = max(groups, key=lambda k: len(groups[k]))
            if len(groups[best]) >= 2:
                return groups[best][0]
            return None

        def _canonical(self, plan: TestPlan) -> str:
            import json
            return json.dumps(
                plan.model_dump(exclude={"model_config"}),
                sort_keys=True, ensure_ascii=False
            )

  Keep BFTVoter unchanged (it still exists for security check role).

CHANGE 2: src/agents/bft_generator.py
  Rewrite bft_generator_node logic to two passes:

    async def bft_generator_node(state: AgentState) -> dict:
        prompt_for_plan = _build_plan_prompt(state)  # requirement → plan
        semaphore = _get_llm_semaphore()

        # PASS 1: BFT on TestPlan (structured JSON, low format failure rate)
        plans: list[TestPlan | None] = []
        for config in BFT_CONFIGS:
            plan = await run_single_plan_generator(config, prompt_for_plan, semaphore)
            plans.append(plan)

        winning_plan = PlanVoter().vote(plans)

        if winning_plan is None:
            return {
                "bft_status": "CONSENSUS_FAILED",
                "bft_node_results": [...],
                "generated_code": None,
            }

        # PASS 2: Single code generator from winning plan (T=0.0 only)
        prompt_for_code = _build_code_prompt(winning_plan, state)
        code_result = await run_single_generator(BFT_CONFIGS[0], prompt_for_code, semaphore)

        if code_result is None:
            return {"bft_status": "CODE_GEN_FAILED", "generated_code": None, ...}

        # Security check: ASTNormalizer detects any protected name remapping
        normalizer = ASTNormalizer()
        if normalizer.normalize(code_result) is None:
            return {"bft_status": "SECURITY_HALT", "generated_code": None, ...}

        # Judge validates the final code
        verdict = await judge_client.evaluate(code_result, state["requirement"], 
                                               state["domain"])
        if not verdict.approved:
            return {"bft_status": "JUDGE_REJECTED", "generated_code": None, ...}

        return {
            "bft_status": "CONSENSUS_REACHED",
            "bft_node_results": [...],
            "generated_code": code_result,
        }

  Add run_single_plan_generator() — same as run_single_generator() but 
  response_model=TestPlan instead of GeneratedCode.

  Add _build_plan_prompt(state) and _build_code_prompt(plan, state):
    - Find existing prompt builders in generator.py (the two-pass workaround)
    - Extract and re-use them here — do NOT write new prompts from scratch
    - They already exist per AUDIT_REPORT §3

CHANGE 3: Add bft_status="CODE_GEN_FAILED" and "SECURITY_HALT" to 
  AgentState and conditional edge routing in graph.py:
    "CODE_GEN_FAILED" → reporter_node
    "SECURITY_HALT"   → reporter_node

================================================================================
TESTS — update only (no new files)
================================================================================
In tests/test_bft_consensus.py:
  Add 3 new tests for PlanVoter:
  [ ] test_plan_voter_quorum_two_identical_plans
  [ ] test_plan_voter_no_quorum_all_different
  [ ] test_plan_canonical_is_deterministic (same plan twice = same JSON)

  Existing 8 BFTVoter + ASTNormalizer tests: DO NOT CHANGE

================================================================================
MEASUREMENT
================================================================================
Re-run audit/sprint3/measure_sprint3.py --live

Expected improvement:
  consensus_reached_rate: 0.20 → ≥ 0.80  (TestPlan JSON easier to agree on)
  avg_bft_latency_ms: lower per case (plan generation faster than code gen)

Print:
  "SPRINT 3 TWO-PASS FIX COMPLETE"
  "consensus_reached_rate: X | avg_bft_latency: Xms | pass_rate: X"