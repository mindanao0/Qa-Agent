BFT consensus consistently fails (0.20) because temperature>0.0 generators 
fail Pydantic schema validation. This is a hardware constraint, not a bug.
Execute the following deactivation + strengthening steps, then close Sprint 3.

Read FIRST:
  1. config/agent.yaml       (llm.bft section)
  2. src/llm/judge_client.py (JudgeClient — strengthen)
  3. audit/sprint3/sprint3_results.json

================================================================================
STEP 1 — Deactivate BFT (2 minutes)
================================================================================
In config/agent.yaml:
  llm:
    bft:
      enabled: false
      deferred_reason: "7B model + max_retries=1 → T>0 generators fail schema"
      reactivate_when: "model upgrade to ≥14B or VRAM ≥ 12GB"
    structured_output_engine: "instructor"

In src/agents/bft_generator.py, bft_generator_node():
  Add at top:
    if not config.llm.bft.enabled:
        # Fallback to single generator (T=0.0) + Judge
        return await _single_generator_with_judge(state)
  
  _single_generator_with_judge() already exists as the MED tier path —
  just call it directly.

Verify: run one generation, confirm bft_status="BFT_DISABLED_FALLBACK" 
in AgentState (add this new status).

================================================================================
STEP 2 — Strengthen Judge (30 minutes)
================================================================================
The Judge is now the PRIMARY anti-hallucination gate (not just post-BFT).
Upgrade JudgeClient.evaluate() prompt to add 2 more checks:

Current checks (keep all 4):
  1. Does the test verify the requirement?
  2. Are there false-positive assertions?
  3. Hallucinated selectors?
  4. Has at least one expect()?

Add check 5: Locator cross-reference
  "Does any locator in the code look fabricated?
   Signs: CSS chains > 4 levels deep, IDs with random suffixes (e.g., #btn-1a2b3c),
   aria-labels that don't match common UI patterns.
   Mark as suspicious if found."

Add check 6: Async completeness
  "Does every await have a matching async def wrapper?
   Does the test import async_playwright or use pytest-asyncio markers?
   Reject if async/await is mismatched."

Keep: judge fails OPEN (returns approved=True on Ollama error).
Keep: max_retries=1 for judge (same as before).

================================================================================
STEP 3 — Write Sprint 3 Final Log
================================================================================
audit/sprint3/SPRINT3_FINAL_LOG.md:

  # Sprint 3 — Final Report
  **Verdict: CLOSED (BFT Deactivated — Hardware Constraint)**

  ## What Was Built (Keep — Not Wasted)
  - src/core/bft_consensus.py: ASTNormalizer, PlanVoter, BFTVoter ← reusable when hardware upgrades
  - src/routing/adaptive_router.py: ConfidenceTier, url_test_cache ← ACTIVE, improves Sprint 4+
  - src/llm/judge_client.py: JudgeClient ← ACTIVE, now primary gate
  - src/agents/bft_generator.py: bft_generator_node with fallback ← ACTIVE via fallback

  ## What Didn't Work and Why
  BFT requires N independent working generators. With 7B model + strict Pydantic V2 
  + max_retries=1, only T=0.0 produces reliable valid output. T=0.3 and T=0.7 
  consistently fail schema validation → only 1 valid candidate → no quorum possible.
  Two-pass approach (vote on TestPlan JSON) had same root cause.

  ## Active Components After Sprint 3
  - Adaptive Router: cache routing (HIGH/MED/LOW tier) ← NEW, working
  - Stronger Judge: 6 checks ← NEW/upgraded
  - BFT code: dormant behind feature flag ← ready to activate on upgrade

  ## Sprint 4 Starting Baseline
  avg_generation_ms: 76,899 (single gen + judge)
  first_run_pass_rate: 1.00 (Sprint 2 level, restored)
  adaptive_router: active (will accelerate Sprint 4)

================================================================================
STEP 4 — Verify restored baseline
================================================================================
Run: uv run python audit/sprint3/measure_sprint3.py --live

Expect:
  first_run_pass_rate: ≥ 0.90 (BFT disabled → back to Sprint 2 level)
  avg_generation_ms: ~76,899ms (single gen, not 3×)
  judge_approved_rate: ≥ 0.85

Print:
  "SPRINT 3 CLOSED — BFT deferred (hardware), Judge strengthened"
  "Baseline restored: pass_rate=X, latency=Xms"
  "Ready for Sprint 4: ContractSkill + Long-term Memory"

DO NOT run 3 generators. DO NOT try another BFT variant.