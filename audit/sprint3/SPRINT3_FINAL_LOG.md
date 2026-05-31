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
