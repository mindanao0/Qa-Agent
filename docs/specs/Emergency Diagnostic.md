Sprint 3 revert caused severe regression: pass_rate 0.00, judge rejects 90%.
DO NOT add features. DO NOT fix anything yet. DIAGNOSE ONLY.

Read FIRST:
  1. src/llm/judge_client.py    (checks 5+6 that were added)
  2. src/agents/bft_generator.py  (_single_generator_with_judge fallback)

================================================================================
STEP 1 — Capture judge rejection reasons (5 minutes)
================================================================================
Add ONE temporary print statement to judge_client.py, inside evaluate():

  # TEMP DIAGNOSTIC — remove after sprint
  if not verdict.approved:
      print(f"[JUDGE_REJECT] confidence={verdict.confidence} "
            f"reason={verdict.rejection_reason!r} "
            f"issues={verdict.issues_found}")

Run ONE case from golden dataset (the first URL only):
  uv run python -c "
  import asyncio, pathlib, json
  from src.agents.bft_generator import bft_generator_node
  # load first case from audit/phase0/golden_dataset/passing.jsonl
  case = json.loads(open('audit/phase0/golden_dataset/passing.jsonl').readline())
  print('requirement:', case['requirement_text'][:80])
  print('url:', case['target_url'])
  # run generation with minimal state
  ...
  "

OR: add a --debug-single flag to measure_sprint3.py that runs 1 case with 
full console output. Pick whichever is faster to implement.

The output we need:
  - The generated code (first 20 lines)
  - The judge rejection reason
  - Whether generation completed or returned None

================================================================================
STEP 2 — Identify which of these 4 scenarios is true
================================================================================

Scenario A: Generator returns None (code is None, judge never runs)
  Evidence: "[JUDGE_REJECT]" line never prints, but pass_rate=0
  Fix: check _single_generator_with_judge() for why it returns None

Scenario B: Generator succeeds but Judge rejects (check 5 or 6 too strict)
  Evidence: "[JUDGE_REJECT]" prints with specific issue in issues_found
  Fix: rollback checks 5 and 6, restore 4-check judge

Scenario C: Generator output is different (shorter/wrong format)
  Evidence: generated code looks different from Sprint 2 output
  Fix: check if generator prompt changed during Sprint 3 work

Scenario D: Timing issue (33ms = some code path bypassing real generation)
  Evidence: code is empty string or stub
  Fix: trace the execution path in bft_generator_node fallback

================================================================================
STEP 3 — Report ONLY (do not fix yet)
================================================================================
Print:
  "DIAGNOSTIC RESULT: Scenario <A/B/C/D>"
  "Generated code (first 5 lines): <paste>"
  "Judge rejection reason: <paste or 'judge never ran'>"
  "Generation ms for this case: <actual>"

Do NOT fix anything. Wait for analysis before changing code.