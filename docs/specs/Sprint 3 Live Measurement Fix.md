Sprint 3 measurement returned mode="dry" with all latencies=0. 
Fix TWO bugs in audit/sprint3/measure_sprint3.py then run with real Ollama.

Read FIRST:
  1. audit/sprint3/measure_sprint3.py     (find the dry/live branching)
  2. src/routing/adaptive_router.py       (record_result signature)
  3. src/agents/bft_generator.py          (bft_generator_node)

================================================================================
BUG 1 — Live path not executing
================================================================================
Find why --live flag produces dry run output.
Common causes:
  a. argparse --live flag exists but conditional is: `if args.live` 
     while default is None (falsy) — but args.live=True still hits dry branch
  b. Import error in bft_generator_node silently caught → falls back to dry
  c. Ollama not reachable → exception caught → falls back to dry

Fix: Add an explicit startup check at the top of the live path:
  import httpx, sys
  resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
  if resp.status_code != 200:
      print("ERROR: Ollama not reachable. Start Ollama first.")
      sys.exit(1)
  print(f"Ollama OK — models: {[m['name'] for m in resp.json()['models']]}")

Then add a print before the first LLM call:
  print(f"[BFT] Starting case {i+1}/{dataset_size}, tier=LOW, 3 serial generators...")

If you see "Ollama OK" but no "[BFT] Starting..." → the conditional branching 
is wrong. Fix it.

================================================================================
BUG 2 — Warm cache never populated
================================================================================
In the measurement script, between pass 1 and pass 2:
  - Pass 1 must call router.record_result() for each case that passed
  - Pass 2 must read the updated SQLite cache

Find where pass 1 results are collected. After each case's result:
  router.record_result(
      url=case["target_url"],
      requirement=case["requirement_text"],
      domain=case["domain"],
      passed=result["passed"],
      generated_code=result.get("generated_code"),
  )

Then at start of pass 2, verify cache has entries:
  conn = sqlite3.connect(str(db_path))
  count = conn.execute("SELECT COUNT(*) FROM url_test_cache").fetchone()[0]
  print(f"[CACHE] Before pass 2: {count} entries in url_test_cache")

If count=0 after pass 1: record_result is not being called or 
db_path in measurement script ≠ db_path in AdaptiveRouter.

================================================================================
AFTER FIXING — Run live measurement
================================================================================
  uv run python audit/sprint3/measure_sprint3.py --live

Expected console output pattern:
  Ollama OK — models: ['qwen2.5-coder:7b-instruct-q4_K_M']
  [BFT] Starting case 1/10, tier=LOW, 3 serial generators...
  [BFT] Generator-0 (T=0.0): 71234ms, PASS
  [BFT] Generator-1 (T=0.3): 68901ms, PASS
  [BFT] Generator-2 (T=0.7): 79234ms, PASS
  [BFT] Consensus: 2_of_3, Judge: APPROVED
  [CACHE] Before pass 2: 10 entries in url_test_cache
  [BFT] Starting case 1/10, tier=HIGH, returning cached...

If you see this pattern → measurement is running correctly.
Expected wall time: ~40 minutes (pass 1: ~35min, pass 2: ~5min with cache)

================================================================================
OUTPUT
================================================================================
Overwrite audit/sprint3/sprint3_results.json with live results.
"mode" field must be "live" not "dry".
All latency fields must be non-zero.
Pass 2 routing_metrics.HIGH_tier_pct must be > 0.

Print final line:
  "SPRINT 3 LIVE MEASUREMENT COMPLETE"
  "Pass 1 avg_bft_latency: Xms | Pass 2 HIGH_tier_pct: Y%"

DO NOT accept zero latencies as a valid live result.
DO NOT mark Sprint 3 closed if mode="dry".