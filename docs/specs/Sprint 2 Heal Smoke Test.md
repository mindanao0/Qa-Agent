Sprint 2 live measurement shows all tests pass on first run, so the healing 
pipeline was never exercised. Before Sprint 3, prove the FUZZY_HIT → 
save_healed_experience round-trip works end-to-end per the recommendation 
in audit/sprint2/SPRINT2_FINAL_LOG.md §7.

Read FIRST:
  1. audit/sprint2/SPRINT2_FINAL_LOG.md  §7 (the smoke test spec)
  2. locators/locators.json              (the locator repository)
  3. src/healing/ai_healer.py            (the heal pipeline)
  4. src/memory/episodic_store.py        (save_healed_experience)

================================================================================
3-STEP SMOKE TEST (inline execution, no subagents)
================================================================================

STEP 1 — Seed locator repo
  Add ONE entry to locators/locators.json using the existing atomic-write 
  pattern (FileLock + temp file rename):
  {
    "selector": "[data-testid='todo-input']",
    "role": "textbox",
    "accessible_name": "What needs to be done?",
    "page_url": "https://demo.playwright.dev/todomvc",
    "healed_count": 0,
    "last_used": null
  }
  
  The "broken" variant we will use: "[data-testid='todo-input-BROKEN']"
  Jaro-Winkler similarity of these two strings must be ≥ 0.85 — verify it 
  manually before running by calling FuzzyMatcher directly:
  
    from src.healing.fuzzy_matcher import FuzzyMatcher
    fm = FuzzyMatcher(locator_repo_path=Path("locators/locators.json"))
    candidates = fm.find_candidates("[data-testid='todo-input-BROKEN']", 0.85)
    print(candidates)  # must return at least 1 HIGH tier candidate
  
  If similarity < 0.85: adjust the broken variant until it is ≥ 0.85.
  Document the actual score in the smoke test log.

STEP 2 — Run a single heal cycle
  Write a small standalone script audit/sprint2/heal_smoke.py:
  
  - Launch Playwright chromium (headless=True)
  - Navigate to https://demo.playwright.dev/todomvc
  - Capture pre-state with StateValidator
  - Simulate a locator failure: try to locate "[data-testid='todo-input-BROKEN']"
    → expect TimeoutError (5s timeout)
  - On TimeoutError: call AIHealer.heal() with:
      failed_locator = "[data-testid='todo-input-BROKEN']"
      action_context.domain = "crud"
      action_context.page_url = page.url
  - Assert: result.strategy == "FUZZY_HIT"
  - Assert: result.locator == "[data-testid='todo-input']"
  - Assert: a row exists in healed_experiences LanceDB table
    (query: episodic_store.retrieve_for_planning("todo-input", "crud", ..., limit=1)
     → must return 1 MemoryHit)
  - Print: "HEAL SMOKE: FUZZY_HIT confirmed, healed_experiences count=N"

STEP 3 — Verify cross-run memory persists
  In the SAME script, after the heal:
  
  - Create a NEW EpisodicStore instance (simulates a fresh session)
  - Query healed_experiences with same failure_signature
  - Assert the stored experience is retrievable
  - Print: "MEMORY PERSISTENCE: cross-session retrieval confirmed"

================================================================================
SUCCESS CRITERIA
================================================================================
  [ ] FuzzyMatcher similarity ≥ 0.85 on the chosen broken/real pair
  [ ] AIHealer returns HealResult(strategy="FUZZY_HIT")
  [ ] LanceDB healed_experiences table has ≥ 1 row after the run
  [ ] New EpisodicStore instance can retrieve the stored experience
  [ ] audit/sprint2/heal_smoke.py exits 0

On success: write 3 lines to audit/sprint2/SPRINT2_FINAL_LOG.md (append):
  "## Heal Smoke Test Result"
  "Verdict: PASS — FUZZY_HIT confirmed, healed_experiences_stored=1"
  "Sprint 2 status: CLOSED → PASS"

On failure: report the exact assertion that failed — do NOT mark as PASS.

================================================================================
THEN PRINT:
  "SPRINT 2 CLOSED — verdict: PASS"
  "Heal chain verified: FUZZY_HIT → LanceDB write → cross-session read"
  "Latency baseline for Sprint 3: avg_generation_ms=76,899"
  "Ready for Sprint 3: BFT Generator"