Sprint 1 is COMPLETE (PASS). Sprint 2 objective: Dual-Layer Self-Healing with 
cross-session episodic memory. The agent must STOP repeating the same locator 
failures across runs.

Read FIRST in this exact order:
  1. audit\phase0\SPRINT1_FINAL_LOG.md           (what Sprint 1 built)
  2. audit\phase0\sprint1_day5_recovery_results.json (baseline: latency=94s, 
                                                       pass_rate=1.00)
  3. src\healing\ai_healer.py                    (existing healer — upgrade target)
  4. src\perception\aom_extractor.py             (Sprint 1 — re-use CDP extractor)
  5. src\perception\grounder.py                  (Sprint 1 — CompactPAM structure)
  6. src\llm\instructor_client.py                (structured output — re-use)
  7. src\llm\schemas.py                          (existing models — extend)
  8. CLAUDE.md, docs\specs\SPEC_CORE.md

================================================================================
ARCHITECTURE — Sprint 2 Self-Healing Pipeline
================================================================================

When executor encounters a Playwright TimeoutError or Locator error:

  HEAL ATTEMPT 1 — Fuzzy Match (fastest, no LLM, no browser)
    Jaro-Winkler similarity ≥ 0.85 on locator string corpus
    Source: locators\locators.json (atomic-write repo, existing)
    If match found at ≥0.85 → swap locator, retry immediately, log as FUZZY_HIT
    If match found at 0.60-0.85 → candidate only (used in Attempt 2 as hint)
    If no match → proceed to Attempt 2

  HEAL ATTEMPT 2 — AOM Re-discovery (one CDP call, no LLM)
    Re-run AOMExtractor on current page state
    Search AOM nodes for element semantically matching the failed locator:
      - role matches OR accessible name has word overlap ≥ 50%
    Generate new locator via locator_synthesizer.py (Sprint 1, re-use)
    If found → swap locator, retry, save to HealedExperiences, log as AOM_HIT
    If not found → proceed to Attempt 3

  HEAL ATTEMPT 3 — LanceDB Memory Lookup (semantic, cross-run)
    Query healed_experiences table: failure_signature + domain + page_type
    Retrieve top-3 past successful fixes ranked by:
      (0.4 × confidence) + (0.3 × impact_score) + (0.3 × recency)
    If top hit confidence ≥ 0.70 → apply strategy, retry
    If it works → update confidence score, log as MEMORY_HIT
    If all 3 fail → save to PostMortems, mark QUARANTINE

  STATE VALIDATION — runs after EVERY action (not just failed ones)
    Before action: AOM snapshot → pre_state_hash
    After action: AOM snapshot → post_state_hash + a11y_delta
    Classify: Success | Error State | Loading State
    If Error State detected: trigger heal pipeline immediately
    If Loading State: wait for stability (MutationObserver via page.evaluate)
    Store validation result in LangGraph state for Reporter

  PLANNER INJECTION — uses memory before EVERY planner decision
    Query healed_experiences + post_mortems for current domain + page
    Inject into planner prompt as:
      "Preferred strategies: [list of proven locator families]"
      "Avoid: [list of known-failing strategies for this signature]"
    Max 3 preferred + 3 avoid strategies (brevity for 7B context)

================================================================================
LANCEDB SCHEMA
================================================================================

Two new tables (next to existing qa_docs table):

Table: healed_experiences
  - memory_id: str           (uuid)
  - session_id: str          (uuid — LangGraph thread_id)
  - domain: str              (from 13-domain classifier, existing)
  - page_url: str
  - page_type: str           (form, dashboard, crud, checkout, etc.)
  - failure_signature: str   (deterministic hash of: error_class + url_path + 
                               role + action_type)
  - bad_strategy: str        (the locator that failed)
  - winning_strategy: str    (the locator that worked)
  - root_cause: str          (short, ≤100 chars: "overlay_intercepted_click",
                               "stale_reference", "renamed_aria_label", etc.)
  - confidence: float        (0.0-1.0, starts at 0.7 for new entries)
  - impact_score: float      (0.0-1.0, based on heal attempt number:
                               FUZZY=0.5, AOM=0.7, MEMORY=0.9)
  - created_at_iso: str
  - vector: list[float]      (embedding of: failure_signature + winning_strategy,
                               use existing embedding model from RAG pipeline)

Table: post_mortems
  - memory_id: str           (uuid)
  - session_id: str
  - attempt_id: int          (which heal attempt failed)
  - domain: str
  - page_url: str
  - failure_signature: str
  - current_plan: str        (what the agent was trying to do)
  - locator_candidates: list[str]
  - error_message: str
  - dom_snapshot_ref: str    (file path to saved PAM, not full DOM)
  - confidence: float = 0.0
  - impact_score: float      (severity, 0.0-1.0)
  - created_at_iso: str
  - vector: list[float]      (embedding of failure_signature + error_message)

Key constraints (from KB):
  - Both tables: append-only — never update or delete records
  - Post-mortems: store dom_snapshot_ref (file path to artifacts), NOT full DOM
  - Healed experiences: one canonical record per failure_signature when possible
  - Conflicting records: version with a new entry, do NOT merge blindly

================================================================================
CLUSTER STRUCTURE — SUBAGENT-DRIVEN (5 clusters)
================================================================================

Use SUBAGENT-DRIVEN approach. This sprint designs non-trivial LanceDB schema, 
a new fuzzy matching layer, and LangGraph state changes — each cluster needs 
clean context to avoid accumulating misunderstandings.

CLUSTER S2-A — Fuzzy Matcher + Jaro-Winkler (~3 hours)
================================================
Subagent context:
  - locators\locators.json (existing repo schema)
  - src\healing\ai_healer.py (existing healer)

Deliverables:
  A1. New module src\healing\fuzzy_matcher.py
      
      class FuzzyMatcher:
        def __init__(self, locator_repo_path: pathlib.Path)
        def find_candidates(self, failed_locator: str, 
                            threshold: float = 0.85) -> list[FuzzyCandidate]
        def _jaro_winkler(self, s1: str, s2: str) -> float
      
      FuzzyCandidate (Pydantic V2):
        - candidate_locator: str
        - similarity_score: float
        - source_file: str
        - last_used_iso: str | None
        - confidence_tier: Literal["HIGH", "MEDIUM"]  
          (HIGH=≥0.85, MEDIUM=0.60-0.85)
      
      IMPORTANT: Implement Jaro-Winkler from scratch (DO NOT use jellyfish 
      or rapidfuzz — add no new deps for a trivial algorithm):
      
        jaro(s1, s2): matching window = floor(max(len)/2) - 1
                      count matches + transpositions
        winkler boost: prefix length p ≤ 4, scaling factor 0.1
      
      The locator corpus is locators\locators.json — read atomically using 
      the existing FileLock pattern already in the codebase.
  
  A2. tests\test_fuzzy_matcher.py:
      [ ] test_exact_match_scores_1_0
      [ ] test_similar_css_selector_scores_above_085
      [ ] test_dissimilar_strings_score_below_060
      [ ] test_jaro_winkler_known_value ("MARTHA"/"MARHTA" = 0.9611)
      [ ] test_find_candidates_returns_sorted_by_score
      [ ] test_empty_repo_returns_empty_list

GATE S2-A:
  [ ] All 6 tests pass
  [ ] FuzzyMatcher loads locators.json without error
  [ ] No new package added to pyproject.toml

CLUSTER S2-B — State Validator (~3 hours)
==========================================
Subagent context:
  - src\perception\aom_extractor.py (Sprint 1 — AOMSnapshot schema)
  - src\perception\grounder.py (Sprint 1 — Grounder class)

Deliverables:
  B1. New module src\healing\state_validator.py
      
      class StateValidator:
        async def capture_pre_state(self, page: Page) -> PreState
        async def classify_post_action(self, page: Page, 
                                       pre: PreState) -> StateClassification
        async def wait_for_stability(self, page: Page, 
                                     timeout_ms: int = 5000) -> bool
      
      PreState (Pydantic V2):
        - url: str
        - pre_state_hash: str         (sha256 of normalized AOM node set)
        - pre_snapshot: AOMSnapshot   (from Sprint 1 extractor)
        - captured_at_iso: str
      
      StateClassification (Pydantic V2):
        - action_id: str
        - outcome_label: Literal["Success","Error_State","Loading_State","Unknown"]
        - pre_state_hash: str
        - post_state_hash: str
        - a11y_delta: A11yDelta
        - url_changed: bool
        - confidence_score: float
        - failure_signature: str | None  (set only on Error_State)
      
      A11yDelta:
        - nodes_added: int
        - nodes_removed: int
        - nodes_changed: int           (name/role/state changed)
        - error_text_found: bool       (AOM contains role=alert or text matching 
                                         error patterns)
        - loading_indicators: bool     (aria-busy=true, role=status, .skeleton)
      
      Classification rules:
        Success   if: url_changed OR large a11y_delta (>5 nodes changed) 
                       AND no error_text_found
        Error     if: error_text_found = true OR 
                       (delta is near-zero AND expected_change was non-trivial)
        Loading   if: loading_indicators = true AND delta < 3
        Unknown   if: none of the above (default graceful)
      
      wait_for_stability: inject a MutationObserver JS via page.evaluate()
        that resolves when no DOM mutations for 500ms — NEVER use page.wait_for_timeout()
  
  B2. tests\test_state_validator.py:
      [ ] test_captures_pre_state_as_aom_snapshot
      [ ] test_classifies_successful_navigation (mock url_changed=True)
      [ ] test_classifies_error_state_from_alert_role
      [ ] test_classifies_loading_state_from_aria_busy
      [ ] test_stability_waits_without_sleep

GATE S2-B:
  [ ] All 5 tests pass
  [ ] No page.wait_for_timeout() anywhere in state_validator.py
  [ ] StateClassification always returns (never raises)

CLUSTER S2-C — Episodic Memory (LanceDB) (~4 hours)
=====================================================
Subagent context:
  - src\rag\ directory (existing RAG + LanceDB setup — understand the pattern)
  - SPEC_CORE.md RAG section (how LanceDB is initialized)
  - The schemas defined above for healed_experiences + post_mortems

Deliverables:
  C1. New module src\memory\episodic_store.py
      
      class EpisodicStore:
        def __init__(self, db_path: pathlib.Path, embedding_fn: callable)
        async def save_healed_experience(self, exp: HealedExperience) -> str
        async def save_post_mortem(self, pm: PostMortem) -> str  
        async def retrieve_for_planning(self, query_text: str, domain: str,
                                         failure_signature: str,
                                         limit: int = 5) -> list[MemoryHit]
        async def evolve_locator_policy(self, domain: str,
                                         failure_signature: str) -> LocatorPolicy
      
      HealedExperience, PostMortem: Pydantic V2 models matching the schema above
      
      MemoryHit (Pydantic V2):
        - memory_type: Literal["healed_experience","post_mortem"]
        - failure_signature: str
        - strategy: str
        - root_cause: str
        - confidence: float
        - impact_score: float
      
      LocatorPolicy:
        - domain: str
        - failure_signature: str
        - preferred_strategies: list[str]   (top-3 by confidence × impact)
        - avoid_strategies: list[str]       (top-3 failed patterns)
      
      Key constraints from KB:
        - Both tables append-only (never upsert, never delete)
        - Hybrid search: 70% vector + 30% structured filter on domain + signature
        - Re-use the SAME embedding model from the existing RAG pipeline
          (find it in src\rag\ — do NOT add another embedding model)
        - Tables live in the SAME LanceDB database as qa_docs, just new tables
        - Paths: always under %LOCALAPPDATA%\ai-qa-agent\vectors\
        - Use pathlib.Path — never hardcode strings
  
  C2. New module src\memory\__init__.py (package init)
  
  C3. tests\test_episodic_store.py:
      [ ] test_save_and_retrieve_healed_experience
      [ ] test_post_mortem_is_append_only (save 2, assert table grows)
      [ ] test_retrieve_filtered_by_domain_and_signature
      [ ] test_evolve_policy_prefers_high_confidence_strategies
      [ ] test_retrieve_returns_ranked_by_composite_score

GATE S2-C:
  [ ] All 5 tests pass against a REAL LanceDB (not mocked)
  [ ] Same embedding model used as RAG pipeline (grep to confirm)
  [ ] No new LanceDB database file created — only new tables in existing DB

CLUSTER S2-D — Dual-Layer Healer Integration (~4 hours)
=========================================================
Subagent context:
  - Clusters S2-A, B, C outputs (FuzzyMatcher, StateValidator, EpisodicStore)
  - src\healing\ai_healer.py (existing — UPGRADE, not replace)
  - src\llm\instructor_client.py (structured output — for Attempt 3 if needed)

Deliverables:
  D1. Upgrade src\healing\ai_healer.py to implement the 3-attempt pipeline:
  
      async def heal(self, page: Page, failed_locator: str,
                     action_context: ActionContext) -> HealResult:
        
        failure_sig = self._compute_failure_signature(failed_locator, page.url)
        self._log_attempt_start(failure_sig)
        
        # Attempt 1: Fuzzy Match
        candidates = self.fuzzy_matcher.find_candidates(failed_locator, 0.85)
        for c in candidates[:3]:  # try top 3 HIGH confidence only
            if await self._try_locator(page, c.candidate_locator, action_context):
                await self.episodic_store.save_healed_experience(...)
                return HealResult(strategy="FUZZY_HIT", locator=c.candidate_locator)
        
        # Attempt 2: AOM Re-discovery
        aom_snapshot = await self.aom_extractor.extract(page)
        candidate = self._match_aom_node(aom_snapshot, failed_locator)
        if candidate:
            new_locator = self.locator_synthesizer.best_locator(candidate)
            if await self._try_locator(page, new_locator, action_context):
                await self.episodic_store.save_healed_experience(...)
                return HealResult(strategy="AOM_HIT", locator=new_locator)
        
        # Attempt 3: Memory lookup
        memories = await self.episodic_store.retrieve_for_planning(
            query_text=failed_locator + " " + action_context.action_type,
            domain=action_context.domain,
            failure_signature=failure_sig,
            limit=3
        )
        for m in memories:
            if m.confidence >= 0.70:
                if await self._try_locator(page, m.strategy, action_context):
                    return HealResult(strategy="MEMORY_HIT", locator=m.strategy)
        
        # All failed → quarantine
        await self.episodic_store.save_post_mortem(...)
        return HealResult(strategy="QUARANTINE", locator=None)
      
      HealResult (Pydantic V2):
        - strategy: Literal["FUZZY_HIT","AOM_HIT","MEMORY_HIT","QUARANTINE"]
        - locator: str | None
        - attempt_count: int
        - latency_ms: int
        - failure_signature: str
      
      ActionContext (Pydantic V2):
        - action_type: str          (click, fill, select, etc.)
        - domain: str
        - page_url: str
        - element_description: str  (human-readable, for embedding)
  
  D2. Add Planner Memory Injection to src\agents\planner.py:
      
      Before EVERY planner invocation:
        policy = await episodic_store.evolve_locator_policy(domain, failure_sig)
        
        Add to planner prompt (append, NOT replace):
          "## Memory-Based Strategy"
          "Preferred (proven): {policy.preferred_strategies[:3]}"
          "Avoid (known-fail): {policy.avoid_strategies[:3]}"
          "Based on {len(healed_experiences)} healed cases."
      
      If no memories exist yet → skip injection (first run has no history)
  
  D3. Integrate StateValidator into graph.py executor node:
      Before action: pre = await validator.capture_pre_state(page)
      After action:  classification = await validator.classify_post_action(page, pre)
      If Error_State: trigger heal pipeline with classification.failure_signature
      Store classification in LangGraph state (add field to AgentState)

GATE S2-D:
  [ ] ai_healer.py contains all 3 attempt strategies
  [ ] HealResult.strategy values are exactly the 4 Literals
  [ ] planner.py memory injection only fires when memories exist (not first-run)
  [ ] graph.py StateValidator integration present with Error_State trigger
  [ ] Full run: uv run python main.py --mode run 
      --script tests\generated\test_todomvc.py → pass with 0 quarantines

CLUSTER S2-E — Measurement + Sprint 2 Acceptance (~2 hours)
============================================================
Subagent context: All S2-A through S2-D outputs + baseline metrics

Deliverables:
  E1. Create audit\sprint2\measure_sprint2.py — runs golden dataset through 
      the upgraded healer + validator and writes 
      audit\sprint2\sprint2_results.json:
      {
        "measured_at_iso": "...",
        "dataset_size": N,
        "healing_metrics": {
          "heal_attempts_avg_per_failed_test": 0.0,
          "heal_success_rate": 0.0,
          "strategy_distribution": {
            "FUZZY_HIT": 0, "AOM_HIT": 0, "MEMORY_HIT": 0, "QUARANTINE": 0
          },
          "avg_heal_latency_ms": 0
        },
        "quality_metrics": {
          "first_run_pass_rate": 0.0,
          "post_healing_pass_rate": 0.0,
          "false_pass_rate_estimated": 0.0,
          "state_validation_error_detections": 0
        },
        "memory_metrics": {
          "healed_experiences_stored": 0,
          "post_mortems_stored": 0,
          "memory_hits_per_session": 0.0
        },
        "latency_metrics": {
          "avg_generation_ms": 0,
          "avg_healing_ms": 0,
          "avg_validation_ms": 0,
          "total_overhead_vs_sprint1_pct": 0.0
        }
      }
  
  E2. Write audit\sprint2\SPRINT2_FINAL_LOG.md with:
      - Architecture decisions made
      - Actual numbers vs targets
      - Tech debt (TDs) carried or added
      - Sprint 3 readiness assessment

GATE S2-E — SPRINT 2 ACCEPTANCE:
  PASS iff ALL of:
    [ ] post_healing_pass_rate ≥ 0.95             (Sprint 1 target from Q15)
    [ ] heal_success_rate ≥ 0.70                  (3 attempts → success ≥70%)
    [ ] total_overhead_vs_sprint1_pct ≤ 50%       (healing adds latency, bound it)
    [ ] healed_experiences_stored > 0             (at least 1 cross-run learning)
  
  FAIL (not regression) iff:
    - heal_success_rate 0.50-0.70 OR overhead 50-100%
  
  REGRESSION iff:
    - post_healing_pass_rate < 0.85 (below Sprint 1 PASS level)

================================================================================
RULES FROM SPRINT 1 (CARRY FORWARD)
================================================================================
- Never use page.wait_for_timeout() — event-driven only (MutationObserver, 
  page.wait_for_selector, expect)
- Never use page.accessibility (gone since 1.34) — use CDP via AOMExtractor
- Never raise from perception layer — always degrade to emergency_pam
- Locator priority: data-testid > id > aria-label > role+name > text > CSS
- NEVER XPath as primary locator
- asyncio.Semaphore(1) on ALL Ollama calls
- pathlib.Path everywhere, NEVER os.path string concat

================================================================================
NEW RULES FOR SPRINT 2
================================================================================
- EpisodicStore tables are APPEND-ONLY — no update, no delete, never
- Healing attempts ≤ 3 before quarantine (do not add Attempt 4+)
- Planner memory injection ≤ 6 total strategies (3 preferred + 3 avoid)
- StateValidator classification must complete in < 2 seconds (AOM is fast via CDP)
- DO NOT embed full DOM in post_mortem records — store file path only

================================================================================
BEGIN with Cluster S2-A. Use SUBAGENT-DRIVEN, report at each gate.