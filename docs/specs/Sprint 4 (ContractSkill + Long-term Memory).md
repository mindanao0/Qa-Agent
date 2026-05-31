Sprint 3 CLOSED at 0.60. Root cause of 4 failing domains: requirements 
lack concrete target URLs. Sprint 4 solution: State Flow Graph maps real 
UI structure first, then generates tests from the map.

Read FIRST:
  1. CLAUDE.md + docs/specs/SPEC_CORE.md
  2. src/memory/episodic_store.py       (Sprint 2 — extend for SFG)
  3. src/perception/grounder.py         (Sprint 1 — re-use for SFG crawl)
  4. src/healing/ai_healer.py           (Sprint 2 — re-use in repair ops)
  5. src/routing/adaptive_router.py     (Sprint 3 — extend for SFG cache)
  6. audit/sprint3/sprint3_results.json (baseline: 0.60, 26,365ms gen)

================================================================================
ARCHITECTURE — ContractSkill + SFG Pipeline
================================================================================

State Flow Graph (SFG): directed graph of app UI states
  Node = unique page state (hash of AOM snapshot + URL)
  Edge = interaction (action_type + locator + result_state)
  Stored in: SQLite (relational edges) + LanceDB (semantic node search)

ContractSkill: compiled test artifact from SFG trajectory
  goal: str                     (what the test proves)
  preconditions: list[str]      (auth state, data state required)
  steps: list[ContractStep]     (ordered actions with locators from SFG)
  postconditions: list[str]     (assertions derived from result_state)
  repair_operators: list[str]   (SelReplace, PreInsert, ArgCorrect)

Repair Operators (minimal patch, NOT full rewrite):
  SelReplace  → swap failed locator with AOM-discovered alternative
  PreInsert   → add missing precondition step
  ArgCorrect  → fix wrong argument (fill value, URL, role name)

Pipeline flow:
  URL → SFG Crawler → State Flow Graph → Trajectory Planner →
  ContractSkill Compiler → Test Generator → Executor →
  Result → EpisodicStore (success) / RepairOperator (failure)

================================================================================
CLUSTER STRUCTURE — SUBAGENT-DRIVEN (5 clusters)
================================================================================

CLUSTER S4-A — State Flow Graph (SFG) Data Model (~3 hours)
============================================================
Subagent context:
  - src/perception/aom_extractor.py (Sprint 1 AOMExtractor — re-use)
  - src/perception/grounder.py (Grounder — re-use for page state)
  - src/routing/adaptive_router.py (SQLite pattern — follow same approach)

Deliverables:
  A1. New module src/contractskill/sfg.py

      SFGNode (Pydantic V2):
        node_id: str          (sha256 of url_path + aom_hash)
        url: str
        page_title: str
        aom_hash: str         (hash of AOMSnapshot content)
        pam_content: str      (CompactPAM from grounder, ≤1000 tokens)
        coverage_tags: list[str]  (auth_required, form, modal, list, etc.)
        outgoing_edges: list[str]  (edge_ids)
        discovered_at_iso: str
        visit_count: int = 0

      SFGEdge (Pydantic V2):
        edge_id: str           (sha256 of source_id + action_type + locator)
        source_node_id: str
        target_node_id: str
        action_type: str       (click, fill, navigate, select)
        locator: str           (best stable locator from locator_synthesizer)
        input_value: str | None
        safety_flag: str       (SAFE | BLOCKED | PENDING)
        replay_script: str     (minimal Playwright snippet to reproduce)

      class SFGStore:
        def __init__(self, db_path: pathlib.Path)

        def upsert_node(self, node: SFGNode) -> None
        def upsert_edge(self, edge: SFGEdge) -> None
        def get_node(self, node_id: str) -> SFGNode | None
        def get_edges_from(self, node_id: str) -> list[SFGEdge]
        def find_path(self, start_url: str, goal_description: str,
                      max_depth: int = 10) -> list[SFGEdge] | None
        def node_count(self) -> int
        def edge_count(self) -> int

      SQLite schema:
        sfg_nodes table: all SFGNode fields
        sfg_edges table: all SFGEdge fields, indexed on source_node_id
        Same db as url_test_cache (Sprint 3 SQLite) — new tables only

  A2. Safety filter:
      BLOCKED_ACTION_PATTERNS = frozenset({
          "delete", "remove", "transfer", "payment",
          "password", "logout", "deactivate"
      })

      def is_safe_action(locator: str, input_value: str | None) -> bool:
          combined = (locator + (input_value or "")).lower()
          return not any(p in combined for p in BLOCKED_ACTION_PATTERNS)

GATE S4-A:
  tests/test_sfg.py (5 tests):
  [ ] test_node_upsert_and_retrieve
  [ ] test_edge_creates_connection_between_nodes
  [ ] test_find_path_returns_edge_sequence
  [ ] test_safety_filter_blocks_delete_actions
  [ ] test_same_url_different_aom_creates_different_nodes

CLUSTER S4-B — SFG Crawler (~4 hours)
========================================
Subagent context:
  - Cluster S4-A outputs (SFGStore, SFGNode, SFGEdge)
  - src/perception/aom_extractor.py (AOMExtractor — CDP-based)
  - src/perception/grounder.py (Grounder)
  - src/perception/locator_synthesizer.py (best_locator)
  - config/agent.yaml (exploration limits from Q27)

Deliverables:
  B1. New module src/contractskill/crawler.py

      class SFGCrawler:
        def __init__(self, sfg_store: SFGStore, grounder: Grounder,
                     config: CrawlerConfig)

        async def crawl(self, seed_url: str,
                        credentials: dict | None = None) -> SFGNode:
          """
          BFS exploration from seed_url.
          Returns root SFGNode of the discovered graph.
          """

        async def _visit_node(self, page: Page,
                               parent_edge: SFGEdge | None) -> SFGNode
        async def _discover_edges(self, page: Page,
                                   node: SFGNode) -> list[SFGEdge]
        async def _is_new_state(self, page: Page,
                                 known_nodes: dict[str, SFGNode]) -> bool

      CrawlerConfig (Pydantic V2, read from agent.yaml exploration section):
        max_time_minutes: int = 30
        max_tokens: int = 500_000
        max_pages: int = 100
        max_depth: int = 40
        max_outgoing_per_node: int = 60
        model_config = ConfigDict(frozen=True)

      Crawl rules:
        - BFS (not DFS) to discover breadth-first
        - Dedup by node_id (url_path + aom_hash)
        - Skip BLOCKED_ACTION_PATTERNS edges
        - Stop when ANY hard limit hit (time OR tokens OR pages)
        - After navigation: call StateValidator to confirm new state
        - Store every discovered node+edge in SFGStore immediately
          (resumable if interrupted)

      Token counter: use src/perception/token_counter.py (Sprint 1)
        Track running total — stop when max_tokens approached

GATE S4-B:
  tests/test_crawler.py (4 tests — use mock page):
  [ ] test_crawl_discovers_nodes_from_seed
  [ ] test_deduplication_prevents_revisit
  [ ] test_hard_limit_stops_crawl
  [ ] test_blocked_actions_skipped

CLUSTER S4-C — ContractSkill Compiler (~3 hours)
=================================================
Subagent context:
  - Cluster S4-A outputs (SFGStore, SFGEdge)
  - src/llm/instructor_client.py (instructor_generator, T=0.0)
  - src/llm/schemas.py (TestPlan, GeneratedCode — extend)

Deliverables:
  C1. New module src/contractskill/compiler.py

      ContractStep (Pydantic V2):
        step_number: int
        action_type: str
        locator: str              (from SFGEdge, verified against AOM)
        input_value: str | None
        expected_state_hash: str  (target SFGNode.node_id after action)
        model_config = ConfigDict(extra="forbid", frozen=True)

      ContractSkill (Pydantic V2):
        skill_id: str              (sha256 of goal + trajectory hash)
        goal: str
        target_url: str
        domain: str
        preconditions: list[str]
        steps: list[ContractStep]
        postconditions: list[str]
        repair_operators: list[Literal["SelReplace","PreInsert","ArgCorrect"]]
        created_at_iso: str
        success_count: int = 0
        failure_count: int = 0
        model_config = ConfigDict(extra="forbid")

      class ContractSkillCompiler:
        async def compile(self, goal: str, trajectory: list[SFGEdge],
                          domain: str) -> ContractSkill
          """
          Given a path through the SFG (list of edges), compile into
          a ContractSkill artifact with LLM-inferred postconditions.
          """

        async def _infer_postconditions(self, goal: str,
                                         last_node: SFGNode) -> list[str]
          """Single LLM call (T=0.0) to infer what should be true 
          after executing the trajectory. Keep under 3 postconditions."""

      C2. ContractSkill storage:
          Add new LanceDB table: contract_skills
          Fields: skill_id, goal, domain, target_url, steps_json (serialized),
                  vector (embedding of goal + domain + steps summary),
                  created_at_iso, success_count, failure_count

          Query method:
          async def find_matching_skill(goal: str, url: str,
                                         domain: str) -> ContractSkill | None
            Hybrid search: vector similarity + url filter
            Returns None if best match confidence < 0.70

GATE S4-C:
  tests/test_compiler.py (4 tests):
  [ ] test_compile_produces_valid_contract_skill
  [ ] test_postconditions_inferred_from_last_node
  [ ] test_skill_stored_and_retrievable_by_goal
  [ ] test_find_matching_skill_returns_none_below_threshold

CLUSTER S4-D — Repair Operators + Planner Integration (~3 hours)
=================================================================
Subagent context:
  - All previous S4 outputs
  - src/healing/ai_healer.py (Sprint 2 — re-use heal chain)
  - src/agents/planner.py (Sprint 2 — extend memory injection)

Deliverables:
  D1. New module src/contractskill/repair.py

      class RepairEngine:
        async def repair(self, skill: ContractSkill,
                          failed_step: ContractStep,
                          failure_sig: str,
                          page: Page) -> ContractSkill | None:
          """
          Applies minimal patch operators. Returns patched ContractSkill
          or None if unable to repair (caller falls back to full regeneration).
          """

        async def _sel_replace(self, step: ContractStep,
                                page: Page) -> ContractStep | None
          # Re-run AOM discovery for step's expected element
          # Return new step with updated locator or None if not found

        async def _pre_insert(self, skill: ContractSkill,
                               failed_step: ContractStep) -> ContractSkill
          # Add missing precondition step before the failed step
          # Use LLM to infer what navigation step was missing (1 call, T=0.0)

        async def _arg_correct(self, step: ContractStep,
                                failure_sig: str) -> ContractStep | None
          # Fix wrong argument (URL, fill value, role)
          # Pattern-match failure_sig → known corrections

      Repair cascade:
        1. Try SelReplace (no LLM, fast)
        2. Try ArgCorrect (pattern match, no LLM)
        3. Try PreInsert (1 LLM call)
        4. If all fail: return None → trigger full SFG re-crawl of page

  D2. Extend planner.py ContractSkill injection:
      Before every planner decision:
        skill = await contract_store.find_matching_skill(goal, url, domain)
        if skill:
            inject into prompt:
              "## ContractSkill Available"
              "Proven trajectory for this goal (success_count={N}):"
              "Steps: {[s.action_type + ' ' + s.locator for s in skill.steps[:5]]}"
              "Postconditions: {skill.postconditions}"
              "Use this trajectory as the preferred plan."
        else:
            # No skill yet — trigger SFG crawl first (new behavior)
            if url not in sfg_store.known_urls():
                plan_type = "SFG_CRAWL_REQUIRED"
            else:
                plan_type = "STANDARD_GENERATION"

  D3. Update Reporter node to record ContractSkill outcomes:
      On test pass: skill.success_count += 1, update LanceDB
      On test fail: trigger RepairEngine, update skill or create post_mortem

GATE S4-D:
  [ ] SelReplace repairs locator using AOM (no LLM call)
  [ ] PreInsert makes exactly 1 LLM call
  [ ] Planner injects skill when confidence ≥ 0.70
  [ ] Reporter updates success_count after pass
  tests/test_repair.py: 4 tests

CLUSTER S4-E — Integration + Measurement (~3 hours)
====================================================
Subagent context: All S4-A through S4-D outputs + graph.py

Deliverables:
  E1. Wire SFGCrawler into main.py as new --mode explore:
      uv run python main.py --mode explore --url https://... --max-pages 20

  E2. Wire ContractSkill lookup into bft_generator_node:
      Before generation:
        skill = await contract_store.find_matching_skill(goal, url, domain)
        if skill and skill.success_count >= 2:
            → skip generation entirely, use skill.steps directly
            → bft_status = "CONTRACT_CACHE_HIT"
        else:
            → proceed with normal generation

  E3. Create audit/sprint4/measure_sprint4.py:
      For the 4 previously-failing domains:
        1. Run --mode explore on a URL that has user-list/billing/notifications
           (use https://the-internet.herokuapp.com which has many page types)
        2. Compile ContractSkills from discovered trajectories
        3. Re-run the 4 failing requirements using ContractSkill
        4. Measure pass_rate improvement

      sprint4_results.json:
        before_sprint4_pass_rate: 0.60  (Sprint 3 final)
        after_sprint4_pass_rate: X
        contract_skills_compiled: N
        skills_used_in_generation: N
        repair_operator_uses: {SelReplace: 0, PreInsert: 0, ArgCorrect: 0}
        sfg_nodes_discovered: N
        sfg_edges_discovered: N

GATE S4-E — SPRINT 4 ACCEPTANCE:
  PASS iff ALL of:
    [ ] after_sprint4_pass_rate ≥ 0.75     (improvement from 0.60)
    [ ] contract_skills_compiled ≥ 4       (one per previously-failing domain)
    [ ] skills_used_in_generation ≥ 2      (at least 2 cases use skill cache)
    [ ] sfg_nodes_discovered ≥ 10          (crawler actually works)
  
  Note: Target deliberately ≥0.75 not ≥0.90 — fine-tuning (Sprint 5+)
  needed for full domain coverage. ContractSkill provides the URL+structure
  foundation; memory+fine-tune will close the remaining gap.

  REGRESSION iff after_sprint4_pass_rate < 0.55

================================================================================
RULES FROM SPRINTS 1-3 (CARRY FORWARD)
================================================================================
  - asyncio.Semaphore(1) on ALL Ollama calls
  - CDP via AOMExtractor (never page.accessibility)
  - pathlib.Path everywhere
  - Pydantic V2 ConfigDict(extra="forbid")
  - EpisodicStore tables append-only
  - Judge: 4 checks only (no expanding beyond 4 for 7B model)
  - BFT: disabled (feature flag), dormant code preserved

NEW RULES FOR SPRINT 4:
  - SFG crawler: NEVER execute BLOCKED_ACTION_PATTERNS
  - ContractSkill: append success_count/failure_count, never delete skills
  - Repair operators: SelReplace + ArgCorrect first (no LLM)
                      PreInsert last (1 LLM call max)
  - SFG crawl hard limits: enforce from config/agent.yaml exploration section
  - skill.success_count ≥ 2 before using as generation shortcut

================================================================================
SUBAGENT-DRIVEN: Report at each cluster gate. BEGIN with S4-A.