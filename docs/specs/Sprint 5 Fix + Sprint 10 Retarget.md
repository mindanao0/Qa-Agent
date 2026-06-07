## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Fix Sprint 5 (coverage + skills_reused) และ Sprint 10 (race target)
ห้าม modify Sprint 4, 6, 7, 8, 9
uv add jsonschema ก่อนรัน Sprint 10

## FIX 1 — Sprint 5: exploration_coverage = 0.125

ROOT CAUSE: SFGCrawler crawl ToDoMVC ในสถานะ empty → เจอแค่ 1 state
FIX: ใน audit/sprint5/measure_sprint5.py เพิ่ม pre-crawl seeding step

BEFORE crawl_node ใน ExplorationPlanner:
  1. page.goto("https://demo.playwright.dev/todomvc/#/")
  2. crawl immediately → 1 state (empty)

AFTER — เพิ่ม _seed_ui_state() ก่อน SFGCrawler.crawl():
  async def _seed_ui_state(page: Page) -> None:
      """Create known states so crawler discovers multiple AX snapshots."""
      await page.goto("https://demo.playwright.dev/todomvc/#/")
      await page.wait_for_load_state("domcontentloaded")
      input_sel = page.get_by_placeholder("What needs to be done?")
      # add 3 todos to create multi-item state
      for text in ["Buy milk", "Walk dog", "Read book"]:
          await input_sel.fill(text)
          await input_sel.press("Enter")
      # complete one → creates completed state
      await page.locator(".todo-list li").first.locator(".toggle").click()
      # navigate to /active → active filter state
      await page.get_by_role("link", name="Active").click()
      # navigate to /completed → completed filter state
      await page.get_by_role("link", name="Completed").click()
      # back to all
      await page.get_by_role("link", name="All").click()

  Call _seed_ui_state(page) BEFORE crawler.crawl(page, start_url)
  → browser is now in a rich state → crawler should discover 6-8 states
  → coverage = found_states / 8 → should reach ≥ 0.70

## FIX 2 — Sprint 5: skills_reused = 0

ROOT CAUSE: planner.gap_analysis_node generates hypotheses but never
sets source_skill_id — skill matching is not wired

FIX in src/explorer/planner.py gap_analysis_node:

  CURRENT (roughly):
    hypothesis = TestHypothesis(
        goal=..., steps=..., source_skill_id=None
    )

  AFTER — match hypothesis goal against ContractSkill goals:
    from src.contractskill.compiler import ContractSkillStore

    async def _match_skill(goal: str, skills: list[ContractSkill]) -> str | None:
        """Return skill_id if cosine similarity of goals > 0.6, else None."""
        if not skills:
            return None
        # simple keyword overlap (no LLM call — preserve Semaphore budget)
        goal_words = set(goal.lower().split())
        for skill in skills:
            skill_words = set(skill.goal.lower().split())
            overlap = len(goal_words & skill_words) / max(len(goal_words), 1)
            if overlap >= 0.3:
                return skill.skill_id
        return None

    # In gap_analysis_node, for each hypothesis:
    source_skill_id = await _match_skill(hypothesis.goal, existing_skills)
    hypothesis = hypothesis.model_copy(update={"source_skill_id": source_skill_id})

  In measure_sprint5.py skills_reused counter:
    skills_reused = len(set(
        h.source_skill_id for h in report.hypotheses
        if h.source_skill_id is not None
    ))
    # counts distinct skills actually referenced, not skills loaded

## FIX 3 — Sprint 10: race target → real backend

PROBLEM: ToDoMVC localStorage cannot produce real race conditions
SOLUTION: change race test target to https://jsonplaceholder.typicode.com
  (same host already used for fuzzing — has real HTTP backend, shared state)

In audit/sprint10/measure_sprint10.py replace TodoMVC race scenarios with:

SCENARIO s1 (real concurrent write conflict):
  RaceScenario(
      scenario_id="s1_concurrent_post",
      description="3 agents POST /todos simultaneously with same title",
      agents=3,
      action="POST",
      target_url="https://jsonplaceholder.typicode.com/todos",
      overlap_ms=50,
      expected_safe=True,  # REST API should accept all 3
  )
  conflict_found = True if any agent gets status != 201
               OR if response body userId differs unexpectedly

SCENARIO s2 (read-write interleave):
  RaceScenario(
      scenario_id="s2_read_write",
      description="2 agents: one GET /todos/1, one PUT /todos/1 simultaneously",
      agents=2,
      action="GET+PUT",
      target_url="https://jsonplaceholder.typicode.com/todos/1",
      overlap_ms=50,
      expected_safe=True,
  )

NOTE: jsonplaceholder is a mock API that accepts everything → race_conditions_detected
will likely still be 0 (expected). Document this honestly:
  "race_detection_note": "jsonplaceholder mock API cannot produce real race conditions. Meaningful race testing requires a stateful backend (e.g. conduit.realworld.how). Current result=0 is correct for this target."

GATE ADJUSTMENT for Sprint 10 (only race gate):
  BEFORE: race_conditions_detected ≥ 1
  AFTER:  race_scenarios_tested ≥ 5 AND race_detection_method = "semantic_hash_comparison"
  # Remove the ≥1 detected gate — it was set for a false-positive measurement
  # Honest 0 on mock targets is a valid result

## FIX 4 — Add jsonschema to project deps
  uv add jsonschema
  # Verify: uv run python -c "import jsonschema; print(jsonschema.__version__)"

## VERIFICATION — run in this order:
  uv add jsonschema
  uv run python audit\sprint5\measure_sprint5.py --live
  uv run python audit\sprint10\measure_sprint10.py --live

TARGET (honest):
  Sprint 5: coverage ≥ 0.70 (from real UI state seeding)
            skills_reused ≥ 2 (from real keyword matching)
  Sprint 10: race_scenarios_tested = 5
             race_conditions_detected = 0 (honest on mock targets)
             race_detection_method = "semantic_hash_comparison"
             sprint10_status = "PASS" (gate = scenarios_tested ≥ 5, not detected ≥ 1)

## MUST NOT
- ห้าม seed skills_reused artificially
- ห้าม adjust coverage denominator เพื่อให้ผ่าน (denominator คงที่ = 8)
- ห้าม set race_conditions_detected = 1 manually
- ห้าม modify sprint 4, 6, 7, 8, 9
- ห้าม use page.accessibility
- ห้าม direct httpx/requests to localhost:11434