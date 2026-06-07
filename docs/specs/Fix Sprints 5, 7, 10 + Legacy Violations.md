## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Fix measurement integrity + legacy violations found by audit
READ audit report output before touching any file.
ห้าม modify sprint 4, 6, 8, 9 (trusted — leave untouched)

## OBJECTIVE
GOAL: แก้ 3 sprint measure scripts และ legacy violations ให้วัดของจริง
ไม่ใช่ sentinel values หรือ structurally-guaranteed outcomes

---
## FIX 1 — Sprint 5: exploration_coverage + skills_reused

FILE: audit/sprint5/measure_sprint5.py

PROBLEM A — coverage sentinel (line 162-172):
  BEFORE: return 1.0 if nodes else 0.0
  AFTER:
    REACHABLE_ESTIMATE = 8  # ToDoMVC known states: empty/one/multi/active/completed/all-done + 2 edge
    coverage = min(1.0, len(unique_node_ids) / REACHABLE_ESTIMATE)
    # unique_node_ids = set of node.node_id from SFGStore.all_nodes()

PROBLEM B — skills_reused sentinel (line 126-129, 212):
  BEFORE: seed exactly 2 skills; count = len(loaded_skills)
  AFTER:
    - Remove _seed_contract_skills() forced seeding
    - skills_reused = count of skills where skill.skill_id appears in
      any GeneratedTest.source_skill_id from the hypothesis execution run
    - If ContractSkillStore is empty → skills_reused = 0 (do NOT seed to meet gate)
    - Gate ≥2 must be earned, not seeded

GATE ADJUSTMENT (only if real measurement cannot reach ≥0.70 coverage):
  Update gate comment to reflect real reachable states:
  # coverage gate ≥0.70 = found ≥6 of 8 known ToDoMVC states

---
## FIX 2 — Sprint 7: self_heal_triggered + test_pass_rate

FILE: audit/sprint7/measure_sprint7.py

PROBLEM A — self_heal_triggered constant (line 206-214):
  BEFORE:
    try:
        await repair_engine.repair(...)
        triggered = 1
    except:
        triggered = 1   # ← always 1 regardless
  AFTER:
    triggered = 0
    try:
        result = await repair_engine.repair(...)
        if result.strategy != RepairStrategy.NONE:
            triggered = 1
    except RepairFailedError:
        triggered = 0   # repair attempted but failed = not triggered successfully

PROBLEM B — trivial smoke tests for Sprint 7 modules:
  In src/codetest/generator.py — remove/replace the hardcoded
  `assert hasattr(...)` / `assert extractor is not None` test bodies
  for these classes: ShadowDOMExtractor, SPARouteTracker, HydrationGuard, ShadowLocatorBuilder

  REPLACE WITH: generator must produce real behavioral tests e.g.:
    ShadowDOMExtractor → test that extract() returns list[ShadowNode] with shadow_mode field
    HydrationGuard → test that detect_framework() returns one of "react"|"vue"|"angular"|"unknown"
    SPARouteTracker → test that RouteEvent has from_url != to_url after navigation
    ShadowLocatorBuilder → test that build() raises ValueError for XPath starting with "/"

---
## FIX 3 — Sprint 10: race condition false positive

FILE: src/race/swarm.py

PROBLEM — conflict_found always True (swarm.py:122):
  CURRENT:
    conflict_found = bool(errors) or (len(set(ax_hashes)) > 1 and len(ax_hashes) >= 2)
  
  WHY IT FAILS: CDP nodeId/backendDOMNodeId differ between isolated contexts
  by construction → set(ax_hashes) always has >1 element regardless of behavior

  FIX — compare SEMANTIC state, not raw AX tree hash:
    def _semantic_hash(ax_tree: dict) -> str:
        # Extract only: role, name, checked, value (ignore nodeId/backendDOMNodeId/childIds)
        nodes = ax_tree.get("nodes", [])
        semantic = sorted([
            {"role": n.get("role",""), "name": n.get("name",{}).get("value",""), "checked": n.get("checked",{}).get("value","")}
            for n in nodes
            if n.get("role","") in ("checkbox","textbox","button","listitem")
        ], key=lambda x: (x["role"], x["name"]))
        return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()

    conflict_found = bool(errors) or (len(set(semantic_hashes)) > 1)
    # Now: identical app state across agents = same semantic_hash = no false positive

FILE: audit/sprint10/measure_sprint10.py

PROBLEM — scenarios s4 + s5 constructed to always conflict:
  REMOVE scenario s4 (random unique todo text that guarantees hash difference)
  REMOVE scenario s5 (Clear-completed timeout constructed to always error)

  REPLACE WITH 2 scenarios that can genuinely pass without conflict:
    s4: 3 agents add SAME todo text simultaneously
        expected_safe=True → if app deduplicates → no conflict
    s5: 2 agents read-only (filter Active, filter Completed) simultaneously
        expected_safe=True → read operations never conflict

  True conflict scenario (keep 1):
    s1: 2 agents toggle-all simultaneously on same localStorage
        NOTE: on ToDoMVC (localStorage-only) this WON'T produce real conflict
        → document this honestly: "localStorage isolation = no backend race possible"
        → recommend real backend target for meaningful race testing

ADD to sprint10_results.json:
  "race_detection_method": "semantic_hash_comparison",
  "false_positive_risk": "low (semantic fields only) | was: high (CDP nodeId included)"

---
## FIX 4 — Legacy Violations

FILE: src/core/sfg_engine.py line 53
  BEFORE: await page.accessibility.snapshot()
  AFTER:
    cdp = await page.context.new_cdp_session(page)
    ax_tree = await cdp.send("Accessibility.getFullAXTree")
    await cdp.detach()
    # use ax_tree["nodes"] directly

FILES: agents/generator_agent.py:90, agents/planner_agent.py:75,
       agents/healer_agent.py:79, agents/observer_driver/driver_agent.py:141,
       core/sfg_engine.py:157, finetune/export.py:222
  PATTERN to fix in each file:
    BEFORE: httpx.post("http://localhost:11434/...") or requests.post(...)
    AFTER:
      from src.core.adapter import OllamaAdapter
      # OllamaAdapter already holds _inference_semaphore
      # use adapter.generate() / adapter.embed() instead of direct HTTP calls

## VERIFICATION
After all fixes, rerun:
  uv run python audit\sprint5\measure_sprint5.py --live
  uv run python audit\sprint7\measure_sprint7.py --live
  uv run python audit\sprint10\measure_sprint10.py --live

Target (honest gates):
  Sprint 5:  coverage ≥ 0.70 (earned, not sentinel)
  Sprint 7:  self_heal_triggered = real 0 or 1 based on RepairStrategy
  Sprint 10: race_conditions_detected = real semantic conflicts (may be 0 on ToDoMVC)

## MUST NOT
- ห้าม modify sprint 4, 6, 8, 9
- ห้าม adjust gate thresholds เพื่อให้ผ่าน — fix the measurement, not the bar
- ห้าม seed/inject values เพื่อ meet gate
- ห้าม use page.accessibility ในทุก fix
- ห้าม direct httpx/requests to localhost:11434 — ต้องผ่าน OllamaAdapter เสมอ