# Sprint 7: SPA Hydration + Shadow DOM traversal (TD-12)
# Project: D:\Code\qa-agent
# Baseline: pass_rate=1.00, all prior sprints green
# Goal: Agent handles SPA route transitions + open/closed Shadow DOM
#       without page.accessibility — CDP/AOMExtractor only

## ACCEPTANCE GATE (sprint7)
#   shadow_dom_elements_found  ≥ 5
#   spa_transitions_handled    ≥ 3
#   test_pass_rate             ≥ 0.75
#   self_heal_triggered        ≥ 1   (Shadow DOM locator drift → RepairEngine)
#   REGRESSION if pass_rate    < 0.75

## ARCHITECTURE — New files:

### src/spa/hydration_guard.py
# HydrationGuard — waits for SPA framework to finish rendering before AX snapshot
#
# class HydrationGuard:
#     """No required constructor args."""
#
#     async def wait_stable(self, page: Page, timeout_ms: int = 5000) -> None:
#         """
#         Wait until:
#           1. No pending fetch/XHR (networkidle2-equivalent via CDP)
#           2. No React/Vue/Angular pending state
#              - React:   window.__REACT_FIBER__ mutation observer settled
#              - Vue:     window.__vue_app__ scheduler flushed
#              - Angular: window.getAllAngularTestabilities() all stable
#           3. document.readyState == "complete"
#         CDP only — no page.wait_for_load_state("networkidle")
#         Semaphore NOT needed here (no Ollama)
#         """
#
#     async def detect_framework(self, page: Page) -> str:
#         """Returns: "react" | "vue" | "angular" | "unknown" """
#         # evaluate JS probes in order, return first match

### src/spa/route_tracker.py
# SPARouteTracker — detects client-side navigation without full page reload
#
# class RouteEvent(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     from_url:   str
#     to_url:     str
#     trigger:    str    # "pushState" | "replaceState" | "hashchange" | "popstate"
#     timestamp:  float
#
# class SPARouteTracker:
#     """No required constructor args."""
#
#     async def attach(self, page: Page) -> None:
#         """Inject JS listeners for history.pushState/replaceState + hashchange"""
#         # Override history.pushState via page.evaluate()
#         # Emit window.__spa_route_events__ array on each transition
#
#     async def flush(self, page: Page) -> list[RouteEvent]:
#         """Read and clear window.__spa_route_events__"""

### src/shadow/extractor.py
# ShadowDOMExtractor — CDP-based traversal for open + closed Shadow DOM
#
# class ShadowNode(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     node_id:    str
#     host_role:  str
#     shadow_mode: Literal["open", "closed"]
#     children:   list[str]   # child node_ids
#     ax_label:   str | None
#
# class ShadowDOMExtractor:
#     """No required constructor args."""
#
#     async def extract(self, page: Page) -> list[ShadowNode]:
#         """
#         CDP DOM.getFlattenedDocument(depth=-1, pierce=True)
#         → filter nodes with shadowRootType in ("open","closed","user-agent")
#         → build ShadowNode list
#         No page.accessibility. No page.evaluate piercing closed roots.
#         """
#
#     def merge_into_ax(
#         self,
#         ax_nodes: list[dict],
#         shadow_nodes: list[ShadowNode],
#     ) -> list[dict]:
#         """
#         Inject ShadowNode ax_label into matching ax_nodes by host_role.
#         Returns merged list safe for Grounder.ground()
#         """

### src/shadow/locator_builder.py
# ShadowLocatorBuilder — builds Playwright locators that pierce Shadow DOM
#
# def build(host_selector: str, inner_selector: str) -> str:
#     """
#     Returns Playwright CSS piercing locator:
#       f"{host_selector} >> css={inner_selector}"
#     Raises ValueError if inner_selector is an absolute XPath (starts with /)
#     """
#
# def build_chain(selectors: list[str]) -> str:
#     """Multi-level pierce: "host >> css=slot >> css=button" """

### src/spa/spa_agent.py
# SPAAgent — LangGraph sub-graph combining SPA + Shadow DOM
#
# Nodes:
#   1. attach_trackers  → HydrationGuard + SPARouteTracker attach
#   2. navigate         → page.goto(url), HydrationGuard.wait_stable()
#   3. extract_ax       → AOMExtractor + ShadowDOMExtractor.extract()
#                         → ShadowDOMExtractor.merge_into_ax()
#   4. ground           → Grounder.ground() on merged AX
#   5. generate_test    → Ollama (Semaphore(1), temp=0.2, format="json")
#   6. judge            → 4-check CodeJudge (reuse Sprint 6)
#   7. flush_routes     → SPARouteTracker.flush() → record RouteEvents
#   8. store_sfg        → SFGStore.upsert_node/edge with route transitions
#
# Edges:
#   START → attach_trackers → navigate → extract_ax → ground
#         → generate_test → judge → flush_routes → store_sfg → END
#   judge routes back to generate_test if needs_revision (max 2 retries)

### audit/sprint7/measure_sprint7.py
# Test targets:
#   SPA:         https://demo.playwright.dev/todomvc/#/   (hash-router)
#   Shadow DOM:  https://the-internet.herokuapp.com/shadowdom  (open shadow)
#
# Measure:
#   shadow_dom_elements_found  — len(ShadowDOMExtractor.extract(page))
#   spa_transitions_handled    — len(SPARouteTracker.flush())
#   test_pass_rate             — pass/total from TestExecutor
#   self_heal_triggered        — RepairEngine call count > 0

## OUTPUT audit/sprint7/sprint7_results.json:
{
  "shadow_dom_elements_found": <int>,
  "spa_transitions_handled":   <int>,
  "test_pass_rate":            <float>,
  "self_heal_triggered":       <int>,
  "regression":                <bool>,
  "sprint7_status":            "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only — no page.accessibility, no page.wait_for_load_state("networkidle")
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS enforced in execution layer
# - SecurityASTChecker before any subprocess
# - BFT: disabled (feature flag preserved)
# - ShadowLocatorBuilder: reject absolute XPath (starts with /)