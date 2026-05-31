REGRESSION ROOT CAUSE: Grounder raises RuntimeError on perception failures 
instead of degrading gracefully. The original spec mandated "NEVER raise — 
degrade gracefully" but this rule was applied only to budget overflow, not to 
perception layer failures. Two underlying bugs trigger the raises:

  TD-15a: page.accessibility was removed from Playwright Python ≥1.34
          → AOMExtractor unconditionally raises on ALL pages
  TD-15b: el.className is SVGAnimatedString (not str) on SVG elements
          → DOMPruner crashes on any page with SVG (which is most modern sites)

This recovery sprint fixes both bugs + adds the safety net the spec required. 
Goal: re-run measurement, achieve PASS or upgrade FAIL to non-regression.

Read FIRST:
  1. audit\phase0\SPRINT1_FINAL_LOG.md          (TD-15 details + verdict)
  2. audit\phase0\sprint1_day5_results.json     (current numbers)
  3. audit\phase0\sprint1_day5_measurement.log  (the actual RuntimeError traces)
  4. src\perception\aom_extractor.py            (TD-15a location)
  5. src\perception\dom_pruner.py               (TD-15b location)
  6. src\perception\grounder.py                 (where safety net belongs)
  7. Web-fetch if needed:
     - https://playwright.dev/python/docs/api/class-cdpsession
     - https://playwright.dev/python/docs/aria-snapshots

================================================================================
CLUSTER R1 — Fix Bugs (TD-15a + TD-15b) — ~3 hours
================================================================================

Subagent context: aom_extractor.py, dom_pruner.py, current Playwright version 
in pyproject.toml. Verify Playwright version FIRST:
  uv pip show playwright | grep -i version

R1.1 — TD-15a fix (AOM Extractor):

  Playwright Python ≥1.34 removed page.accessibility. The replacement is the 
  Chrome DevTools Protocol (CDP) directly. CDP is universal, works on any 
  Chromium version Playwright supports, and gives the full accessibility tree.
  
  Refactor src\perception\aom_extractor.py:
  
    async def extract(self, page: Page, root_selector: str | None = None) 
        -> AOMSnapshot:
        cdp = await page.context.new_cdp_session(page)
        try:
            await cdp.send("Accessibility.enable")
            result = await cdp.send("Accessibility.getFullAXTree")
            # result["nodes"] is a flat list of AXNode dicts
            # Each has: nodeId, role.value, name.value, properties, 
            # parentId, childIds, backendDOMNodeId
            
            root_node = self._build_tree_from_flat(result["nodes"])
            
            if root_selector:
                # Optionally filter to a subtree
                root_node = self._find_subtree(root_node, root_selector, page)
            
            # Then compute bbox per node via Accessibility.queryAXTree 
            # OR resolve backendDOMNodeId → DOM.getBoxModel
            # For now: skip bbox population if too complex, set bbox=None 
            # and add TD-14 to known debt (BBox population deferred — already 
            # marked in Sprint 1 Day 5 known debt)
            
            return AOMSnapshot(
                snapshot_id=...,
                url=page.url,
                root_node=root_node,
                node_count=len(result["nodes"]),
                ...
            )
        finally:
            await cdp.detach()
  
  Critical details:
    - Skip nodes with role="none" or role="generic" with no name (noise)
    - Skip nodes with ignored=True
    - Tree-build from flat list: index by nodeId, then walk childIds
    - Some childIds reference nodes not in the list (cross-frame) — handle 
      gracefully, mark as cross_frame_boundary instead of erroring
  
  Update tests\test_aom_extractor.py:
    - The current 1/1 test must have been mocked or hit a code path that 
      bypassed the real API. Find it. Replace with a REAL integration test:
      
      @pytest.mark.integration
      async def test_extract_returns_valid_snapshot_via_cdp():
          async with async_playwright() as pw:
              browser = await pw.chromium.launch()
              page = await browser.new_page()
              await page.goto("https://demo.playwright.dev/todomvc")
              snapshot = await AOMExtractor().extract(page)
              assert snapshot.node_count > 10
              assert snapshot.root_node is not None
              await browser.close()
      
      Add 2 more:
      - test_cdp_session_detached_after_extract — verify no leak
      - test_handles_navigation_during_extract — race condition guard

R1.2 — TD-15b fix (DOM Pruner SVG crash):

  In src\perception\dom_pruner.py, find the page.evaluate() string that runs 
  the in-browser pruning JS. The crash is at el.className.split(...) when el 
  is an SVG element.
  
  Replace EVERY occurrence of `el.className` in the JS payload with a helper:
  
    function getClassString(el) {
      const c = el.className;
      if (typeof c === 'string') return c;
      // SVGAnimatedString case
      if (c && typeof c.baseVal === 'string') return c.baseVal;
      // Fallback via getAttribute (works for any element)
      return el.getAttribute('class') || '';
    }
  
  Then use: getClassString(el).split(' ').filter(Boolean) 
  instead of: el.className.split(' ')
  
  Also harden:
    - el.id might be empty string — check truthy before using
    - el.getAttribute can return null — coalesce with || '' everywhere
    - el.getBoundingClientRect() can return all-zeros on display:none — skip 
      those nodes early in the scoring loop
  
  Test fixture:
    Create tests\fixtures\svg_heavy.html with mix of <svg><circle class="x"/></svg> 
    and regular <div class="y">. Add test:
    
    @pytest.mark.integration
    async def test_prune_handles_svg_elements_without_crashing():
        ...load the fixture...
        snapshot = await DOMPruner().prune(page)
        assert snapshot is not None
        assert snapshot.elements is not None  # not raised

GATE R1:
  [ ] uv pip show playwright | head -1 confirms version >= 1.34
  [ ] grep "page.accessibility" src\perception\ → 0 hits
  [ ] grep "el.className.split" src\perception\dom_pruner.py → 0 hits
  [ ] All test_aom_extractor.py tests pass (now including REAL CDP integration)
  [ ] All test_dom_pruner.py tests pass + the new SVG fixture test
  [ ] uv run pytest tests\test_grounder.py -v → still 6/6 pass

================================================================================
CLUSTER R2 — Add Safety Net (Graceful Degradation) — ~2 hours
================================================================================

Subagent context: grounder.py + planner.py + generator.py (the callers)

The spec said "NEVER raise on budget overflow — degrade gracefully." That rule 
must extend to ALL perception failures. The Grounder is the boundary — anything 
that escapes the Grounder kills generation.

R2.1 — Wrap the Grounder pipeline in defensive layers:

  In src\perception\grounder.py modify Grounder.ground():
  
    async def ground(self, page: Page, 
                     context_budget_tokens: int = 1000) -> CompactPAM:
        aom_snapshot = None
        dom_snapshot = None
        errors = []
        
        # Layer 1: AOM (best-effort, never raise)
        try:
            aom_snapshot = await self.aom_extractor.extract(page)
            if is_aom_sparse(aom_snapshot):
                errors.append("aom_sparse")
                aom_snapshot = None  # discard sparse, force DOM fallback
        except Exception as e:
            errors.append(f"aom_failed:{type(e).__name__}:{str(e)[:80]}")
            aom_snapshot = None
        
        # Layer 2: DOM pruning (best-effort, never raise)
        if aom_snapshot is None:
            try:
                dom_snapshot = await self.dom_pruner.prune(page)
            except Exception as e:
                errors.append(f"dom_failed:{type(e).__name__}:{str(e)[:80]}")
                dom_snapshot = None
        
        # Layer 3: Compaction (best-effort)
        if aom_snapshot is None and dom_snapshot is None:
            # COMPLETE FAILURE — return minimal viable PAM, do not raise
            errors.append("all_layers_failed")
            return self._emergency_pam(page, errors)
        
        try:
            pam = self.compactor.compact(aom_snapshot, dom_snapshot, 
                                         max_items=25, 
                                         output_format="md")
            if pam.estimated_tokens > context_budget_tokens:
                # Retry at tighter limit (existing logic preserved)
                pam = self.compactor.compact(..., max_items=15, ...)
            if pam.estimated_tokens > context_budget_tokens:
                pam = self.compactor.compact(..., max_items=10, ...)
            # Even if still over budget: log + return (existing graceful behavior)
            return pam
        except Exception as e:
            errors.append(f"compaction_failed:{type(e).__name__}")
            return self._emergency_pam(page, errors)
    
    def _emergency_pam(self, page: Page, errors: list[str]) -> CompactPAM:
        """Last-resort minimum-viable PAM when all perception layers fail.
        Surfaces page title + URL only. Better than nothing — the LLM at least 
        knows where it is."""
        return CompactPAM(
            format="md",
            content=f"## Page State (perception unavailable)\n"
                    f"- URL: {page.url}\n"
                    f"- title: {page.title()}\n"
                    f"- errors: {', '.join(errors)}\n"
                    f"- Note: full DOM extraction unavailable, plan with care\n",
            controls_count=0,
            forms_count=0,
            lists_count=0,
            estimated_tokens=estimate_tokens("...content..."),
            source="failure",  # <-- new enum literal in CompactPAM
            dropped_nodes=0,
        )

R2.2 — Update CompactPAM.source enum:
  
  Add "failure" to the Literal in CompactPAM.source (currently 
  Literal["aom","dom","hybrid"]). The new type is:
  Literal["aom","dom","hybrid","failure"]
  
  Update tests that assert on .source to allow "failure" in the perception-down 
  case.

R2.3 — Caller defensiveness in planner_node + generator_node:

  Wherever pam.content is interpolated into the LLM prompt, if pam.source == 
  "failure", append an explicit hint to the prompt:
  
    "Perception layer failed. You are operating in DEGRADED MODE. 
     Use only the URL and page title above to make a best-effort plan. 
     Prefer generic actions (navigate, wait) over selector-specific ones. 
     If you cannot proceed safely, return an empty plan with a NEEDS_HUMAN_REVIEW 
     status."
  
  Also emit a metric to audit\phase0\grounder_metrics.jsonl with 
  source="failure" so we can track frequency.

GATE R2:
  [ ] Grep src\perception\grounder.py for "raise " → only emit_grounder_metric 
      or comments, no actual raise statements escape ground()
  [ ] New unit test tests\test_grounder.py::test_emergency_pam_when_all_fail 
      passes (mock both extractors to raise, assert returns valid PAM)
  [ ] CompactPAM.source enum updated, type-check clean
  [ ] planner.py + generator.py handle pam.source == "failure" with degraded 
      prompt hint

================================================================================
CLUSTER R3 — Re-Enable + Re-Measure + Verdict — ~2 hours
================================================================================

R3.1 — Re-enable Grounder:
  In config\agent.yaml: perception.use_grounder: true
  In CLAUDE.md: update Sprint 1 status to "recovery in progress"

R3.2 — Sanity smoke test BEFORE full measurement:
  Hit each of the 10 golden dataset URLs once with Grounder, capture:
  
    audit\phase0\recovery_smoke.json:
    {
      "url": "...",
      "grounder_source": "aom|dom|hybrid|failure",
      "estimated_tokens": N,
      "errors": [...]
    }
  
  Expected after fix: at least 7/10 should be source="aom" or "dom" (NOT 
  "failure"). If ≥3/10 still source="failure", STOP — investigate, do not 
  proceed to full measurement.

R3.3 — Full measurement (only if smoke check passes):
  uv run python audit\phase0\measure_sprint1_day5.py
  
  Output: audit\phase0\sprint1_day5_recovery_results.json (NEW file, do not 
  overwrite the regression results — keep them for comparison)

R3.4 — Final verdict in audit\phase0\SPRINT1_FINAL_LOG.md (overwrite):

  Compare 3 result files:
    - sprint1_day2.5_results.json   (Instructor only, no Grounder)
    - sprint1_day5_results.json     (Grounder REGRESSION)
    - sprint1_day5_recovery_results.json (after R1+R2 fixes)
  
  Apply ORIGINAL Gate E criteria (no lowered bar):
    PASS iff:
      - avg_context_tokens ≤ 1000
      - p95_context_tokens ≤ 1500
      - first_run_pass_rate ≥ 0.85
      - avg_generation_latency_ms < 71,489
    
    FAIL (not regression) iff context met but pass_rate 0.50-0.85
    REGRESSION iff pass_rate < 0.50

  Take exactly ONE action per verdict (same as previous mini-sprint).

GATE R3:
  [ ] sprint1_day5_recovery_results.json exists with non-failure source 
      distribution (failure_count < 30% of dataset_size)
  [ ] SPRINT1_FINAL_LOG.md overwritten with 3-way comparison + final verdict
  [ ] CLAUDE.md reflects the actual final state, not aspirational

================================================================================
WHAT NOT TO DO
================================================================================
- Do NOT lower Gate E thresholds 
- Do NOT use page.accessibility (gone)
- Do NOT use aria_snapshot() unless verified to exist in current Playwright 
  version (it's 1.46+; check before using)
- Do NOT silence errors silently — always append to the errors list and log
- Do NOT skip the smoke check in R3.2

================================================================================
ESCALATION
================================================================================
- If CDP session can't even be created: report Chromium / driver mismatch, 
  ask user to reinstall playwright browsers via: 
  uv run playwright install chromium
- If ≥5/10 URLs still source="failure" after R1+R2: there's a third bug. 
  Surface the error patterns from grounder_metrics.jsonl and STOP — do not 
  run full measurement on a known-broken pipeline
- If recovery measurement shows pass_rate still < 0.50: this is a deeper 
  architectural issue with PAM content, not just bugs. Report root cause 
  hypothesis and stop — do not pivot to Sprint 2

================================================================================
BEGIN with R1. Report at every gate. Do not proceed past a failing gate.