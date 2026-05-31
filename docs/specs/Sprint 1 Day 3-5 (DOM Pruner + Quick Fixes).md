Read these files FIRST in this exact order, do not skip:
  1. audit\phase0\SPRINT1_DAY2_LOG.md       (Day 2 outcome + recommended fixes)
  2. audit\phase0\sprint1_day2_results.json (numbers to beat)
  3. audit\phase0\sprint1_targets.yaml      (dom_pruner target: ≤1000 tokens)
  4. audit\phase0\AUDIT_REPORT.md           (TD-4 specifically, section §8)
  5. src\llm\schemas.py                     (TestStep.action — needs validator fix)
  6. src\llm\instructor_client.py           (max_retries — needs reduction)
  7. src\agents\planner.py + src\agents\generator.py
  8. CLAUDE.md and docs\specs\SPEC_CORE.md

Then execute the Sprint 1 Day 3-5 plan using SUBAGENT-DRIVEN with 5 clusters and 
mandatory review gates.

================================================================================
ARCHITECTURE — Universal DOM Compression Pipeline
================================================================================

The pipeline produces a Page Abstract Model (PAM) — a compact representation 
the LLM consumes instead of raw DOM. Per the knowledge base 
DOM_Compression__AOM_Extraction spec, the pipeline is layered with explicit 
fallback chain:

  Layer 1 — AOM Snapshot       (cheapest, best for semantic web)
    Playwright accessibility.snapshot(interestingOnly=true)
    → flat list of actionable nodes with role/name/state/bbox
  
  Layer 2 — Prune4Web DOM Pruning (when AOM is sparse)
    Score each DOM node on 4 axes (semantic, visibility, interaction, textuality)
    Run scoring program IN-BROWSER via page.evaluate() — keeps LLM out of the 
    heavy work
    Reduce candidate nodes 25-50× before any LLM grounding
  
  Layer 3 — Semantic Compactor (always runs)
    Group AOM + pruned DOM into 5 top-level arrays:
      controls, containers, lists, forms, images, visuals (orphan bboxes)
    Cluster siblings with IoU > 0.8 + matching role/label
    Output as compact Markdown (human-debuggable) OR JSON (LLM-consumable)
    Hard cap: 25 cluster representatives per page state
  
  Layer 4 — Visual Fallback   (last resort, NOT implemented this sprint)
    Tagged TODO for Sprint 2: OCR on bboxes not covered by AOM
    Stub it but raise NotImplementedError if called

Locator stability priority (per spec): 
  data-testid > id > aria-label > role+name > visible text > robust CSS path > XPath
NEVER use XPath as primary. Preserve at least one stable locator per element.

================================================================================
CLUSTER STRUCTURE
================================================================================

CLUSTER A — Quick Fixes + Re-measure (Task 0, ~2 hours)
  Subagent context: 
    - src\llm\schemas.py (find TestStep model)
    - src\llm\instructor_client.py (find max_retries default)
    - src\agents\generator.py (find the instructor invocation)
    - audit\phase0\measure_sprint1_day2.py
  
  Deliverables:
    A1. In schemas.py TestStep, add field_validator (mode="before") on .action 
        that coerces list → str via "+".join(str(x) for x in v).
        Also add equivalent validator on any other field that legacy silently 
        accepted as list — grep the legacy_repair fallback to find which fields.
    
    A2. In instructor_client.py, expose max_retries as constructor parameter.
        Add a second InstructorClient instantiation for the GENERATOR path with 
        max_retries=1 (planner/healer keep max_retries=3).
        Use named instances: instructor_planner, instructor_generator, 
        instructor_healer — clear ownership.
    
    A3. Re-run audit\phase0\measure_sprint1_day2.py against the same dataset.
        Write results to audit\phase0\sprint1_day2.5_results.json (NEW file, 
        do not overwrite Day 2's).
    
  REVIEW GATE A:
    [ ] sprint1_day2.5.instructor.json_parse_failure_rate ≤ 0.05
    [ ] sprint1_day2.5.instructor.first_run_pass_rate ≥ 0.85 (no regression)
    [ ] sprint1_day2.5.instructor.avg_generation_latency_ms < 71,489 
        (validator + max_retries=1 should help)
    
    If any fail: investigate. The validator should be additive (no risk). If 
    max_retries=1 causes pass_rate drop > 5%, revert to max_retries=2 and 
    document.

CLUSTER B — AOM Extractor (Day 3, ~6 hours)
  Subagent context:
    - Cluster A outputs
    - Playwright Python docs (web_fetch if needed): 
      https://playwright.dev/python/docs/api/class-accessibility
    - src\browser\* (existing Playwright wrappers)
  
  Deliverables:
    B1. New module src\perception\__init__.py (empty package init)
    
    B2. New module src\perception\aom_extractor.py:
        class AOMExtractor:
          async def extract(page: Page, root_selector: str | None = None) 
            -> AOMSnapshot
        
        AOMSnapshot is a Pydantic V2 model with:
          - snapshot_id: str            (sha256 of timestamp + url)
          - url: str
          - timestamp_iso: str
          - root_node: AOMNode          (recursive tree, NOT flat yet)
          - node_count: int
          - extraction_latency_ms: int
        
        AOMNode (recursive):
          - role: str
          - name: str
          - value: str | None
          - state: dict[str, bool]      (checked, disabled, expanded, focused)
          - bbox: BBox | None           (normalized to viewport %)
          - children: list[AOMNode]
          - source_node_id: str         (deterministic from path + role + name)
        
        BBox:
          - x_pct: float = Field(ge=0, le=100)
          - y_pct: float = Field(ge=0, le=100)
          - w_pct: float = Field(ge=0, le=100)
          - h_pct: float = Field(ge=0, le=100)
        
        Implementation: 
          - Use page.accessibility.snapshot(interesting_only=True)
          - Use page.evaluate() to read viewport dimensions
          - Normalize coordinates to percentages
          - Walk the tree, generate deterministic source_node_ids
        
        Failure mode: if accessibility.snapshot returns None or < 5 nodes, 
        raise AOMSparseError. The caller will fall back to DOM pruning.
    
    B3. Sparse-AOM detector helper:
        def is_aom_sparse(snapshot: AOMSnapshot) -> bool
          Returns True if:
            - node_count < 5 OR
            - >50% of nodes have empty name AND empty value AND no state flags
  
  REVIEW GATE B:
    Write tests\test_aom_extractor.py with 3 tests:
    [ ] test_extract_returns_valid_snapshot — against demo.playwright.dev/todomvc
    [ ] test_sparse_aom_detection_on_canvas — against a known canvas-heavy page 
        (e.g., a small HTML fixture with only <canvas>)
    [ ] test_coordinates_are_normalized — assert all bbox values 0-100
    
    All 3 must pass before Cluster C.

CLUSTER C — DOM Pruner / Prune4Web (Day 4 morning, ~4 hours)
  Subagent context:
    - Cluster B outputs (AOMExtractor)
    - KB knowledge: DOM_Compression__AOM_Extraction + DOM_Pruned_Playwright_Agent_for_7B_LLM specs
  
  Deliverables:
    C1. New module src\perception\dom_pruner.py:
        class DOMPruner:
          async def prune(page: Page, root_selector: str | None = None) 
            -> PrunedDOMSnapshot
        
        Implementation rules:
          - Inject pruning script via page.evaluate() — runs entirely in browser
          - DO NOT pull full HTML to Python and parse there (token-expensive + slow)
          - Scoring program weights (configurable in agent.yaml):
              semantic_score:    role/tag/ARIA match (button, input, link → +10)
              visibility_score:  computedStyle.display != 'none' + 
                                  intersectionRatio > 0 (+5)
              interaction_score: tabindex >= 0 OR onclick OR role in 
                                  {button,link,input,...} (+8)
              textuality_score:  textContent.length 1-200 chars (+2), >200 (+1)
          - Drop nodes with total score < 5
          - For list-like / table-like structures: detect repeating patterns, 
            collapse to single template with item_count + first_item representative
          - Strip ALL of: <script>, <style>, <svg> innards, <link>, <meta>, 
            HTML comments, type-attribute defaults
          - Preserve: id, data-*, aria-*, role, alt, name, placeholder, value
        
        PrunedDOMSnapshot Pydantic V2 model:
          - snapshot_id, url, timestamp_iso (matches AOMSnapshot)
          - elements: list[PrunedElement]     (FLAT, not tree)
          - reduction_ratio: float            (original_count / pruned_count)
          - extraction_latency_ms: int
        
        PrunedElement:
          - element_id: str                   (deterministic fingerprint)
          - tag: str
          - role: str | None
          - accessible_name: str | None       (priority: aria-label > alt > text > placeholder)
          - text: str = Field(max_length=200) (trimmed with ellipsis)
          - state: dict[str, bool]
          - locator: str                      (best stable selector per priority)
          - bbox: BBox | None
          - score: float                      (Prune4Web total)
          - meta: dict                        (e.g., row_index, column_index for grids)
    
    C2. Locator synthesizer (helper module):
        src\perception\locator_synthesizer.py
        
        def best_locator(element_attrs: dict) -> str
          Priority chain:
            1. data-testid → "[data-testid='{value}']"
            2. data-pw     → "[data-pw='{value}']"
            3. id          → "#{id}"
            4. aria-label  → role + accessible name (Playwright getByRole format)
            5. role + visible text
            6. shortest unique CSS path (NOT XPath, NEVER XPath)
          
          Always return a Playwright-compatible string.
  
  REVIEW GATE C:
    tests\test_dom_pruner.py with 4 tests:
    [ ] test_prune_reduces_node_count_25x — pick a known dense page 
        (e.g., a Bootstrap docs page), assert reduction_ratio >= 25
    [ ] test_no_xpath_in_output — grep all locators in output, assert no "//"
    [ ] test_repeating_list_collapsed — fixture with 50 identical <li>, 
        assert collapsed to template + count
    [ ] test_strips_scripts_and_styles — fixture with <script> + <style>, 
        assert nothing of them in output
    
    All 4 must pass before Cluster D.

CLUSTER D — Semantic Compactor + Grounder Entry Point (Day 4 afternoon, ~4 hours)
  Subagent context:
    - Cluster B + C outputs
    - src\agents\planner.py + src\agents\generator.py (current callers of 
      whatever extracts page state)
  
  Deliverables:
    D1. New module src\perception\semantic_compactor.py:
        class SemanticCompactor:
          def compact(aom: AOMSnapshot | None, dom: PrunedDOMSnapshot | None,
                      max_items: int = 25, output_format: Literal["md","json"] = "md") 
            -> CompactPAM
        
        CompactPAM (Pydantic V2):
          - format: Literal["md", "json"]
          - content: str                      (the actual compact representation)
          - controls_count: int
          - forms_count: int
          - lists_count: int
          - estimated_tokens: int             (use tiktoken cl100k_base for estimate)
          - source: Literal["aom", "dom", "hybrid"]
          - dropped_nodes: int                (clipped due to max_items)
        
        Compaction logic:
          - Cluster siblings: IoU > 0.8 + same role + matching name pattern 
            → single cluster with cluster_id, count, representative
          - Group into 5 sections: controls | containers | lists | forms | images
          - Output Markdown format (default, easy to debug):
            
            ## Controls
            - [ref=c1] role=button name="Sign in" state=enabled @[78%,12%]
            - [ref=c2] role=link  name="Forgot password?" @[85%,12%]
            
            ## Forms
            - [ref=f1] role=textbox name="email" placeholder="Email address"
            - [ref=f2] role=textbox name="password" type=password
            
            (similar for lists, images, visuals)
          
          - Each ref must be deterministic — same page state → same refs
          - Keep total estimated_tokens ≤ 1000 (HARD LIMIT per sprint1_targets)
          - If over limit: drop lowest-score elements, log dropped_nodes count
    
    D2. New module src\perception\grounder.py — THE entry point:
        class Grounder:
          async def ground(page: Page, 
                          context_budget_tokens: int = 1000) -> CompactPAM
        
        Logic:
          1. Try AOMExtractor → check is_aom_sparse()
          2. If sparse OR exception: try DOMPruner
          3. Pass result(s) to SemanticCompactor
          4. If compaction exceeds budget: re-run with tighter max_items (15, 10)
          5. If still over budget at max_items=10: log warning, return what fits
          6. NEVER raise on budget overflow — degrade gracefully
        
        Failure metric: emit_grounder_metric() helper that writes to 
        audit\phase0\grounder_metrics.jsonl (one line per call) with:
          {timestamp, url, source, tokens, latency_ms, budget_overflow: bool}
        
        This is the data source for Day 5 measurement.
  
  REVIEW GATE D:
    tests\test_grounder.py:
    [ ] test_ground_returns_compact_pam_within_budget — against todomvc
    [ ] test_ground_falls_back_to_dom_on_sparse_aom — fixture page with empty AOM
    [ ] test_hybrid_source_when_aom_partial — mix of AOM + DOM elements
    [ ] test_budget_enforced — set budget=200, assert estimated_tokens ≤ 200
    
    All 4 must pass before Cluster E.

CLUSTER E — Integration + Measurement (Day 5, ~6 hours)
  Subagent context: All previous outputs + planner.py + generator.py
  
  Deliverables:
    E1. Wire Grounder into src\agents\planner.py:
        - Find where the planner currently extracts page state (likely a direct 
          accessibility snapshot or DOM dump)
        - Replace with: pam = await grounder.ground(page, context_budget_tokens=1000)
        - Pass pam.content (the compact string) into the planner prompt as 
          {{page_state}}
        - Feature flag: perception.use_grounder: true in agent.yaml 
          (default true; legacy path callable for A/B)
    
    E2. Wire Grounder into src\agents\generator.py the same way.
        Verify the generator's prompt template references {{page_state}} 
        (or rename/refactor as needed).
    
    E3. config\agent.yaml — add perception section:
        perception:
          use_grounder: true
          context_budget_tokens: 1000
          fallback_to_legacy_extraction: true
          prune_score_threshold: 5
          max_cluster_items: 25
          output_format: "md"
    
    E4. Create audit\phase0\measure_sprint1_day5.py — runs the full golden 
        dataset (use_grounder=true vs use_grounder=false) and writes:
        
        audit\phase0\sprint1_day5_results.json:
        {
          "measured_at_iso": "...",
          "dataset_size": N,
          "with_grounder": {
            "avg_context_tokens": 0,
            "p95_context_tokens": 0,
            "json_parse_failure_rate": 0.0,
            "first_run_pass_rate": 0.0,
            "avg_generation_latency_ms": 0,
            "p95_generation_latency_ms": 0,
            "grounder_source_distribution": {"aom": 0, "dom": 0, "hybrid": 0},
            "budget_overflow_count": 0
          },
          "without_grounder": {
            "avg_context_tokens": 0,                   /* from raw DOM/AOM */
            "json_parse_failure_rate": 0.0,
            "first_run_pass_rate": 0.0,
            "avg_generation_latency_ms": 0
          },
          "delta": {
            "context_tokens_reduction_pct": 0.0,
            "latency_change_pct": 0.0,
            "pass_rate_change_pct": 0.0
          },
          "verdict": "PASS | FAIL | REGRESSION"
        }
    
    E5. Write audit\phase0\SPRINT1_DAY5_LOG.md (full post-mortem)
        Include:
          - Context token reduction numbers
          - Latency change (expect significant improvement vs Day 2)
          - Any pass_rate change
          - Grounder source distribution (what % of pages used AOM vs DOM)
          - Issues encountered (Playwright API quirks, locator priority surprises)
          - Verdict per acceptance gate below
          - Recommendation for Sprint 2 (Self-Healing)
    
    E6. Update CLAUDE.md (point to src\perception\grounder.py as PAM entry point)
        Update docs\specs\SPEC_CORE.md (Perception layer section)
  
  REVIEW GATE E — THE ACCEPTANCE GATE:
    PASS iff ALL of:
      - with_grounder.avg_context_tokens ≤ 1000
      - with_grounder.p95_context_tokens ≤ 1500
      - with_grounder.first_run_pass_rate ≥ 0.85
      - with_grounder.avg_generation_latency_ms < 71,489 
        (Day 2 instructor latency — should drop significantly)
    
    FAIL (but not REGRESSION) iff:
      - context_tokens met but latency or pass_rate slightly off
      → document and propose tuning, but mark verdict FAIL
    
    REGRESSION iff:
      - first_run_pass_rate drops below 0.50 (half of Day 2 0.90)
      → set perception.use_grounder: false, do NOT update docs, STOP

================================================================================
ESCALATION RULES
================================================================================
- If Playwright accessibility.snapshot() returns None on >30% of pages → 
  investigate, possibly need to wait_for_load_state first
- If pruning script (page.evaluate) throws on real-world pages with strict CSP →
  document the failure mode, mark as TD-12, proceed with reduced scope
- If estimated_tokens via tiktoken consistently mismatches actual Ollama token 
  count by >20% → switch to a per-model tokenizer (or accept variance, document)
- If any cluster takes > 6 hours wall time → report progress, ask user go/no-go
- If OOM or VRAM pressure during measurement → reduce dataset to 5 samples, 
  document partial measurement

================================================================================
WHAT NOT TO DO
================================================================================
- Do NOT pull raw page.content() to Python for parsing (defeats the purpose)
- Do NOT use XPath anywhere in locator synthesis
- Do NOT implement OCR / visual fallback this sprint (Layer 4 — tag as TODO)
- Do NOT change instructor_client.py beyond Cluster A's max_retries change
- Do NOT touch src\llm\structured.py legacy path
- Do NOT exceed context_budget_tokens=1000 at the default setting

================================================================================
OUTPUT FORMAT — Final summary line at the end:
  "SPRINT 1 DAY 3-5 COMPLETE — verdict: <PASS|FAIL|REGRESSION>"
  "Context: avg <X> tokens (was raw <Y>) — reduction <Z>%"
  "Latency: avg <X> ms (was Day 2 instructor 71,489) — change <Y>%"
  "Pass rate: <X>% (was Day 2 0.90)"
  "Next: Sprint 2 Self-Healing — pending user review"

BEGIN with Cluster A. Report at every gate.