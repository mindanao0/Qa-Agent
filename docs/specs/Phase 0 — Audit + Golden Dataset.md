Read CLAUDE.md, docs/specs/SPEC_CORE.md, and docs/specs/SPEC_UNIVERSAL_DOMAIN.md 
first. Then perform a comprehensive Phase 0 audit BEFORE any new feature work.

================================================================================
HARD CONSTRAINTS
================================================================================
- OS: Windows 11 bare-metal. Use pathlib.Path. Base path: D:\Code\qa-agent
- Ollama at http://localhost:11434, model qwen2.5-coder:7b-instruct-q4_K_M
- Python 3.13 with uv package manager
- Read-only audit — DO NOT modify any existing source code in this phase
- Output everything under D:\Code\qa-agent\audit\phase0\

================================================================================
DELIVERABLES (4 artifacts, all under audit\phase0\)
================================================================================

DELIVERABLE 1 — AUDIT_REPORT.md
================================
Walk the entire src\ directory tree. Produce a single markdown file with these 
sections in this exact order:

## 1. Module Inventory
A table listing every .py file under src\ with columns:
  - path
  - LOC (lines of code, excluding blank/comment)
  - exports (top-level functions/classes)
  - imports_internal (other src.* modules it imports)
  - imports_external (third-party packages)
  - has_tests (yes/no — check tests\ for matching test file)

## 2. Dependency Graph
A Mermaid flowchart (```mermaid block) showing internal module dependencies. 
Cluster by subsystem (core, graph, rag, finetune, locators, browser, etc.).

## 3. Architecture Compliance Check
For each existing LangGraph node (planner, generator, executor, healer, 
reporter), verify against SPEC_CORE.md. Mark each as:
  - ✅ COMPLIANT — matches spec
  - ⚠️  PARTIAL — exists but missing required behavior
  - ❌ MISSING — not implemented or broken
  - 🚫 ANTI-PATTERN — present but violates spec
For PARTIAL/ANTI-PATTERN, cite specific file:line and the spec violation.

## 4. Anti-Pattern Scan
Grep the codebase for these known anti-patterns. List every occurrence with 
file:line:
  - time.sleep( and page.wait_for_timeout(  → forbidden, must use web-first waits
  - page.locator('div >') and nth-child       → brittle structural selectors
  - eval( and exec(                            → security risk
  - except: \n    pass                         → bare swallowed exceptions
  - print(  in src\ (excluding logging)       → should be logger
  - hardcoded credentials (regex search for 
    'password\s*=\s*"', 'api_key\s*=\s*"', 'token\s*=\s*"')
  - XPath selectors with // outside of test fixtures
  - direct os.path string concatenation       → should use pathlib

## 5. JSON / Pydantic Output Path
Trace every place the LLM returns structured output:
  - Where is the raw response parsed?
  - Is Pydantic V2 used? (BaseModel with model_config = ConfigDict)
  - Is Instructor library used or manual parsing?
  - List every Pydantic model used for LLM output with file:line
  - Identify the 3-stage repair pipeline if present, note its location

## 6. Existing RAG Pipeline Audit
Document the current state:
  - LanceDB tables that exist (count documents per table)
  - Embedding model in use (name, dimensions)
  - Hybrid search implementation (BM25 weight, semantic weight, fusion method)
  - Document chunking strategy (size, overlap, AST-aware? Y/N)
  - Where are queries built?

## 7. Locator Repository State
  - Count entries in locators\locators.json
  - Schema validation: every entry has {selector, role, healed_count, last_used}?
  - File locking mechanism present? (FileLock yes/no)
  - Atomic write mechanism present? (temp file + rename pattern Y/N)

## 8. Tech Debt — Prioritized List
Number each item 1-N by impact (high impact first). For each:
  - Title
  - Estimated effort (S/M/L)
  - Blocking which Sprint? (Sprint 1 / Sprint 2 / etc.)
  - Brief description

## 9. Sprint 1 Readiness Assessment
Final verdict: Can Sprint 1 (Format fix + DOM Pruner + Action Space) start 
immediately? List any blockers with concrete remediation steps.

================================================================================
DELIVERABLE 2 — golden_dataset\ (JSONL test cases)
================================================================================

Create exactly 3 JSONL files. Each line is one test case.

audit\phase0\golden_dataset\passing.jsonl  (50 entries):
  Each entry has fields:
    case_id, requirement_text, target_url, role, expected_outcome, 
    notes, created_at_iso
  Source: Generate using the EXISTING agent in --mode generate against 
    well-known stable test sites: https://playwright.dev, 
    https://demo.playwright.dev/todomvc, https://the-internet.herokuapp.com
  For each: run generate, then run the generated test, only save to 
    passing.jsonl if it actually passes execution.
  Use a fixed seed by setting OLLAMA temperature=0.0 for reproducibility.

audit\phase0\golden_dataset\failing.jsonl  (50 entries):
  Same schema PLUS additional fields:
    failure_signature (one of: LOCATOR_NOT_FOUND, JSON_PARSE_FAIL, 
      SYNTAX_ERROR, TIMEOUT, WRONG_ASSERTION, AUTH_REQUIRED),
    raw_error_text (the actual exception/stack trace)
  Source: Run generate against the same sites with intentionally vague 
    requirements ("test the thing", "click stuff") and against 
    https://uitestingplayground.com which has known broken elements.

audit\phase0\golden_dataset\flaky.jsonl  (20 entries):
  Same schema PLUS:
    pass_rate_observed (run each test 5 times, record passes / 5),
    suspected_cause (e.g., "race condition on load", "animation timing")
  Source: Tests that pass sometimes but not always over 5 runs.

Build a small driver script: audit\phase0\build_golden_dataset.py
  - Reads target sites from a YAML manifest
  - Calls the existing main.py --mode generate for each
  - Runs each generated test 5 times
  - Categorizes by outcome and writes the JSONL
  - Logs progress to stdout

================================================================================
DELIVERABLE 3 — baseline_metrics.json
================================================================================

Create a script audit\phase0\compute_baseline.py that runs the entire 
golden dataset through the existing agent and produces 
audit\phase0\baseline_metrics.json with this exact schema:

{
  "measured_at_iso": "...",
  "model": "qwen2.5-coder:7b-instruct-q4_K_M",
  "sample_size": {"passing": 50, "failing": 50, "flaky": 20},
  "generation_metrics": {
    "json_parse_failure_rate": 0.0,
    "syntax_error_rate": 0.0,
    "avg_tokens_input": 0,
    "avg_tokens_output": 0,
    "avg_generation_latency_ms": 0,
    "p95_generation_latency_ms": 0
  },
  "execution_metrics": {
    "first_run_pass_rate": 0.0,
    "locator_timeout_rate": 0.0,
    "false_pass_rate_estimated": 0.0,
    "avg_test_runtime_ms": 0
  },
  "healing_metrics": {
    "heal_attempts_avg_per_failed_test": 0.0,
    "heal_success_rate": 0.0,
    "avg_heal_latency_ms": 0
  },
  "resource_metrics": {
    "peak_vram_mb": 0,
    "peak_ram_mb": 0,
    "ollama_oom_events": 0
  },
  "rag_metrics": {
    "total_chunks": 0,
    "avg_retrieval_latency_ms": 0,
    "cache_hit_rate_estimated": 0.0
  }
}

Resource metrics: poll nvidia-smi every 2 seconds during runs, record peaks.
Use psutil for RAM. Use Windows-native paths only.

================================================================================
DELIVERABLE 4 — sprint1_targets.yaml
================================================================================

Based on baseline_metrics.json, write audit\phase0\sprint1_targets.yaml 
with concrete measurable goals:

sprint_1:
  duration_days: 10
  
  format_fix:
    baseline_json_parse_fail_rate: <copy from baseline>
    target_json_parse_fail_rate: 0.05
    method: "Instructor library + Pydantic V2 + Ollama format=json + temp=0.1"
    
  dom_pruner:
    baseline_avg_context_tokens: <copy from baseline>
    target_max_context_tokens: 1000
    method: "Tree-sitter + AOM-first + Prune4Web heuristics"
    
  action_space:
    baseline_invalid_action_rate: "unknown"
    target_invalid_action_rate: 0.02
    method: "Closed enum + JSON schema validator + AST whitelist"

  acceptance_gate:
    must_pass: "all 3 targets met on full golden dataset re-run"
    rollback_trigger: "any regression > 5% on passing.jsonl"

================================================================================
EXECUTION ORDER
================================================================================
Step 1: Read SPEC_CORE.md and CLAUDE.md
Step 2: Walk src\ directory — produce DELIVERABLE 1 sections 1-9
Step 3: Build build_golden_dataset.py — produce DELIVERABLE 2
Step 4: Build compute_baseline.py — produce DELIVERABLE 3
Step 5: Read DELIVERABLE 3 numbers — produce DELIVERABLE 4
Step 6: Add a final section to AUDIT_REPORT.md titled "## 10. Phase 0 Summary"
        listing all 4 deliverables with file paths and a one-line verdict 
        per item.

================================================================================
OUTPUT RULES
================================================================================
- Do not modify any file outside audit\phase0\
- Specify exact file path as comment at top of every script
- All scripts must be runnable standalone with: uv run python audit\phase0\<script>.py
- Use pathlib.Path everywhere, NEVER os.path string concat
- All logging via the existing logger pattern (check src\ for the convention)
- After completion, print a single summary line to stdout:
  "PHASE 0 COMPLETE — see audit\phase0\AUDIT_REPORT.md"