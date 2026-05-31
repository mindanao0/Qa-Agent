Sprint 1 Day 3-5 wrote code but did not measure. The acceptance gate is unverified. 
Before declaring Sprint 1 done and proceeding to Sprint 2, execute this 3-cluster 
mini-sprint to close the loop properly.

Read FIRST:
  1. audit\phase0\SPRINT1_DAY5_LOG.md           (what was claimed)
  2. audit\phase0\sprint1_day2.5_results.json   (Gate A verified numbers)
  3. audit\phase0\measure_sprint1_day5.py       (the unrun measurement script)
  4. src\perception\semantic_compactor.py       (TD-13 location — token estimation)
  5. src\perception\grounder.py                 (budget enforcement caller)

================================================================================
WHY THIS MATTERS
================================================================================
The Day 3-5 work claims Gate E "deferred" while marking the sprint complete. 
This is incorrect — a deferred gate means the sprint is NOT complete. The 
measurement is the only thing that proves DOM Pruner actually does what it 
was built to do. Without it, we cannot proceed to Sprint 2 with confidence.

TD-13 is also a blocker: if our token estimator under-counts CJK content by 4×, 
the 1000-token "hard cap" silently fails on Thai/Chinese/Japanese pages — every 
non-English target site becomes a stealth context overflow. Per Q3 ("ทุกเว็บที่
เป็นไปได้"), this is a universality blocker, not a polish task.

================================================================================
CLUSTER M1 — Fix TD-13 (Token Estimation) — ~1 hour
================================================================================
Subagent context: 
  - src\perception\semantic_compactor.py (find the estimator)
  - src\perception\grounder.py (find the budget retry loop 25→15→10)

Deliverables:
  M1.1. Install tiktoken: uv add tiktoken
  
  M1.2. Replace `len(text)//4` with a real tokenizer call. Create a small 
        utility module src\perception\token_counter.py:
        
        from functools import lru_cache
        import tiktoken
        
        @lru_cache(maxsize=4)
        def _get_encoder(model: str = "cl100k_base"):
            return tiktoken.get_encoding(model)
        
        def estimate_tokens(text: str, encoding: str = "cl100k_base") -> int:
            return len(_get_encoder(encoding).encode(text))
        
        Caveat: tiktoken's cl100k_base is OpenAI's tokenizer; qwen2.5-coder uses 
        a different tokenizer (qwen tokenizer). cl100k_base over-estimates Qwen 
        tokens by ~10-15% which is SAFE (conservative). DO NOT switch to qwen 
        tokenizer — it's not in tiktoken stdlib and adding the qwen tokenizer is 
        out of scope. Document this 10-15% safety margin in the docstring.
  
  M1.3. Update SemanticCompactor.compact() and all callers in grounder.py to 
        use the new estimate_tokens(). Remove the old len/4 logic entirely.
  
  M1.4. Add tests\test_token_counter.py with 3 cases:
        [ ] test_english_text_estimate — "hello world" → 2 tokens
        [ ] test_thai_text_estimate    — "สวัสดี" → assert > len("สวัสดี")//4 
                                           (proves CJK fix works)
        [ ] test_caching               — call twice same text, assert encoder 
                                           created only once (mock _get_encoder)
  
  M1.5. Re-run the existing tests\test_grounder.py budget enforcement test 
        with the new estimator. Verify it still works (might need to adjust 
        the test fixture if it relied on the wrong estimate).

GATE M1 (must pass before M2):
  [ ] All new + existing perception tests pass: 
      uv run pytest tests\test_grounder.py tests\test_token_counter.py -v
  [ ] No call to `len(...)//4` for token counting anywhere in src\perception\:
      verify with: grep -rn "len.*//.*4" src\perception\
      (should return 0 hits, or only unrelated math)

================================================================================
CLUSTER M2 — Run Gate E Measurement — ~2 hours
================================================================================
Subagent context:
  - audit\phase0\measure_sprint1_day5.py
  - audit\phase0\golden_dataset\*.jsonl (whatever's available — pilot is fine)
  - Live Ollama server (verify it's running first)

Deliverables:
  M2.1. Pre-flight check:
        - curl http://localhost:11434/api/tags → confirm Ollama up + qwen2.5-coder 
          loaded
        - check VRAM headroom: nvidia-smi → assert >= 5GB free for the run
        - count golden dataset entries: 
          wc -l audit\phase0\golden_dataset\*.jsonl
        - If pilot < 15 entries total, that's fine — run on what's available
          and document dataset_size in the result file
  
  M2.2. Execute the measurement:
        uv run python audit\phase0\measure_sprint1_day5.py
        
        Capture stdout AND stderr to audit\phase0\sprint1_day5_measurement.log
        
        Expected runtime: 10 entries × 2 modes × ~70s/gen ≈ 25 minutes. 
        If it takes > 90 minutes, something is wrong — kill and investigate.
  
  M2.3. Verify the output:
        - audit\phase0\sprint1_day5_results.json must exist
        - Must contain both `with_grounder` and `without_grounder` blocks
        - Must contain `delta` with non-zero values (zero deltas = silent bug)
        - `verdict` must be one of PASS / FAIL / REGRESSION
  
  M2.4. Sanity-check the numbers against expectations:
        - with_grounder.avg_context_tokens should be in range 300-1100 
          (1000 cap with 10-15% tiktoken safety margin)
        - without_grounder.avg_context_tokens should be much larger 
          (5,000-50,000 range typical for raw DOM)
        - context_tokens_reduction_pct should be 80-99%
        
        If with_grounder.avg_context_tokens > 1500: budget enforcement is 
        broken — STOP and report.

GATE M2:
  [ ] sprint1_day5_results.json exists with non-zero deltas
  [ ] No VRAM OOM in the measurement log
  [ ] verdict field populated

================================================================================
CLUSTER M3 — Verdict + Sprint 1 Final Decision — ~1 hour
================================================================================
Subagent context: M1 + M2 outputs

Deliverables:
  M3.1. Read sprint1_day5_results.json. Compute the verdict per ORIGINAL 
        Gate E criteria (do not lower the bar):
        
        PASS iff ALL of:
          - with_grounder.avg_context_tokens ≤ 1000
          - with_grounder.p95_context_tokens ≤ 1500
          - with_grounder.first_run_pass_rate ≥ 0.85
          - with_grounder.avg_generation_latency_ms < 71,489
        
        FAIL (not regression) iff:
          - context_tokens met but latency/pass_rate slightly off
        
        REGRESSION iff:
          - first_run_pass_rate < 0.50
  
  M3.2. Write audit\phase0\SPRINT1_FINAL_LOG.md — this REPLACES (not appends to) 
        the premature SPRINT1_DAY5_LOG.md claim. Structure:
        
        # Sprint 1 — Final Report
        
        ## Verdict: <PASS|FAIL|REGRESSION>
        
        ## Journey Recap
        | Phase   | What was done                          | Status     |
        | Day 1-2 | Instructor + Pydantic V2               | ✓ verified |
        | Day 2.5 | action validator + max_retries=1       | ✓ verified |
        | Day 3-5 | DOM Pruner + AOM + Grounder            | ✓ built    |
        | Day 5+  | TD-13 fix + Gate E measurement         | <result>   |
        
        ## Final Numbers (live Ollama, dataset_size=N)
        <full table from sprint1_day5_results.json>
        
        ## What Worked
        ## What Didn't
        ## Tech Debt Carried to Sprint 2
        - TD-12 Shadow DOM (still open)
        - TD-14 BBox population (still open)
        - <any new TDs from this measurement>
        
        ## Sprint 2 Readiness
        Either: "✅ Approved — proceed with Self-Healing"
        Or:     "⚠️  Conditional — fix <X> first"
        Or:     "❌ Blocked — rollback perception, investigate"
  
  M3.3. Based on verdict, take exactly ONE of these actions:
  
        IF PASS:
          - Update CLAUDE.md: mark Sprint 1 complete with the actual numbers
          - Append to docs\specs\SPEC_CORE.md "Sprint 1 acceptance verified on <date>"
          - Print: "SPRINT 1 COMPLETE — verdict: PASS — ready for Sprint 2"
        
        IF FAIL (not regression):
          - Keep perception.use_grounder: true (it's better than nothing)
          - Document the gap clearly in SPRINT1_FINAL_LOG.md "What Didn't"
          - Print: "SPRINT 1 PARTIAL — verdict: FAIL — Sprint 2 conditional"
          - DO NOT proceed to Sprint 2 prompt request automatically — wait for 
            user decision on whether to spend time fixing or accept the gap
        
        IF REGRESSION:
          - Edit config\agent.yaml: perception.use_grounder: false
          - Run a quick smoke test to confirm the legacy path still works
          - Add TD-15 to SPRINT1_FINAL_LOG.md with full root cause
          - Print: "SPRINT 1 REGRESSION — perception disabled — needs root cause"

================================================================================
ABSOLUTE RULES
================================================================================
- Do NOT lower the Gate E thresholds to make it pass. The thresholds are the 
  whole point of measurement.
- Do NOT skip M1 to get to M2 faster. Bad estimator = bad measurement.
- Do NOT write the FINAL log before M2 is verified — no premature victory laps.
- Do NOT touch Sprint 2 territory (self-healing, episodic memory, ContractSkill).
- Do NOT modify any of the perception module logic except token estimation 
  in M1 (that's the one allowed change).

================================================================================
ESCALATION
================================================================================
- If Ollama is down: report immediately, ask user to start it. Do not stub.
- If golden dataset has < 5 entries total: report and ask user to extend, OR 
  proceed with a documented n=N caveat in the final log.
- If measurement reveals avg_context_tokens > 2000: this means budget 
  enforcement is broken. STOP, do not write a "FAIL" verdict — investigate 
  and report root cause first.
- If M2 takes > 90 min wall time: kill, report, ask for go/no-go.

================================================================================
BEGIN with M1. Report at each gate.