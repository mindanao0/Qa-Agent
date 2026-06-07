## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Mine Sprint 6–14 run artifacts → build training JSONL
SCRIPT: scripts/collect_training_data.py (scaffold exists)
TARGET: data/training/raw_examples.jsonl ≥ 500 examples
GPU: ready (TDR fixed) — dataset collection is CPU-only

## OBJECTIVE
GOAL: สร้าง high-quality (prompt, completion) pairs จาก Sprint logs
      ที่มี quality ≥ 0.70 เท่านั้น สำหรับ QLoRA fine-tune

## STEP 1 — Implement collect_training_data.py

แต่ละ source ให้ implement _collect_{source}() function:

### Source A — Sprint 6: Python test generation
# Location: audit/sprint6/ + src/codetest/generator.py logs
# Mine: EpisodicStore หรือ generation logs ที่บันทึก (prompt, completion)
# Filter: quality=1.0 ถ้า test passed pytest, quality=0.0 ถ้า judge rejected
#
# Prompt format:
#   "Generate a pytest test for the following Python function:\n{func_spec_json}"
# Completion format:
#   "{generated_test_code}"
# Expected yield: ~17 examples (17 tests generated, all passed)

### Source B — Sprint 9: JS/TS test generation
# Location: audit/sprint9/
# Same pattern as Source A but for Vitest
# Prompt: "Generate a Vitest test for the following TypeScript function:\n{spec}"
# Expected yield: ~12 examples

### Source C — Sprint 13: Schema inference
# Location: audit/sprint13/ + src/fuzzer/schema_inferrer.py
# Mine: (TraceRecord list, InferredSchema) pairs
# Prompt: "Infer an OpenAPI schema from these API traces:\n{traces_json}"
# Completion: "{inferred_schema_json}"
# Filter: quality=1.0 if coverage_score ≥ 1, is_candidate=True
# Expected yield: ~5–7 examples (5 schemas inferred)

### Source D — Sprint 14: Invariant extraction
# Location: audit/sprint14/
# Mine: (FunctionSpec, Invariant) pairs
# Prompt: "Extract a testable invariant from this function:\n{func_spec}"
# Completion: "{invariant_json}"
# Filter: quality=1.0 if invariant led to passing Hypothesis test
#         quality=0.5 if invariant defined but no counterexample
# Expected yield: ~12 examples (12 properties defined)

### Source E — Sprint 5: Hypothesis generation
# Location: audit/sprint5/
# Mine: (AX snapshot + URL, TestHypothesis) pairs
# Prompt: "Generate test hypotheses for this web app state:\n{ax_summary}"
# Completion: "{hypothesis_json}"
# Filter: quality=1.0 if hypothesis_pass_rate=1.0
# Expected yield: ~6 examples

### Source F — Augmentation (if total < 500)
# If A+B+C+D+E < 500 examples:
# Use AdvancedVectorGenerator patterns → create synthetic variants
# Vary: func names, arg types, return types (keep structure, change content)
# Mark metadata: {"augmented": true}
# quality = 0.8 (slightly lower than real examples)
# Generate enough to reach 500 total

## STEP 2 — Implement validation

uv run python scripts/validate_dataset.py

# Checks (implement in validate_dataset.py):
#   1. count ≥ 500
#   2. mean quality ≥ 0.70
#   3. source diversity ≥ 3 distinct sources
#   4. 0 duplicate example_ids
#   5. prompt non-empty + completion non-empty
#   6. no BLOCKED_ACTION_PATTERNS in completions
#      (delete/remove/transfer/payment/password)
#   7. no Authorization header values or passwords in any field
#      (redact check — scan for "Bearer ", "password": pattern)
#
# Output: data/training/validation_report.json + print summary

## STEP 3 — Quality split

# After validation, split dataset:
# data/training/train.jsonl  — 90% (≥450 examples)
# data/training/val.jsonl    — 10% (≥50 examples)
#
# Split strategy: stratified by source
# (maintain source distribution across train/val)
# Shuffle with seed=42 before split

## STEP 4 — Dataset stats

Print final summary:
  Total examples:     <int>
  By source:
    sprint6_pytest:   <int>
    sprint9_vitest:   <int>
    sprint13_schema:  <int>
    sprint14_pbt:     <int>
    sprint5_hypothesis: <int>
    augmented:        <int>
  Mean quality:       <float>
  Train split:        <int>
  Val split:          <int>
  Ready for fine-tune: true/false

## RULES:
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid") on TrainingExample
# - NEVER include Authorization/password values in any example
# - quality filter: skip examples with quality < 0.50
# - augmented examples: max 40% of total
#   (ถ้า real examples ≥ 300 → augmented ≤ 200)
# - JSONL format: one JSON object per line, UTF-8
# - shuffle seed=42 (reproducible)
# - validate_dataset.py exit(1) if any check fails

## OUTPUT CONTRACT
DONE WHEN:
- [ ] data/training/raw_examples.jsonl exists, ≥ 500 lines
- [ ] data/training/train.jsonl + val.jsonl created
- [ ] validate_dataset.py exits 0
- [ ] data/training/validation_report.json written

DATASET REPORT format:
DATASET COLLECTION REPORT
=========================
total_examples:     <int>
sources:
  sprint6_pytest:   <int>
  sprint9_vitest:   <int>
  sprint13_schema:  <int>
  sprint14_pbt:     <int>
  sprint5_hypothesis: <int>
  augmented:        <int>
mean_quality:       <float>
train_split:        <int>
val_split:          <int>
validation:         PASS | FAIL
ready_for_finetune: true | false