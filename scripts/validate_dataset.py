"""Validate raw_examples.jsonl before fine-tuning.

Checks:
  1. Total examples >= 500
  2. quality distribution: mean >= 0.70
  3. source diversity: >= 3 distinct sources
  4. No duplicate example_ids
  5. prompt + completion both non-empty
  6. No BLOCKED_ACTION_PATTERNS in completions

Output: data/training/validation_report.json
Exit 1 if any check fails
"""

from __future__ import annotations

import json
import pathlib
import sys

INPUT_PATH = pathlib.Path("data/training/raw_examples.jsonl")
REPORT_PATH = pathlib.Path("data/training/validation_report.json")

BLOCKED_ACTION_PATTERNS = [
    "delete",
    "remove",
    "transfer",
    "payment",
    "password",
]


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found — run collect_training_data.py first")
        sys.exit(1)

    examples = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    # Check 1: count
    checks["min_500_examples"] = len(examples) >= 500
    details["total_examples"] = len(examples)

    # Check 2: quality mean
    qualities = [ex.get("quality", 0.0) for ex in examples]
    mean_quality = sum(qualities) / len(qualities) if qualities else 0.0
    checks["mean_quality_gte_0_70"] = mean_quality >= 0.70
    details["mean_quality"] = round(mean_quality, 4)

    # Check 3: source diversity
    sources = {ex.get("source", "") for ex in examples}
    checks["source_diversity_gte_3"] = len(sources) >= 3
    details["distinct_sources"] = sorted(sources)

    # Check 4: no duplicate ids
    ids = [ex.get("example_id", "") for ex in examples]
    checks["no_duplicate_ids"] = len(ids) == len(set(ids))
    details["duplicate_count"] = len(ids) - len(set(ids))

    # Check 5: non-empty prompt + completion
    empty_count = sum(
        1 for ex in examples
        if not ex.get("prompt", "").strip() or not ex.get("completion", "").strip()
    )
    checks["no_empty_pairs"] = empty_count == 0
    details["empty_pairs"] = empty_count

    # Check 6: no blocked patterns in completions
    blocked_hits = []
    for ex in examples:
        completion = ex.get("completion", "")
        for pattern in BLOCKED_ACTION_PATTERNS:
            if pattern.lower() in completion.lower():
                blocked_hits.append({"example_id": ex.get("example_id"), "pattern": pattern})
    checks["no_blocked_patterns"] = len(blocked_hits) == 0
    details["blocked_pattern_hits"] = blocked_hits

    # Check 7: no Authorization header values or passwords in any field
    import re as _re
    auth_hits = []
    for ex in examples:
        for field_name in ("prompt", "completion"):
            field_val = ex.get(field_name, "")
            if "Bearer " in field_val:
                auth_hits.append({
                    "example_id": ex.get("example_id"),
                    "field": field_name,
                    "pattern": "Bearer ",
                })
            if _re.search(r'"password"\s*:\s*"[^"]+', field_val):
                auth_hits.append({
                    "example_id": ex.get("example_id"),
                    "field": field_name,
                    "pattern": "password value",
                })
    checks["no_auth_or_password_values"] = len(auth_hits) == 0
    details["auth_hits"] = auth_hits

    all_pass = all(checks.values())
    report = {"passed": all_pass, "checks": checks, "details": details}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nReport written to {REPORT_PATH}")

    if not all_pass:
        print("VALIDATION FAILED")
        sys.exit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
