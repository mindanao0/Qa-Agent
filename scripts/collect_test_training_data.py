"""
Convert Universal QA Agent results to training examples.

Usage:
    python scripts/collect_test_training_data.py \
        --results reports/multi_site/saucedemo_detail.json \
        --site-label saucedemo \
        --output data/training/train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def make_example_id(prompt: str, completion: str) -> str:
    """Content-addressed id.

    The old scheme hashed (site, title, positional idx) — the same test case
    re-collected in a later round sat at a different idx, got a fresh id, and
    was appended again. That is how train.jsonl reached ~49% duplicate rows by
    2026-07-11. Hashing the actual prompt+completion makes dedup-by-id hold
    across rounds (scripts/dedupe_training_data.py rewrites old rows to this
    same scheme).
    """
    raw = f"{prompt}\x00{completion}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def result_to_training_example(
    result: dict,
    site_label: str,
) -> dict | None:
    tc = result.get("test_case", {})
    if not tc.get("steps"):
        return None

    source_url = tc.get("source_url", "")
    element_name = tc.get("title", "")
    # Try to extract element name from title (e.g. "ทดสอบ Add to cart" → "Add to cart")
    for prefix in ("ทดสอบ ", "ตรวจสอบ ", "Test ", "Check "):
        if element_name.startswith(prefix):
            element_name = element_name[len(prefix):]
            break

    # NOTE: the literal word "element" below is part of the frozen prompt format.
    # Existing train.jsonl rows were built with it; changing the wording would give
    # the same logical test case a different content hash and reintroduce dups.
    prompt = (
        f"เขียน test case 1 ข้อสำหรับ element นี้ (JSON ภาษาไทย)\n"
        f"หน้า: {source_url}\n"
        f"Element: element \"{element_name}\"\n\n"
        f"กฎ steps: ภาษาไทย, ชื่อปุ่มใน double quotes\n"
        f"ตัวอย่าง: คลิกปุ่ม \"Add to cart\"\n\n"
        f"JSON: {{\"test_cases\": [{{\"title\": \"...\", "
        f"\"priority\": \"high\"|\"medium\"|\"low\", "
        f"\"preconditions\": [...], \"steps\": [\"...\", ...], "
        f"\"expected_outcome\": \"...\"}}]}}"
    )
    completion = json.dumps({
        "test_cases": [{
            "title": tc.get("title", ""),
            "priority": tc.get("priority", "medium"),
            "preconditions": tc.get("preconditions", []),
            "steps": tc.get("steps", []),
            "expected_outcome": tc.get("expected_outcome", ""),
        }]
    }, ensure_ascii=False)

    quality = 1.0 if result.get("passed") else 0.5

    return {
        "example_id": make_example_id(prompt, completion),
        "source": f"universal_qa_{site_label}",
        "prompt": prompt,
        "completion": completion,
        "quality": quality,
        "metadata": {
            "site": site_label,
            "test_type": tc.get("type", "functional"),
            "passed": result.get("passed", False),
            "duration_ms": result.get("duration_ms", 0),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def collect(results_path: str, site_label: str, output_path: str) -> None:
    results = json.loads(pathlib.Path(results_path).read_text(encoding="utf-8"))
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing example_ids to avoid duplicates
    existing_ids: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing_ids.add(json.loads(line)["example_id"])
                except Exception:
                    pass

    new_count = 0
    with output.open("a", encoding="utf-8") as f:
        for result in results:
            ex = result_to_training_example(result, site_label)
            if ex is None:
                continue
            if ex["example_id"] in existing_ids:
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            existing_ids.add(ex["example_id"])
            new_count += 1

    passed = sum(1 for r in results if r.get("passed"))
    print(f"[{site_label}] {len(results)} results → {new_count} new examples added "
          f"(passed: {passed}/{len(results)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--site-label", required=True)
    parser.add_argument("--output", default="data/training/train.jsonl")
    args = parser.parse_args()
    collect(args.results, args.site_label, args.output)
