"""Dedupe data/training/train.jsonl by (prompt, completion) content.

Why this exists: collect_test_training_data.make_example_id used to hash the
positional idx, so the same test case re-collected in a later round got a fresh
id and was appended again — by 2026-07-11 train.jsonl had 7,158 rows but only
~3,641 unique prompt+completion pairs (~49% duplicates). The collector is fixed
to content-hash ids; this script brings the EXISTING file onto the same scheme.

Behaviour:
  * key = sha256(prompt + NUL + completion)[:16]  (same as the fixed collector)
  * keeps ONE row per key — the highest-quality occurrence wins (a later run
    where the test passed beats an earlier 0.5-quality copy), original order
    is preserved by first appearance
  * rewrites example_id to the content hash
  * writes <file>.bak before touching anything; idempotent (2nd run: 0 removed)

Usage:
    python scripts/dedupe_training_data.py [--file data/training/train.jsonl] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys


def content_id(prompt: str, completion: str) -> str:
    raw = f"{prompt}\x00{completion}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/training/train.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="report stats only, write nothing")
    args = ap.parse_args()

    path = pathlib.Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    kept: dict[str, dict] = {}          # cid -> record (best quality)
    order: list[str] = []               # first-appearance order
    total = bad = dups = upgraded = 0

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(f"  skip line {lineno}: invalid JSON", file=sys.stderr)
                continue
            prompt = rec.get("prompt")
            completion = rec.get("completion")
            if not prompt or not completion:
                # keep unknown-shape rows untouched (keyed by raw line so they
                # dedupe only against exact copies of themselves)
                prompt, completion = line, ""
            cid = content_id(prompt, completion)
            if cid in kept:
                dups += 1
                if rec.get("quality", 0) > kept[cid].get("quality", 0):
                    rec["example_id"] = cid
                    kept[cid] = rec
                    upgraded += 1
                continue
            rec["example_id"] = cid
            kept[cid] = rec
            order.append(cid)

    print(
        f"{path}: total={total} unique={len(order)} removed_dups={dups} "
        f"(quality-upgraded {upgraded}) bad_lines={bad}"
    )
    if args.dry_run:
        print("dry-run — nothing written")
        return 0
    if dups == 0 and bad == 0:
        print("already clean — nothing to do")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"backup written: {backup}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for cid in order:
            fh.write(json.dumps(kept[cid], ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"rewritten: {path} ({len(order)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
