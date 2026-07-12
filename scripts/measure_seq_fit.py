"""Measure how much of train.jsonl survives the trainer's max_seq_length filter.

The 7B preset uses seq=256 and prepare_dataset DROPS any example whose
chat-templated text exceeds it (truncating a completion would teach the model
to emit truncated JSON — worse than dropping). This script reports the real
kept-% at several seq lengths so seq experiments are decided from data, not
stale comments (a 2026-06 comment claimed 75% kept; measured June-27 was ~36%).

Run with the finetune venv (needs transformers + the cached Qwen tokenizer):
    ~/.qa-finetune-env/bin/python scripts/measure_seq_fit.py [--file PATH]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

CANDIDATE_SEQ = [256, 320, 384, 512]
BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/training/train.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # slow import — keep after argparse

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    path = pathlib.Path(args.file)
    lengths: list[int] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if rec.get("messages"):
            messages = rec["messages"]
        elif rec.get("prompt") and rec.get("completion"):
            messages = [
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["completion"]},
            ]
        else:
            skipped += 1
            continue
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        lengths.append(len(tok(text, add_special_tokens=False).input_ids))

    if not lengths:
        print("no measurable rows", file=sys.stderr)
        return 1

    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        return lengths[min(n - 1, int(p * n))]

    print(f"{path}: rows={n} skipped={skipped}")
    print(f"token lengths: min={lengths[0]} p50={pct(0.50)} p75={pct(0.75)} "
          f"p90={pct(0.90)} p99={pct(0.99)} max={lengths[-1]}")
    for seq in CANDIDATE_SEQ:
        fit = sum(1 for x in lengths if x <= seq)
        print(f"  seq={seq:>4}: fits {fit}/{n} ({100.0 * fit / n:.1f}%)")
    print("note: VRAM cost rises with seq — any seq change needs a short "
          "validation run (activation memory) before a long run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
