"""Summarize the eval track: baseline / +grammar / +few-shot / +RAG.

Reads the per-config result JSONs produced by run_eval.py and prints the 4-row
comparison table (3 metrics + delta vs baseline) the plan asks for, plus a
machine-readable eval/summary.json. Only the configs whose files exist are shown,
so it is usable after each phase as well as at the end.

Run:  .venv/bin/python eval/summarize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (label, filename) in phase order
CONFIGS = [
    ("baseline", "baseline.json"),
    ("+grammar", "phase1_grammar.json"),
    ("+few-shot", "phase2_fewshot.json"),
    ("+RAG", "phase3_rag.json"),
]
METRICS = ["json_valid_rate", "coverage_score", "assertion_quality"]


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _delta(v, base) -> str:
    if not isinstance(v, (int, float)) or not isinstance(base, (int, float)):
        return ""
    d = v - base
    sign = "+" if d >= 0 else ""
    return f" ({sign}{d:.4f})"


def main() -> None:
    loaded = []
    for label, fname in CONFIGS:
        p = ROOT / fname
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            loaded.append((label, data.get("metrics", {}), data.get("diagnostics", {})))

    if not loaded:
        sys.exit("no result JSONs found in eval/ — run run_eval.py first")

    base_metrics = loaded[0][1]

    # ── markdown table ──
    header = "| config | " + " | ".join(METRICS) + " |"
    sep = "|" + "---|" * (len(METRICS) + 1)
    print(header)
    print(sep)
    for label, metrics, _ in loaded:
        cells = []
        for m in METRICS:
            v = metrics.get(m)
            d = _delta(v, base_metrics.get(m)) if label != loaded[0][0] else ""
            cells.append(_fmt(v) + d)
        print(f"| {label} | " + " | ".join(cells) + " |")

    # ── gate read-out ──
    print("\nGates:")
    by_label = {lbl: (m, diag) for lbl, m, diag in loaded}
    if "+grammar" in by_label:
        m = by_label["+grammar"][0]
        jv = m.get("json_valid_rate")
        cov_ok = m.get("coverage_score", 0) >= base_metrics.get("coverage_score", 0) - 1e-9
        aq_ok = m.get("assertion_quality", 0) >= base_metrics.get("assertion_quality", 0) - 1e-9
        ok = isinstance(jv, (int, float)) and jv >= 0.98 and cov_ok and aq_ok
        print(f"  Phase 1 (grammar): json_valid>=0.98 & no coverage/assertion regression -> "
              f"{'PASS' if ok else 'FAIL'} (json_valid={_fmt(jv)})")
    if "+few-shot" in by_label:
        m = by_label["+few-shot"][0]
        ok = m.get("assertion_quality", 0) > base_metrics.get("assertion_quality", 0)
        print(f"  Phase 2 (few-shot): assertion_quality > baseline -> "
              f"{'PASS' if ok else 'FAIL'} ({_fmt(m.get('assertion_quality'))} vs {_fmt(base_metrics.get('assertion_quality'))})")
    if "+RAG" in by_label:
        m = by_label["+RAG"][0]
        ok = m.get("coverage_score", 0) > base_metrics.get("coverage_score", 0)
        print(f"  Phase 3 (RAG): coverage_score > baseline -> "
              f"{'PASS' if ok else 'FAIL'} ({_fmt(m.get('coverage_score'))} vs {_fmt(base_metrics.get('coverage_score'))})")

    # ── machine-readable ──
    out = {
        "configs": [
            {"label": lbl, "metrics": m, "diagnostics": diag}
            for lbl, m, diag in loaded
        ],
        "baseline_metrics": base_metrics,
    }
    (ROOT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
