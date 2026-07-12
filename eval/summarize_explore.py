"""Summarize the explorer/locator track.

Two tables (per the plan):
  * explore: baseline / +SFG(E1) / +Layer1(E2) × {states_avg, reachable_coverage,
    dedup_precision}
  * locator: E1 / E2 × {click_attempts, click_fail, locator_success_rate,
    healed_by_layer1, llm_call_pct} — locator workload = the crawler's real clicks.

Reads whichever result JSONs exist in eval/. Run:
  .venv/bin/python eval/summarize_explore.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPLORE = [
    ("baseline", "explore_baseline.json"),
    ("+SFG (E1)", "explore_sfg.json"),
    ("+Layer1 (E2)", "explore_e2.json"),
]
# locator workload comes from the SFG-mode runs (the crawl's real clicks)
LOCATOR = [
    ("E1 (no heal)", "explore_sfg.json"),
    ("E2 (Layer-1)", "explore_e2.json"),
]


def _load(fn: str) -> dict | None:
    p = ROOT / fn
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "—")


def main() -> None:
    print("## Explore (states / coverage / dedup)\n")
    print("| config | states_avg | reachable_coverage | dedup_precision |")
    print("|---|---|---|---|")
    for label, fn in EXPLORE:
        d = _load(fn)
        if not d:
            continue
        m = d["metrics"]
        print(f"| {label} | {_fmt(m.get('states_discovered_avg'))} | "
              f"{_fmt(m.get('reachable_coverage'))} | {_fmt(m.get('dedup_precision'))} |")

    print("\n## Locator (success rate + Layer-1 share, no LLM)\n")
    print("| config | clicks | fail | success_rate | healed_by_L1 | llm_call_pct |")
    print("|---|---|---|---|---|---|")
    out_loc = []
    for label, fn in LOCATOR:
        d = _load(fn)
        if not d:
            continue
        clicks = fail = healed = 0
        for p in d.get("per_target", []):
            diag = p.get("diagnostics", {}) or {}
            clicks += diag.get("clicks", 0) or 0
            fail += diag.get("click_fail", 0) or 0
            healed += diag.get("healed_l1", 0) or 0
        succ = (clicks - fail) / clicks if clicks else None
        # Layer-1 is deterministic — zero LLM calls by construction (E3 adds Layer-2).
        row = {"label": label, "clicks": clicks, "fail": fail,
               "success_rate": round(succ, 4) if succ is not None else None,
               "healed_by_l1": healed, "llm_call_pct": 0.0}
        out_loc.append(row)
        print(f"| {label} | {clicks} | {fail} | {_fmt(row['success_rate'])} | "
              f"{healed} | {row['llm_call_pct']} |")

    print("\n## Form-filling (post-login traversal)\n")
    print("| config | post_login_states | auth_success_rate | coverage | blocked_submits |")
    print("|---|---|---|---|---|")
    out_form = []
    for label, fn in [("baseline", "form_baseline.json"),
                      ("+fill (F1)", "explore_f1.json"),
                      ("+submit (F2)", "explore_f2.json")]:
        d = _load(fn)
        if not d:
            continue
        m = d["metrics"]
        bs = sum((p.get("diagnostics", {}) or {}).get("blocked_submits", 0) or 0
                 for p in d.get("per_target", []))
        out_form.append({"label": label, "post_login_states": m.get("post_login_states_total"),
                         "auth_success_rate": m.get("auth_success_rate"),
                         "reachable_coverage": m.get("reachable_coverage"), "blocked_submits": bs})
        print(f"| {label} | {m.get('post_login_states_total')} | {_fmt(m.get('auth_success_rate'))} | "
              f"{_fmt(m.get('reachable_coverage'))} | {bs} |")

    summary = {"explore": [], "locator": out_loc, "form": out_form}
    for label, fn in EXPLORE:
        d = _load(fn)
        if d:
            summary["explore"].append({"label": label, **d["metrics"]})
    (ROOT / "explore_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {ROOT / 'explore_summary.json'}")


if __name__ == "__main__":
    main()
