"""
CI report generator.

Reads all audit/sprint*_results.json → produces:
  1. audit/ci/full_report.md   — markdown table of all sprint gates
  2. audit/ci/metrics.json     — AgentMetrics.export() snapshot
  3. audit/ci/audit_verify.txt — CryptoAuditTrail.verify() result

Exits with code 1 if any sprint regression=true or chain verify=False.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_AUDIT_DIR = _ROOT / "audit"
_CI_DIR = _ROOT / "audit" / "ci"
_AUDIT_CHAIN_PATH = Path.home() / ".qa-agent" / "audit_chain.jsonl"


def _collect_sprint_results() -> list[tuple[str, dict]]:
    results = []
    for sprint_dir in sorted(_AUDIT_DIR.glob("sprint*")):
        for result_file in sprint_dir.glob("*_results.json"):
            sprint_name = sprint_dir.name
            data = json.loads(result_file.read_text())
            results.append((sprint_name, data))
    return results


def _generate_markdown_table(sprint_results: list[tuple[str, dict]]) -> str:
    lines = ["# QA Agent Sprint Gate Report", "", "| Sprint | Key | Value |", "|--------|-----|-------|"]
    for sprint_name, data in sprint_results:
        for key, value in data.items():
            lines.append(f"| {sprint_name} | {key} | {value} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    _CI_DIR.mkdir(parents=True, exist_ok=True)

    sprint_results = _collect_sprint_results()

    # 1. Markdown report
    md = _generate_markdown_table(sprint_results)
    (_CI_DIR / "full_report.md").write_text(md)
    print(f"[generate_report] full_report.md written ({len(sprint_results)} sprints)")

    # 2. AgentMetrics snapshot
    try:
        from src.observability.metrics import AgentMetrics
        snapshot = AgentMetrics.instance().export()
    except Exception as exc:
        snapshot = {"error": str(exc)}
    (_CI_DIR / "metrics.json").write_text(json.dumps(snapshot, indent=2))
    print("[generate_report] metrics.json written")

    # 3. Audit chain verification
    chain_valid = True
    try:
        from src.observability.audit_chain import CryptoAuditTrail
        if _AUDIT_CHAIN_PATH.exists():
            trail = CryptoAuditTrail(path=_AUDIT_CHAIN_PATH)
            chain_valid = trail.verify()
    except Exception as exc:
        chain_valid = False
        print(f"[generate_report] audit chain error: {exc}", file=sys.stderr)
    (_CI_DIR / "audit_verify.txt").write_text(
        "VALID" if chain_valid else "INVALID"
    )
    print(f"[generate_report] audit_verify.txt written: {'VALID' if chain_valid else 'INVALID'}")

    # Check exit conditions
    has_regression = any(
        data.get("regression") is True
        for _, data in sprint_results
    )
    if has_regression:
        print("[generate_report] FAIL — regression detected in sprint results", file=sys.stderr)
        sys.exit(1)
    if not chain_valid:
        print("[generate_report] FAIL — audit chain is invalid", file=sys.stderr)
        sys.exit(1)
    print("[generate_report] All gates PASS — exit 0")


if __name__ == "__main__":
    main()
