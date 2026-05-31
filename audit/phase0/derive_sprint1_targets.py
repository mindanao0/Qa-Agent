# audit/phase0/derive_sprint1_targets.py
"""
Phase 0 — Sprint 1 target derivation.

Reads ``audit/phase0/baseline_metrics.json`` and emits
``audit/phase0/sprint1_targets.yaml`` in the exact schema declared in the
Phase 0 spec (with baseline values copied directly from the JSON).

Targets that are not derivable from the baseline JSON (because the agent
does not currently report them — e.g. invalid_action_rate) are set to the
spec-required default and tagged with a comment.

Standalone usage::

    uv run python audit/phase0/derive_sprint1_targets.py

Honours Phase 0 hard constraints:
  * pathlib.Path everywhere
  * read-only against src/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "audit" / "phase0"
BASELINE_IN = AUDIT_DIR / "baseline_metrics.json"
TARGETS_OUT = AUDIT_DIR / "sprint1_targets.yaml"


def _load_baseline() -> dict:
    if not BASELINE_IN.exists():
        raise FileNotFoundError(
            f"{BASELINE_IN} not found — run compute_baseline.py first"
        )
    return json.loads(BASELINE_IN.read_text(encoding="utf-8"))


def main() -> int:
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    baseline = _load_baseline()
    gen = baseline.get("generation_metrics", {})
    exe = baseline.get("execution_metrics", {})
    heal = baseline.get("healing_metrics", {})
    res = baseline.get("resource_metrics", {})
    sample = baseline.get("sample_size", {})

    baseline_json_parse_fail_rate = float(gen.get("json_parse_failure_rate", 0.0))
    baseline_syntax_error_rate = float(gen.get("syntax_error_rate", 0.0))
    baseline_avg_context_tokens = int(gen.get("avg_tokens_input", 0))
    baseline_first_run_pass_rate = float(exe.get("first_run_pass_rate", 0.0))
    baseline_locator_timeout_rate = float(exe.get("locator_timeout_rate", 0.0))
    baseline_avg_test_runtime_ms = float(exe.get("avg_test_runtime_ms", 0.0))
    baseline_heal_success = float(heal.get("heal_success_rate", 0.0))
    baseline_peak_vram_mb = int(res.get("peak_vram_mb", 0))

    doc = {
        "sprint_1": {
            "duration_days": 10,
            "_baseline_source": str(BASELINE_IN.relative_to(PROJECT_ROOT)),
            "_baseline_sample_size": sample,
            "_baseline_measured_at_iso": baseline.get("measured_at_iso", ""),

            "format_fix": {
                "baseline_json_parse_fail_rate": baseline_json_parse_fail_rate,
                "baseline_syntax_error_rate": baseline_syntax_error_rate,
                "target_json_parse_fail_rate": 0.05,
                "method": (
                    "Instructor library + Pydantic V2 ConfigDict(strict=True,extra='forbid') "
                    "+ Ollama format=json + temp=0.1 + 3-stage repair fallback "
                    "(strip-fences → _repair_json → schema-specific regex)"
                ),
                "blocking_techdebt": ["TD-1", "TD-2", "TD-3"],
                "owner": "tbd",
            },

            "dom_pruner": {
                "baseline_avg_context_tokens": baseline_avg_context_tokens,
                "target_max_context_tokens": 1000,
                "method": (
                    "Tree-sitter Python+HTML grammars + AOM-first (extend "
                    "src/browser/ax_extractor.prune_axtree) + Prune4Web "
                    "heuristics (collapse decorative containers, drop "
                    "aria-hidden, score-sort by interactive role density)"
                ),
                "blocking_techdebt": ["TD-4"],
                "owner": "tbd",
            },

            "action_space": {
                # Spec: 'baseline_invalid_action_rate: "unknown"' until we
                # have a positive action enum to measure against.
                "baseline_invalid_action_rate": "unknown",
                "target_invalid_action_rate": 0.02,
                "method": (
                    "Closed Pydantic Union enum of allowed actions "
                    "(Click | Fill | GoTo | ExpectVisible | ExpectText | "
                    "ExpectURL | Press | Select) + JSON schema validator on "
                    "PlaywrightScript.code + AST whitelist of permitted "
                    "call sites (extend src/agents/generator.validate_playwright_ast)"
                ),
                "blocking_techdebt": ["TD-5"],
                "owner": "tbd",
            },

            "secondary_metrics_observed": {
                "first_run_pass_rate": baseline_first_run_pass_rate,
                "locator_timeout_rate": baseline_locator_timeout_rate,
                "avg_test_runtime_ms": baseline_avg_test_runtime_ms,
                "heal_success_rate": baseline_heal_success,
                "peak_vram_mb": baseline_peak_vram_mb,
            },

            "acceptance_gate": {
                "must_pass": (
                    "all 3 targets met on full golden dataset re-run "
                    "(passing.jsonl=50 + failing.jsonl=50 + flaky.jsonl=20)"
                ),
                "rollback_trigger": (
                    "any regression > 5% on passing.jsonl OR "
                    "json_parse_failure_rate > baseline OR "
                    "first_run_pass_rate < baseline"
                ),
                "verification_steps": [
                    "uv run python audit/phase0/build_golden_dataset.py --resume",
                    "uv run python audit/phase0/compute_baseline.py",
                    "uv run python audit/phase0/derive_sprint1_targets.py",
                    "diff sprint1_targets.yaml against pre-sprint snapshot",
                ],
            },
        }
    }

    TARGETS_OUT.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, indent=2),
        encoding="utf-8",
    )
    logger.info(f"SPRINT1_TARGETS_WRITTEN -> {TARGETS_OUT}")
    print(TARGETS_OUT.read_text(encoding="utf-8"))
    # Phase 0 spec demands this exact final stdout line on completion:
    print("PHASE 0 COMPLETE — see audit\\phase0\\AUDIT_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
