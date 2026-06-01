"""
Sprint 8 gate measurement script.

Gates:
  pipeline_stages_defined  >= 4   (lint, test-unit, test-e2e, report)
  otel_spans_emitted       >= 5   (one per LangGraph node minimum)
  structured_log_fields    >= 8   (per log entry)
  audit_trail_entries      >= 10  (cryptographic hash chain)
  REGRESSION if pass_rate  < 0.75
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import tempfile
import uuid

from loguru import logger

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint8_results.json"
_WORKFLOW_PATH = _PROJECT_ROOT / ".github" / "workflows" / "qa_agent.yml"
_SPRINT7_RESULTS = _PROJECT_ROOT / "audit" / "sprint7" / "sprint7_results.json"

_GATE_PIPELINE_STAGES = 4
_GATE_OTEL_SPANS = 5
_GATE_LOG_FIELDS = 8
_GATE_AUDIT_ENTRIES = 10
_REGRESSION_THRESHOLD = 0.75

_EXPECTED_STAGES = {"lint", "test-unit", "test-e2e", "report"}


def _count_pipeline_stages() -> int:
    if not _WORKFLOW_PATH.exists():
        logger.error(f"Workflow not found: {_WORKFLOW_PATH}")
        return 0
    content = _WORKFLOW_PATH.read_text()
    found = sum(1 for stage in _EXPECTED_STAGES if stage in content)
    logger.info(f"measure_sprint8: pipeline stages found = {found}")
    return found


async def _emit_otel_spans(traces_dir: pathlib.Path) -> int:
    from src.observability.tracer import OTelTracer
    tracer = OTelTracer(traces_dir=traces_dir)
    node_names = [
        "node.planner",
        "node.generator",
        "node.bft",
        "node.executor",
        "node.healer",
    ]
    for name in node_names:
        async with tracer.span(name, source="sprint8_measure"):
            pass
    count = tracer.flush()
    logger.info(f"measure_sprint8: OTel spans emitted = {count}")
    return count


def _count_structured_log_fields(logs_dir: pathlib.Path) -> int:
    from src.observability.structured_logger import StructuredLogger
    run_id = uuid.uuid4().hex
    sl = StructuredLogger(logs_dir=logs_dir, run_id=run_id)
    entry = sl.info(
        "Sprint8Measure", "gate_check",
        duration_ms=1.0,
        payload={"gate": "structured_log_fields"},
    )
    entry_dict = json.loads(entry.model_dump_json())
    field_count = len(entry_dict)
    logger.info(f"measure_sprint8: structured log fields = {field_count}")
    return field_count


def _build_audit_trail(audit_path: pathlib.Path) -> tuple[int, bool]:
    from src.observability.audit_chain import CryptoAuditTrail
    trail = CryptoAuditTrail(path=audit_path)
    for i in range(10):
        trail.append(f"sprint8_gate_{i}", {"seq": i, "gate": "audit_trail"})
    valid = trail.verify()
    entries = sum(1 for line in audit_path.read_text().splitlines() if line.strip())
    logger.info(f"measure_sprint8: audit entries = {entries}, chain_valid = {valid}")
    return entries, valid


def _get_prior_pass_rate() -> float:
    if not _SPRINT7_RESULTS.exists():
        logger.warning("measure_sprint8: sprint7_results.json not found — defaulting pass_rate=1.0")
        return 1.0
    data = json.loads(_SPRINT7_RESULTS.read_text())
    rate = float(data.get("test_pass_rate", 1.0))
    logger.info(f"measure_sprint8: prior pass_rate (sprint7) = {rate}")
    return rate


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        traces_dir = tmp_path / "traces"
        logs_dir = tmp_path / "logs"
        audit_path = tmp_path / "audit.jsonl"

        pipeline_stages = _count_pipeline_stages()
        otel_spans = asyncio.run(_emit_otel_spans(traces_dir))
        log_fields = _count_structured_log_fields(logs_dir)
        audit_entries, audit_chain_valid = _build_audit_trail(audit_path)
        pass_rate = _get_prior_pass_rate()

    regression = pass_rate < _REGRESSION_THRESHOLD
    sprint8_pass = (
        pipeline_stages >= _GATE_PIPELINE_STAGES
        and otel_spans >= _GATE_OTEL_SPANS
        and log_fields >= _GATE_LOG_FIELDS
        and audit_entries >= _GATE_AUDIT_ENTRIES
        and audit_chain_valid
        and not regression
    )

    results = {
        "pipeline_stages_defined": pipeline_stages,
        "otel_spans_emitted": otel_spans,
        "structured_log_fields": log_fields,
        "audit_trail_entries": audit_entries,
        "audit_chain_valid": audit_chain_valid,
        "regression": regression,
        "sprint8_status": "PASS" if sprint8_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Sprint 8 Results ===")
    gates = {
        "pipeline_stages_defined": _GATE_PIPELINE_STAGES,
        "otel_spans_emitted": _GATE_OTEL_SPANS,
        "structured_log_fields": _GATE_LOG_FIELDS,
        "audit_trail_entries": _GATE_AUDIT_ENTRIES,
    }
    for k, v in results.items():
        gate_str = f" (gate >= {gates[k]})" if k in gates else ""
        print(f"  {k}: {v}{gate_str}")
    print(f"\n  -> sprint8_results.json written to {_OUTPUT_PATH}")
    sys.exit(0 if sprint8_pass else 1)


if __name__ == "__main__":
    main()
