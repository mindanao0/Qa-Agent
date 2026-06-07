# Sprint 8: CI/CD GitHub Actions + Observability (Structured Logging + Metrics)
# Project: D:\Code\qa-agent
# Baseline: all sprints green, pass_rate=1.00
# Goal: Ship a working GitHub Actions pipeline + OpenTelemetry-compatible
#       structured logging so every agent run is traceable end-to-end

## ACCEPTANCE GATE (sprint8)
#   pipeline_stages_defined    ≥ 4     (lint, test-unit, test-e2e, report)
#   otel_spans_emitted         ≥ 5     (one per LangGraph node minimum)
#   structured_log_fields      ≥ 8     (per log entry)
#   audit_trail_entries        ≥ 10    (cryptographic hash chain)
#   REGRESSION if pass_rate    < 0.75

## ARCHITECTURE — New files:

### .github/workflows/qa_agent.yml
# GitHub Actions pipeline — 4 stages:
#
# Stage 1: lint
#   - ruff check src/ audit/
#   - pyright src/ (strict mode)
#
# Stage 2: test-unit
#   - uv run pytest tests/unit/ -v --tb=short --timeout=60
#
# Stage 3: test-e2e
#   - Install Playwright browsers: uv run playwright install chromium
#   - uv run python audit/sprint7/measure_sprint7.py --live
#   - uv run python audit/sprint6/measure_sprint6.py --live
#
# Stage 4: report
#   - uv run python audit/ci/generate_report.py
#   - Upload sprint*_results.json as artifacts
#   - Post summary to GitHub Step Summary ($GITHUB_STEP_SUMMARY)
#
# Rules:
#   - runs-on: ubuntu-latest (CI is Linux; Windows dev is local only)
#   - Python: 3.11 (uv managed)
#   - Ollama service container: ollama/ollama:latest on port 11434
#   - Pull qwen2.5-coder:7b-instruct-q4_K_M in a setup step
#   - Cache: uv pip cache + playwright browsers via actions/cache
#   - Fail-fast: false (collect all stage results before failing)
#   - Timeout per job: 20 minutes

### src/observability/tracer.py
# OTelTracer — OpenTelemetry-compatible span emitter (no external collector needed)
# Uses opentelemetry-sdk (local export to JSONL file)
#
# class OTelTracer:
#     """No required constructor args. Lazy-init SDK on first use."""
#
#     def span(self, name: str, **attrs) -> contextlib.AbstractContextManager:
#         """
#         Usage:
#             async with tracer.span("llm.generate", model="qwen2.5-coder"):
#                 result = await ollama_call(...)
#         Emits: trace_id, span_id, parent_span_id, name, start_ms, end_ms, attrs
#         Writes to: logs/traces/otel_{date}.jsonl (pathlib.Path, append mode)
#         """
#
#     def flush(self) -> int:
#         """Returns count of spans flushed to disk."""

### src/observability/structured_logger.py
# StructuredLogger — JSON log entries with ≥8 fields per entry
#
# Required fields per entry (exactly these, ConfigDict extra="forbid"):
# class LogEntry(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     ts:          float   # Unix timestamp
#     level:       Literal["DEBUG","INFO","WARN","ERROR"]
#     component:   str     # e.g. "SFGCrawler", "PytestGenerator"
#     event:       str     # short snake_case description
#     trace_id:    str     # from OTelTracer active span
#     span_id:     str
#     run_id:      str     # UUID per agent invocation
#     duration_ms: float | None
#     payload:     dict    # arbitrary extra context (not forbidden — inner dict)
#
# Writes to: logs/structured/run_{run_id}.jsonl
# Must flush on every entry (no buffering — audit trail integrity)

### src/observability/audit_chain.py
# CryptoAuditTrail — append-only hash chain for tamper-evidence
#
# class AuditEntry(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     seq:         int
#     ts:          float
#     event:       str
#     payload_hash: str   # sha256 of json.dumps(payload, sort_keys=True)
#     prev_hash:   str    # sha256 of previous AuditEntry JSON ("GENESIS" for seq=0)
#     chain_hash:  str    # sha256 of (payload_hash + prev_hash)
#
# class CryptoAuditTrail:
#     def __init__(self, path: pathlib.Path) -> None: ...
#
#     def append(self, event: str, payload: dict) -> AuditEntry:
#         """Compute chain_hash, append to path as JSONL, return entry."""
#
#     def verify(self) -> bool:
#         """Re-compute all chain_hashes from disk. Return False if any broken."""

### src/observability/metrics.py
# AgentMetrics — lightweight in-process counter/histogram (no Prometheus needed)
#
# class AgentMetrics:
#     """Singleton. No required constructor args."""
#     _instance = None
#
#     def increment(self, key: str, value: int = 1) -> None: ...
#     def record(self, key: str, value: float) -> None:
#         """Histogram bucket: append to list, compute p50/p95/p99 on export."""
#     def export(self) -> dict:
#         """Returns snapshot: {key: {count|p50|p95|p99}} — no external calls."""
#     def reset(self) -> None: ...

### audit/ci/generate_report.py
# Reads all sprint*_results.json → produces:
#   1. audit/ci/full_report.md   — markdown table of all sprint gates
#   2. audit/ci/metrics.json     — AgentMetrics.export() snapshot
#   3. audit/ci/audit_verify.txt — CryptoAuditTrail.verify() result
# Exits with code 1 if any sprint regression=true or verify=False

### Integration — wrap ALL LangGraph nodes with OTelTracer.span()
# In every existing node function (crawl_node, hypothesis_node, generate_test, etc.):
#
#   async with tracer.span("node.crawl", url=start_url):
#       result = await _do_crawl(...)
#   structured_logger.info("crawl_complete", duration_ms=..., nodes=...)
#   audit_trail.append("crawl_complete", {"nodes": n, "url": url})
#
# Minimum 5 spans: crawl, ground, generate, judge, execute

## OUTPUT audit/sprint8/sprint8_results.json:
{
  "pipeline_stages_defined":  <int>,
  "otel_spans_emitted":       <int>,
  "structured_log_fields":    <int>,
  "audit_trail_entries":      <int>,
  "audit_chain_valid":        <bool>,
  "regression":               <bool>,
  "sprint8_status":           "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere (no os.path string concat)
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS in execution layer
# - SecurityASTChecker before subprocess
# - BFT: disabled (feature flag preserved)
# - CryptoAuditTrail: append-only, never mutate existing entries
# - OTelTracer: local JSONL only, no external collector calls
# - generate_report.py: exit(1) on any regression or broken chain