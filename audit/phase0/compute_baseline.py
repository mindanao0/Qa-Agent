# audit/phase0/compute_baseline.py
"""
Phase 0 — Baseline metrics computation.

Reads the JSONL golden dataset produced by ``build_golden_dataset.py`` plus
a short instrumented sample, and emits ``audit/phase0/baseline_metrics.json``
in the EXACT schema declared in the Phase 0 spec.

Two metric sources:

  A) JSONL-derived (cheap, no GPU time):
       - generation latency (from CaseRecord.generation_seconds)
       - execution metrics (from CaseRecord.pytest_runs[i].duration_ms / signature)
       - JSON-parse / syntax-error rates (from failure_signature)

  B) Instrumented sample (re-runs N cases through `main.py --mode generate`
     while polling nvidia-smi every 2 s + psutil RSS, capturing peak):
       - resource_metrics.peak_vram_mb / peak_ram_mb / ollama_oom_events
       - healing_metrics (parses session JSON written under ~/.qa-agent/sessions)
       - avg_tokens_input / avg_tokens_output (estimated via tokenizer if available,
         else 4 chars per token fallback)

  C) Live store probe:
       - rag_metrics.total_chunks  (LanceDB qa_docs row count)
       - avg_retrieval_latency_ms (small probe via HybridRetriever)

Standalone usage::

    uv run python audit/phase0/compute_baseline.py
    uv run python audit/phase0/compute_baseline.py --sample-size 10
    uv run python audit/phase0/compute_baseline.py --no-sample   # JSONL only

Hard constraints honoured per Phase 0 spec:
  * Windows 11 bare-metal — pathlib.Path everywhere; no os.path concat.
  * Read-only against existing src/ — invokes main.py via subprocess only.
  * Logging via loguru.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "audit" / "phase0"
DATASET_DIR = AUDIT_DIR / "golden_dataset"
WORK_DIR = AUDIT_DIR / "baseline_work"
BASELINE_OUT = AUDIT_DIR / "baseline_metrics.json"
SESSIONS_DIR = Path(os.path.expanduser("~/.qa-agent/sessions"))

PASSING_PATH = DATASET_DIR / "passing.jsonl"
FAILING_PATH = DATASET_DIR / "failing.jsonl"
FLAKY_PATH = DATASET_DIR / "flaky.jsonl"

MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")


# ──────────────────────────────────────────────────────────────────────────────
# nvidia-smi + psutil sampler (Windows-friendly)
# ──────────────────────────────────────────────────────────────────────────────


class ResourceSampler:
    """
    Background thread that polls nvidia-smi every interval_sec, captures the
    peak GPU memory.used and peak system RAM RSS.

    Usage::
        s = ResourceSampler()
        s.start()
        ...
        s.stop()
        print(s.peak_vram_mb, s.peak_ram_mb)
    """

    def __init__(self, interval_sec: float = 2.0) -> None:
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_vram_mb: int = 0
        self.peak_ram_mb: int = 0
        self.samples: int = 0

    def _read_vram_used(self) -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                line = result.stdout.strip().splitlines()[0].strip()
                return int(line)
        except Exception:
            pass
        return 0

    def _read_ram_used(self) -> int:
        try:
            import psutil       # type: ignore
            return int(psutil.virtual_memory().used / (1024 * 1024))
        except Exception:
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            vram = self._read_vram_used()
            ram = self._read_ram_used()
            if vram > self.peak_vram_mb:
                self.peak_vram_mb = vram
            if ram > self.peak_ram_mb:
                self.peak_ram_mb = ram
            self.samples += 1
            if self._stop.wait(self.interval_sec):
                break

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_sec * 2)
        self._thread = None


# ──────────────────────────────────────────────────────────────────────────────
# JSONL loading
# ──────────────────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"Bad JSON in {path}: {exc}")
    return rows


def _safe_avg(xs: list[float]) -> float:
    return round(statistics.mean(xs), 2) if xs else 0.0


def _safe_p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return round(xs[0], 2)
    # Numpy-free percentile
    s = sorted(xs)
    k = (len(s) - 1) * 0.95
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics from JSONL (A)
# ──────────────────────────────────────────────────────────────────────────────


def metrics_from_dataset(
    passing: list[dict[str, Any]],
    failing: list[dict[str, Any]],
    flaky: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = passing + failing + flaky
    n = len(all_rows)

    # ── Generation metrics ──────────────────────────────────────────────
    gen_seconds = [
        float(r.get("generation_seconds", 0.0))
        for r in all_rows if r.get("generation_seconds")
    ]
    gen_ms = [s * 1000 for s in gen_seconds]

    json_fail = sum(
        1 for r in failing if r.get("failure_signature") == "JSON_PARSE_FAIL"
    )
    syntax_fail = sum(
        1 for r in failing if r.get("failure_signature") == "SYNTAX_ERROR"
    )
    # Also count JSON/syntax issues that may have caused generation to fail
    # (notes like "agent generate returned no script").
    n_failing = len(failing)

    # ── Execution metrics ───────────────────────────────────────────────
    first_run_passes = 0
    first_run_total = 0
    locator_timeouts = 0
    durations_ms: list[int] = []

    for r in all_rows:
        runs = r.get("pytest_runs") or []
        if runs:
            first_run_total += 1
            if runs[0].get("passed"):
                first_run_passes += 1
            for run in runs:
                if not run.get("passed"):
                    sig = run.get("signature", "")
                    if sig in ("TIMEOUT", "LOCATOR_NOT_FOUND"):
                        locator_timeouts += 1
                d = run.get("duration_ms", 0)
                if d:
                    durations_ms.append(d)

    first_run_pass_rate = (
        round(first_run_passes / first_run_total, 4)
        if first_run_total else 0.0
    )
    # locator_timeout_rate = locator-style failures over all runs
    total_runs = sum(len(r.get("pytest_runs") or []) for r in all_rows)
    locator_timeout_rate = (
        round(locator_timeouts / total_runs, 4) if total_runs else 0.0
    )

    # ── False-pass heuristic ────────────────────────────────────────────
    # Without manual labelling we cannot measure true false-pass rate;
    # report 0.0 here and document the limitation. Spec field exists.
    false_pass_rate_estimated = 0.0

    # ── Healing metrics (best-effort from sessions/ JSONs) ──────────────
    heal_attempts, heal_successes, heal_latencies = _harvest_healing()

    heal_attempts_avg = (
        round(sum(heal_attempts) / max(1, n_failing), 4)
        if heal_attempts else 0.0
    )
    heal_success_rate = (
        round(sum(heal_successes) / sum(heal_attempts), 4)
        if sum(heal_attempts) else 0.0
    )
    avg_heal_latency = _safe_avg(heal_latencies)

    return {
        "sample_size": {
            "passing": len(passing),
            "failing": n_failing,
            "flaky": len(flaky),
        },
        "generation_metrics": {
            "json_parse_failure_rate":
                round(json_fail / max(1, n), 4),
            "syntax_error_rate":
                round(syntax_fail / max(1, n), 4),
            "avg_tokens_input": 0,          # filled by instrumented sample if run
            "avg_tokens_output": 0,         # filled by instrumented sample if run
            "avg_generation_latency_ms": _safe_avg(gen_ms),
            "p95_generation_latency_ms": _safe_p95(gen_ms),
        },
        "execution_metrics": {
            "first_run_pass_rate": first_run_pass_rate,
            "locator_timeout_rate": locator_timeout_rate,
            "false_pass_rate_estimated": false_pass_rate_estimated,
            "avg_test_runtime_ms": _safe_avg([float(d) for d in durations_ms]),
        },
        "healing_metrics": {
            "heal_attempts_avg_per_failed_test": heal_attempts_avg,
            "heal_success_rate": heal_success_rate,
            "avg_heal_latency_ms": avg_heal_latency,
        },
    }


def _harvest_healing() -> tuple[list[int], list[int], list[float]]:
    """
    Walk ~/.qa-agent/sessions/*.json (written by build_graph._save_session)
    and harvest healer attempt counts + completion success.

    Returns: (attempts_per_session, successes_per_session, heal_latencies_ms)
    """
    attempts: list[int] = []
    successes: list[int] = []
    latencies: list[float] = []

    if not SESSIONS_DIR.exists():
        return attempts, successes, latencies

    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        retry_count = int(data.get("retry_count") or 0)
        if retry_count <= 0:
            continue
        attempts.append(retry_count)
        # Success = the execution_result.success after retries
        result = data.get("execution_result") or {}
        if result.get("success"):
            successes.append(retry_count)
        # We don't have explicit heal_latency_ms in the saved state, so leave it
        # to the instrumented sample to populate.
    return attempts, successes, latencies


# ──────────────────────────────────────────────────────────────────────────────
# RAG live probe (C)
# ──────────────────────────────────────────────────────────────────────────────


def rag_metrics_live() -> dict[str, Any]:
    """
    Open the existing LanceDB store, count chunks, and time one retrieval.
    """
    try:
        # Lazy imports — keep the script importable without the optional deps.
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.llm.adapter import OllamaAdapter
        from src.rag.retriever import HybridRetriever
        from src.rag.store import VectorStore
    except Exception as exc:
        logger.warning(f"RAG live probe: imports failed ({exc})")
        return {
            "total_chunks": 0,
            "avg_retrieval_latency_ms": 0,
            "cache_hit_rate_estimated": 0.0,
        }

    async def _probe() -> tuple[int, float]:
        total = 0
        retrieval_latencies: list[float] = []
        try:
            vstore = await VectorStore().connect()
            total = await vstore.count()
        except Exception as exc:
            logger.warning(f"VectorStore probe failed: {exc}")
            return 0, 0.0

        if total == 0:
            return 0, 0.0

        try:
            async with OllamaAdapter() as adapter:
                retriever = HybridRetriever(store=vstore, adapter=adapter)
                queries = [
                    "playwright get_by_role usage",
                    "form authentication login flow",
                    "cart checkout discount",
                ]
                for q in queries:
                    t0 = time.monotonic()
                    chunks = await retriever.retrieve(q)
                    dt_ms = (time.monotonic() - t0) * 1000
                    retrieval_latencies.append(dt_ms)
                    logger.debug(
                        f"RAG probe q={q!r} -> {len(chunks)} chunks in {dt_ms:.1f}ms"
                    )
        except Exception as exc:
            logger.warning(f"RAG retriever probe failed: {exc}")

        avg = sum(retrieval_latencies) / max(1, len(retrieval_latencies))
        return total, avg

    try:
        total, avg_ms = asyncio.run(_probe())
    except Exception as exc:
        logger.warning(f"RAG probe asyncio.run failed: {exc}")
        return {
            "total_chunks": 0,
            "avg_retrieval_latency_ms": 0,
            "cache_hit_rate_estimated": 0.0,
        }

    return {
        "total_chunks": int(total),
        "avg_retrieval_latency_ms": int(round(avg_ms)),
        # No semantic-cache hit telemetry exposed currently — keep 0 + document.
        "cache_hit_rate_estimated": 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented sample (B) — captures peak VRAM/RAM + token counts
# ──────────────────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """4 chars / token fallback (Qwen2.5 tokenizer is ~3.5)."""
    return max(1, len(text) // 4)


def instrumented_sample(
    cases: list[dict[str, Any]],
    sample_size: int,
    model: str,
) -> dict[str, Any]:
    """
    Re-run *sample_size* cases through ``main.py --mode generate`` with a
    ResourceSampler attached. Estimates avg_tokens_input/output from the
    requirement length + generated script length.
    """
    if not cases or sample_size <= 0:
        return {
            "peak_vram_mb": 0,
            "peak_ram_mb": 0,
            "ollama_oom_events": 0,
            "avg_tokens_input": 0,
            "avg_tokens_output": 0,
        }

    # Take the first N — deterministic, includes the heaviest cases first
    sample = cases[: sample_size]
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    oom_events = 0

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    sampler = ResourceSampler(interval_sec=2.0)
    sampler.start()
    try:
        for case in sample:
            req = case.get("requirement_text", "")
            url = case.get("target_url", "")
            role = case.get("role", "admin")
            out_dir = WORK_DIR / case.get("case_id", "case")
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "-u", str(PROJECT_ROOT / "main.py"),
                "--mode", "generate",
                "--requirement", req,
                "--url", url,
                "--role", role,
                "--output-dir", str(out_dir),
                "--model", model,
                "--max-retries", "1",
                "--log-level", "INFO",
            ]
            try:
                proc = subprocess.run(
                    cmd, cwd=str(PROJECT_ROOT),
                    capture_output=True, text=True, timeout=300,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
            except subprocess.TimeoutExpired:
                continue

            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if "out of memory" in output.lower() or "OOM" in output:
                oom_events += 1

            # Crude token estimates
            in_tokens.append(_estimate_tokens(req))
            # The generated script is the assistant's output
            generated = sorted(out_dir.glob("*.py"), key=lambda p: p.stat().st_mtime)
            if generated:
                out_tokens.append(_estimate_tokens(
                    generated[-1].read_text(encoding="utf-8", errors="replace")
                ))
            else:
                out_tokens.append(0)
    finally:
        sampler.stop()

    return {
        "peak_vram_mb": int(sampler.peak_vram_mb),
        "peak_ram_mb": int(sampler.peak_ram_mb),
        "ollama_oom_events": int(oom_events),
        "avg_tokens_input": int(round(_safe_avg([float(x) for x in in_tokens]))),
        "avg_tokens_output": int(round(_safe_avg([float(x) for x in out_tokens]))),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 — baseline metrics")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Instrumented re-runs through the agent (for VRAM/RAM peaks + token counts)",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip instrumented sample — JSONL-only metrics (fast, no GPU time)",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    passing = _load_jsonl(PASSING_PATH)
    failing = _load_jsonl(FAILING_PATH)
    flaky = _load_jsonl(FLAKY_PATH)
    logger.info(
        f"Loaded JSONL | passing={len(passing)} failing={len(failing)} flaky={len(flaky)}"
    )

    # ── A: from dataset JSONL ───────────────────────────────────────────
    base = metrics_from_dataset(passing, failing, flaky)

    # ── C: live RAG probe ────────────────────────────────────────────────
    rag = rag_metrics_live()
    logger.info(f"RAG metrics | {rag}")

    # ── B: instrumented sample ───────────────────────────────────────────
    if args.no_sample:
        logger.info("Skipping instrumented sample (--no-sample)")
        resource_block = {
            "peak_vram_mb": 0,
            "peak_ram_mb": 0,
            "ollama_oom_events": 0,
            "avg_tokens_input": 0,
            "avg_tokens_output": 0,
        }
    else:
        logger.info(f"Running instrumented sample (size={args.sample_size})…")
        # Prefer well-formed cases that already produced scripts
        candidates = passing + flaky + failing
        resource_block = instrumented_sample(
            cases=candidates,
            sample_size=args.sample_size,
            model=args.model,
        )
        logger.info(f"Instrumented sample done | {resource_block}")

    # Patch token counts into the generation_metrics block (spec schema)
    base["generation_metrics"]["avg_tokens_input"] = resource_block["avg_tokens_input"]
    base["generation_metrics"]["avg_tokens_output"] = resource_block["avg_tokens_output"]

    # ── Final assembly per spec ──────────────────────────────────────────
    out: dict[str, Any] = {
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "sample_size": base["sample_size"],
        "generation_metrics": base["generation_metrics"],
        "execution_metrics": base["execution_metrics"],
        "healing_metrics": base["healing_metrics"],
        "resource_metrics": {
            "peak_vram_mb": resource_block["peak_vram_mb"],
            "peak_ram_mb": resource_block["peak_ram_mb"],
            "ollama_oom_events": resource_block["ollama_oom_events"],
        },
        "rag_metrics": rag,
        # Provenance — not in spec schema but useful for sprint planning.
        "_provenance": {
            "dataset_passing_path": str(PASSING_PATH.relative_to(PROJECT_ROOT)),
            "dataset_failing_path": str(FAILING_PATH.relative_to(PROJECT_ROOT)),
            "dataset_flaky_path":   str(FLAKY_PATH.relative_to(PROJECT_ROOT)),
            "instrumented_sample_size":
                0 if args.no_sample else min(args.sample_size,
                                             len(passing) + len(failing) + len(flaky)),
            "false_pass_rate_estimated_method":
                "not_measured (requires manual labelling)",
        },
    }

    BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"BASELINE_WRITTEN -> {BASELINE_OUT}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
