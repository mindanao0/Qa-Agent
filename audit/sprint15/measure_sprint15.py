"""
Sprint 15 gate measurement — Parallel Execution.

Runs 12 reliably-passing TodoMVC flow hypotheses (reused from the Sprint 11
add/mark family) first SEQUENTIALLY on one page, then in PARALLEL across 3
isolated-context browser workers, and compares real wall-clock throughput.

Gates (docs/specs/"Sprint 15 Parallel Execution.md")
----------------------------------------------------
  parallel_workers_used       >= 3
  throughput_gain             >= 1.5   (parallel_tps / sequential_tps)
  vram_stable                 == True  (no OOM / CUDA error; NVML-unavailable -> True)
  test_pass_rate              >= 0.75  (quality not degraded by parallelism)
  worker_isolation_confirmed  == True  (each worker = isolated BrowserContext)
  otel_spans_emitted          >= 10
  REGRESSION if pass_rate     < 0.75

Honest-reporting posture: every number is a real measurement. The LLM/Ollama
path stays serialized (Semaphore(1)); parallelism is browser workers only. The
throughput gain is computed from wall-clock timing, never estimated.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import tempfile
import time

from loguru import logger
from playwright.async_api import async_playwright

from src.explorer.executor import HypothesisExecutor
from src.explorer.hypothesis import TestHypothesis
from src.contractskill.sfg import SFGStore
from src.llm.instructor_client import InstructorClient
from src.observability.audit_chain import CryptoAuditTrail
from src.observability.tracer import OTelTracer
from src.parallel.throughput_meter import ThroughputMeter
from src.parallel.vram_monitor import VRAMExceededError, VRAMMonitor
from src.parallel.worker_pool import BrowserWorkerPool, WorkerConfig

_START_URL = "https://demo.playwright.dev/todomvc/#/"
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint15_results.json"
_AUDIT_PATH = pathlib.Path(__file__).parent / "sprint15_audit.jsonl"
_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]

_N_WORKERS = 3
_N_TASKS = 12

# Gate thresholds
_GATE_WORKERS = 3
_GATE_GAIN = 1.5
_GATE_PASS_RATE = 0.75
_GATE_SPANS = 10
_REGRESSION_THRESHOLD = 0.75


def _build_tasks(start_url: str) -> list[TestHypothesis]:
    """12 reliably-passing TodoMVC hypotheses (Sprint 11 add/mark family).

    role/label/text affordances only (no CSS). Distinct payloads keep the tasks
    independent so a single shared page (sequential) and N pages (parallel)
    execute the *same* work — a fair throughput comparison.
    """
    items = [
        "Buy milk", "Walk the dog", "Read a book", "Write tests",
        "Pay bills", "Call mom", "Plan trip", "Water plants", "Fix bug",
    ]
    tasks: list[TestHypothesis] = []
    for name in items:  # 9 add-todo flows
        tasks.append(
            TestHypothesis(
                hypothesis_id=None,  # type: ignore[arg-type]
                goal=f"Add a new todo item: {name}",
                start_url=start_url,
                preconditions=["the todo app is open"],
                steps=[f'Type "{name}" into the new todo field and press enter'],
                expected_outcome="a new todo item appears in the list",
                confidence=0.7,
            )
        )
    for i in range(3):  # 3 mark-complete flows (executor seeds a todo first)
        tasks.append(
            TestHypothesis(
                hypothesis_id=None,  # type: ignore[arg-type]
                goal=f"Mark a todo as complete #{i + 1}",
                start_url=start_url,
                preconditions=["an existing todo item"],
                steps=["Click the checkbox to mark the first todo complete"],
                expected_outcome="the todo is shown as completed",
                confidence=0.7,
            )
        )
    return tasks


async def _run_sequential(
    tasks: list[TestHypothesis],
    browser,
    executor: HypothesisExecutor,
    tracer: OTelTracer,
) -> tuple[float, list[bool]]:
    """Run every task one-by-one on a single page. Returns (duration_s, passes)."""
    context = await browser.new_context()
    passes: list[bool] = []
    start = time.monotonic()
    try:
        page = await context.new_page()
        for task in tasks:
            async with tracer.span("sequential.task", task_id=task.hypothesis_id):
                try:
                    result = await executor.execute(task, page)
                    passes.append(bool(result.passed))
                except Exception as exc:
                    logger.warning(f"sequential task {task.hypothesis_id} failed: {exc!r}")
                    passes.append(False)
    finally:
        await context.close()
    return time.monotonic() - start, passes


async def _confirm_behavioral_isolation(browser, start_url: str) -> bool:
    """Two contexts must NOT share localStorage — a genuine isolation proof."""
    c1 = await browser.new_context()
    c2 = await browser.new_context()
    try:
        p1 = await c1.new_page()
        await p1.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
        await p1.evaluate("window.localStorage.setItem('sprint15_iso', 'worker-1')")
        p2 = await c2.new_page()
        await p2.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
        leaked = await p2.evaluate("window.localStorage.getItem('sprint15_iso')")
        return leaked is None
    except Exception as exc:
        logger.warning(f"isolation probe failed: {exc!r}")
        return False
    finally:
        await c1.close()
        await c2.close()


async def main() -> dict:
    # Defaults so we always write results, even on partial failure.
    parallel_workers_used = 0
    throughput_gain = 0.0
    sequential_tps = 0.0
    parallel_tps = 0.0
    vram_stable = True
    peak_vram_mb: float | None = None
    test_pass_rate = 0.0
    worker_isolation_confirmed = False
    otel_spans_emitted = 0
    regression = True

    tracer = OTelTracer()
    meter = ThroughputMeter()
    monitor = VRAMMonitor()

    tasks = _build_tasks(_START_URL)

    # Shared executor (LLM path serialized via the adapter's global semaphore).
    sfg_path = pathlib.Path(tempfile.mkdtemp(prefix="sprint15_sfg_")) / "sfg.db"
    sfg_store = SFGStore(db_path=sfg_path)
    instructor = InstructorClient()
    executor = HypothesisExecutor(instructor, sfg_store)

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
    try:
        # 1. SEQUENTIAL baseline ------------------------------------------------
        seq_dur, _seq_passes = await _run_sequential(tasks, browser, executor, tracer)
        sequential_tps = meter.record_sequential(len(tasks), seq_dur)
        logger.info(f"sequential: {len(tasks)} tasks in {seq_dur:.2f}s -> {sequential_tps:.3f} tps")

        # 2. PARALLEL run (3 workers) under VRAM monitoring ---------------------
        pool = BrowserWorkerPool(
            WorkerConfig(n_workers=_N_WORKERS),
            tracer=tracer,
        )
        pool.set_executor(executor)

        par_start = time.monotonic()
        try:
            results, peak_vram_mb = await monitor.monitor_during(
                pool.run_parallel(tasks, browser)
            )
        except VRAMExceededError as exc:
            logger.error(f"VRAM exceeded during parallel run: {exc!r}")
            vram_stable = False
            results = []
        par_dur = time.monotonic() - par_start

        parallel_workers_used = pool.workers_used
        parallel_tps = meter.record_parallel(len(tasks), par_dur, n_workers=pool.workers_used)
        logger.info(f"parallel: {len(tasks)} tasks in {par_dur:.2f}s -> {parallel_tps:.3f} tps")

        # 3. Throughput gain (real wall-clock) ----------------------------------
        try:
            throughput_gain = meter.throughput_gain()
        except ValueError:
            throughput_gain = 0.0

        # 4. Quality — parallel pass rate ---------------------------------------
        if results:
            test_pass_rate = sum(1 for r in results if r.passed) / len(results)

        # 5. Worker isolation — distinct contexts + behavioural localStorage probe
        distinct_contexts = (
            len(set(pool.context_ids)) == pool.workers_used and pool.workers_used >= 1
        )
        behavioural = await _confirm_behavioral_isolation(browser, _START_URL)
        worker_isolation_confirmed = bool(distinct_contexts and behavioural)

        # 6. VRAM stability ------------------------------------------------------
        if vram_stable:  # not already failed by VRAMExceededError
            if not monitor.available:
                vram_stable = True  # graceful fallback on non-NVIDIA hardware
            elif peak_vram_mb is not None:
                vram_stable = peak_vram_mb < 5800.0
            else:
                vram_stable = monitor.is_stable()

        regression = test_pass_rate < _REGRESSION_THRESHOLD

        # 7. Audit trail on the parallel-run summary ----------------------------
        audit = CryptoAuditTrail(path=_AUDIT_PATH)
        audit.append("parallel_run", meter.to_dict())
        audit.append(
            "parallel_quality",
            {
                "workers_used": pool.workers_used,
                "context_ids": pool.context_ids,
                "pass_rate": round(test_pass_rate, 6),
                "peak_vram_mb": peak_vram_mb,
            },
        )

    except Exception as exc:
        logger.error(f"measure_sprint15: unhandled error — {exc!r}")
    finally:
        otel_spans_emitted = tracer.flush()
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await pw.stop()
        except Exception:
            pass
        try:
            await instructor.close()
        except Exception:
            pass

    sprint15_pass = (
        parallel_workers_used >= _GATE_WORKERS
        and throughput_gain >= _GATE_GAIN
        and vram_stable
        and test_pass_rate >= _GATE_PASS_RATE
        and worker_isolation_confirmed
        and otel_spans_emitted >= _GATE_SPANS
        and not regression
    )

    results_dict = {
        "parallel_workers_used": parallel_workers_used,
        "throughput_gain": round(throughput_gain, 4),
        "sequential_tps": round(sequential_tps, 4),
        "parallel_tps": round(parallel_tps, 4),
        "vram_stable": vram_stable,
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "test_pass_rate": round(test_pass_rate, 4),
        "worker_isolation_confirmed": worker_isolation_confirmed,
        "otel_spans_emitted": otel_spans_emitted,
        "regression": regression,
        "sprint15_status": "PASS" if sprint15_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
    logger.info(f"measure_sprint15: results written to {_OUTPUT_PATH}")

    print("\n=== Sprint 15 Results ===")
    print(f"  parallel_workers_used:      {parallel_workers_used} (gate >= {_GATE_WORKERS})"
          f"{' ✓' if parallel_workers_used >= _GATE_WORKERS else ' ✗'}")
    print(f"  throughput_gain:            {results_dict['throughput_gain']} (gate >= {_GATE_GAIN})"
          f"{' ✓' if throughput_gain >= _GATE_GAIN else ' ✗'}")
    print(f"  sequential_tps:             {results_dict['sequential_tps']}")
    print(f"  parallel_tps:               {results_dict['parallel_tps']}")
    print(f"  vram_stable:                {vram_stable} (gate == True){' ✓' if vram_stable else ' ✗'}")
    print(f"  peak_vram_mb:               {results_dict['peak_vram_mb']}")
    print(f"  test_pass_rate:             {results_dict['test_pass_rate']} (gate >= {_GATE_PASS_RATE})"
          f"{' ✓' if test_pass_rate >= _GATE_PASS_RATE else ' ✗'}")
    print(f"  worker_isolation_confirmed: {worker_isolation_confirmed} (gate == True)"
          f"{' ✓' if worker_isolation_confirmed else ' ✗'}")
    print(f"  otel_spans_emitted:         {otel_spans_emitted} (gate >= {_GATE_SPANS})"
          f"{' ✓' if otel_spans_emitted >= _GATE_SPANS else ' ✗'}")
    print(f"  regression:                 {regression} (gate == False){' ✓' if not regression else ' ✗'}")
    print(f"\n  -> sprint15_status: {results_dict['sprint15_status']}")
    return results_dict


if __name__ == "__main__":
    asyncio.run(main())
