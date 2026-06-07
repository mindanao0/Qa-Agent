## CONTEXT
PROJECT: D:\Code\qa-agent
BASELINE: Sprint 14 PASS — HypothesisRunner + PBTResult available,
          ContinuousLoopController (Sprint 11) stable at 41MB growth
CONSTRAINT: 6GB VRAM GTX 1660 Ti — Semaphore(1) on Ollama is NON-NEGOTIABLE
            parallel = browser workers only, NOT parallel LLM calls
STACK: Python 3.13 + uv, LangGraph async, Playwright CDP, Windows 11

## OBJECTIVE
GOAL: รัน multiple Playwright browser workers concurrently (fan-out)
      โดย LLM generation ยังคง serialized (Semaphore(1))
      วัด throughput gain vs sequential baseline

## ACCEPTANCE GATE (sprint15)
  parallel_workers_used       ≥ 3
  throughput_gain             ≥ 1.5   (parallel_tps / sequential_tps)
  vram_stable                 = True  (no OOM, no CUDA error during run)
  test_pass_rate              ≥ 0.75  (quality not degraded by parallelism)
  worker_isolation_confirmed  = True  (each worker isolated BrowserContext)
  otel_spans_emitted          ≥ 10
  REGRESSION if pass_rate     < 0.75

## ARCHITECTURE — New files:

### src/parallel/worker_pool.py  (NEW)
# BrowserWorkerPool — manages N isolated BrowserContext workers
#
# class WorkerConfig(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     n_workers:      int   = 3      # start conservative for 6GB VRAM
#     max_workers:    int   = 5      # hard cap (Windows RAM constraint)
#     task_timeout_s: float = 30.0
#     headless:       bool  = True
#
# class WorkerResult(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     worker_id:    str
#     task_id:      str
#     passed:       bool
#     duration_ms:  float
#     error:        str | None
#
# class BrowserWorkerPool:
#     """
#     Fan-out: distribute tasks across N workers concurrently
#     Each worker = isolated BrowserContext (separate cookies/localStorage)
#     LLM generation = serialized via _llm_semaphore = asyncio.Semaphore(1)
#     Browser actions = fully parallel via asyncio.gather()
#     """
#
#     def __init__(self, config: WorkerConfig | None = None) -> None:
#         self._config = config or WorkerConfig()
#         self._llm_semaphore = asyncio.Semaphore(1)  # INVARIANT: never change
#
#     async def run_parallel(
#         self,
#         tasks: list[TestHypothesis],
#         browser: Browser,
#     ) -> list[WorkerResult]:
#         """
#         1. Chunk tasks into N worker batches
#         2. asyncio.gather(*[self._run_worker(chunk, browser) for chunk in chunks])
#         3. Each worker: new_context() → run tasks sequentially within worker
#         4. LLM calls inside worker: async with self._llm_semaphore
#         5. Collect WorkerResult per task
#         6. context.close() after worker finishes (no leak)
#         """
#
#     async def _run_worker(
#         self,
#         tasks: list[TestHypothesis],
#         browser: Browser,
#         worker_id: str,
#     ) -> list[WorkerResult]:
#         context = await browser.new_context()
#         try:
#             page = await context.new_page()
#             results = []
#             for task in tasks:
#                 result = await self._execute_task(task, page, worker_id)
#                 results.append(result)
#             return results
#         finally:
#             await context.close()   # MUST close — prevent handle leak

### src/parallel/throughput_meter.py  (NEW)
# ThroughputMeter — measures sequential vs parallel TPS
#
# class ThroughputMeter:
#     """No required constructor args."""
#
#     def record_sequential(self, n_tasks: int, duration_s: float) -> float:
#         """Returns sequential TPS = n_tasks / duration_s"""
#
#     def record_parallel(self, n_tasks: int, duration_s: float) -> float:
#         """Returns parallel TPS = n_tasks / duration_s"""
#
#     def throughput_gain(self) -> float:
#         """Returns parallel_tps / sequential_tps. Requires both recorded."""
#
#     def to_dict(self) -> dict:
#         return {
#             "sequential_tps": ...,
#             "parallel_tps":   ...,
#             "throughput_gain": ...,
#             "n_workers":       ...,
#         }

### src/parallel/vram_monitor.py  (NEW)
# VRAMMonitor — polls GPU memory during parallel run (Windows NVML)
#
# class VRAMMonitor:
#     """No required constructor args. Graceful fallback if NVML unavailable."""
#
#     def __init__(self) -> None:
#         try:
#             import ctypes
#             self._nvml = ctypes.CDLL("nvml.dll")  # Windows path
#             self._available = True
#         except OSError:
#             self._available = False
#
#     def used_mb(self) -> float | None:
#         """Returns VRAM used in MB, or None if NVML unavailable."""
#
#     def is_stable(self, threshold_mb: float = 5800.0) -> bool:
#         """True if used_mb < threshold_mb (leave 200MB headroom on 6GB)"""
#
#     async def monitor_during(
#         self,
#         coro: Coroutine,
#         poll_interval_s: float = 0.5,
#     ) -> tuple[Any, float]:
#         """
#         Run coro while polling VRAM every poll_interval_s.
#         Returns (coro_result, peak_vram_mb).
#         If peak_vram_mb > 5800 → raise VRAMExceededError
#         """

### audit/sprint15/measure_sprint15.py
# Test tasks: reuse hypotheses from Sprint 11 (ToDoMVC flows)
# 12 tasks total (4 per worker × 3 workers)
#
# Flow:
#   1. Run 12 tasks SEQUENTIALLY → record sequential TPS via ThroughputMeter
#   2. Run same 12 tasks PARALLEL (3 workers) → record parallel TPS
#   3. VRAMMonitor.monitor_during(parallel_run) → peak_vram_mb
#   4. Verify worker_isolation:
#      each worker must use different BrowserContext.browser_context_id
#   5. OTelTracer.span("parallel.worker") per worker
#   6. CryptoAuditTrail.append("parallel_run", throughput_meter.to_dict())
#
# vram_stable = VRAMMonitor().is_stable() OR (not vram_available → True by default)

## OUTPUT audit/sprint15/sprint15_results.json:
{
  "parallel_workers_used":      <int>,
  "throughput_gain":            <float>,
  "sequential_tps":             <float>,
  "parallel_tps":               <float>,
  "vram_stable":                <bool>,
  "peak_vram_mb":               <float | null>,
  "test_pass_rate":             <float>,
  "worker_isolation_confirmed": <bool>,
  "otel_spans_emitted":         <int>,
  "regression":                 <bool>,
  "sprint15_status":            "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls — NEVER relax for parallelism
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - BLOCKED_ACTION_PATTERNS enforced per worker
# - Each worker: isolated BrowserContext (new_context() + close() in finally)
# - asyncio.gather() for worker fan-out (NOT threading.Thread)
# - Max workers hard cap = 5 (Windows 16GB RAM constraint)
# - VRAMMonitor: graceful fallback if NVML unavailable (vram_stable=True)
# - context.close() in finally block — MANDATORY (prevent browser handle leak)
# - BFT: disabled (feature flag preserved)
# - OTelTracer + CryptoAuditTrail on parallel run summary
# - throughput_gain must be computed from real wall-clock timing
#   NOT estimated or hardcoded