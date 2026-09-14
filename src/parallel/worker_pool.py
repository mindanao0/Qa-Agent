"""BrowserWorkerPool — Sprint 15.

Fan-out: distribute a list of ``TestHypothesis`` tasks across N isolated
``BrowserContext`` workers running concurrently via ``asyncio.gather``. Each
worker owns its own context (separate cookies / localStorage) and runs its slice
of tasks sequentially on a single page.

INVARIANT (non-negotiable, 6 GB VRAM): the LLM/Ollama path stays serialized.
The pool's ``_llm_semaphore = asyncio.Semaphore(1)`` guards any generation step
the pool itself triggers, and the underlying ``HypothesisExecutor`` reaches
Ollama through ``src.llm.adapter._inference_semaphore`` (also ``Semaphore(1)``).
Parallelism is browser actions only — never parallel LLM calls.

Memory rule: ``context.close()`` runs in a ``finally`` block for every worker so
browser handles never leak (Sprint 15 RULES).
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from loguru import logger
from playwright.async_api import Browser, BrowserContext, Page
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.explorer.hypothesis import TestHypothesis
from src.observability.tracer import OTelTracer

# Hard cap on workers — 16 GB Windows RAM constraint (Sprint 15 RULES).
_MAX_WORKERS_HARD_CAP = 5

# A runner takes (task, page, worker_id) and returns whether the task passed.
TaskRunner = Callable[[TestHypothesis, Page, str], Awaitable[bool]]
# A generator takes a task and returns a (possibly enriched) task. Runs under
# the LLM semaphore so only one Ollama call happens at a time across workers.
GeneratorFn = Callable[[TestHypothesis], Awaitable[TestHypothesis]]


class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_workers: int = Field(default=3, ge=1)
    max_workers: int = Field(default=5, ge=1)  # hard cap (Windows RAM constraint)
    task_timeout_s: float = Field(default=30.0, gt=0.0)
    headless: bool = True

    @field_validator("max_workers")
    @classmethod
    def _cap_max_workers(cls, v: int) -> int:
        # Never allow more than the absolute hard cap regardless of input.
        return min(v, _MAX_WORKERS_HARD_CAP)


class WorkerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    task_id: str
    passed: bool
    duration_ms: float
    error: str | None = None


def _chunk(tasks: list[TestHypothesis], n: int) -> list[list[TestHypothesis]]:
    """Split ``tasks`` into ``n`` contiguous, as-even-as-possible chunks.

    Empty chunks are dropped, so the result length is the number of workers that
    will actually run (``<= n`` and ``<= len(tasks)``).
    """
    if n <= 0 or not tasks:
        return []
    total = len(tasks)
    base, extra = divmod(total, n)
    chunks: list[list[TestHypothesis]] = []
    start = 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        if size == 0:
            continue
        chunks.append(tasks[start : start + size])
        start += size
    return chunks


class BrowserWorkerPool:
    """Manages N isolated ``BrowserContext`` workers for parallel task execution.

    The pool is collaborator-agnostic for testability: pass a ``task_runner`` to
    execute each task (defaults to a :class:`HypothesisExecutor` set via
    :meth:`set_executor`), and an ``OTelTracer`` for span emission.
    """

    def __init__(
        self,
        config: WorkerConfig | None = None,
        *,
        task_runner: TaskRunner | None = None,
        generator_fn: GeneratorFn | None = None,
        tracer: OTelTracer | None = None,
    ) -> None:
        self._config = config or WorkerConfig()
        self._llm_semaphore = asyncio.Semaphore(1)  # INVARIANT: never change
        self._task_runner = task_runner
        self._generator_fn = generator_fn
        self._tracer = tracer or OTelTracer()

        # Populated during run_parallel — read by the measure harness to confirm
        # worker isolation (distinct contexts) and worker count.
        self.context_ids: list[str] = []
        self.workers_used: int = 0

    # ── Configuration ───────────────────────────────────────────────────────────

    @property
    def config(self) -> WorkerConfig:
        return self._config

    @property
    def llm_semaphore(self) -> asyncio.Semaphore:
        return self._llm_semaphore

    def set_executor(self, executor) -> None:
        """Use a :class:`HypothesisExecutor`-like object as the task runner.

        ``executor.execute(task, page)`` must return an object with a ``.passed``
        attribute. Kept lazy so unit tests can run the pool without a live LLM.
        """

        async def _runner(task: TestHypothesis, page: Page, worker_id: str) -> bool:
            result = await executor.execute(task, page)
            return bool(getattr(result, "passed", False))

        self._task_runner = _runner

    def _effective_worker_count(self, n_tasks: int) -> int:
        return max(
            1,
            min(self._config.n_workers, self._config.max_workers, max(1, n_tasks)),
        )

    # ── Fan-out ───────────────────────────────────────────────────────────────

    async def run_parallel(
        self,
        tasks: list[TestHypothesis],
        browser: Browser,
        *,
        storage_state: dict | str | None = None,
    ) -> list[WorkerResult]:
        """Distribute ``tasks`` across N isolated-context workers concurrently.

        ``storage_state`` (optional) seeds every worker's fresh
        ``BrowserContext`` with the given cookies/localStorage — e.g. the
        caller's already-authenticated session via
        ``await context.storage_state()`` — so parallel fan-out doesn't lose
        an existing login. ``None`` (the default) is identical to today's
        behaviour: a plain ``browser.new_context()`` per worker.

        Returns one :class:`WorkerResult` per task (order: worker-major).
        """
        self.context_ids = []
        if not tasks:
            self.workers_used = 0
            return []

        n = self._effective_worker_count(len(tasks))
        chunks = _chunk(tasks, n)
        self.workers_used = len(chunks)

        logger.info(
            f"BrowserWorkerPool: {len(tasks)} tasks across {len(chunks)} workers "
            f"(config n={self._config.n_workers}, cap={self._config.max_workers})"
        )

        per_worker = await asyncio.gather(
            *[
                self._run_worker(chunk, browser, f"worker-{i}", storage_state=storage_state)
                for i, chunk in enumerate(chunks)
            ]
        )
        return [r for worker_results in per_worker for r in worker_results]

    async def _run_worker(
        self,
        tasks: list[TestHypothesis],
        browser: Browser,
        worker_id: str,
        *,
        storage_state: dict | str | None = None,
    ) -> list[WorkerResult]:
        # Keep the zero-arg call for the common (no storage_state) case — real
        # Playwright treats storage_state=None identically, but existing
        # test fakes/mocks for browser.new_context() take no kwargs at all.
        context: BrowserContext = (
            await browser.new_context(storage_state=storage_state)
            if storage_state is not None
            else await browser.new_context()
        )
        self.context_ids.append(self._context_guid(context))
        try:
            async with self._tracer.span(
                "parallel.worker", worker_id=worker_id, n_tasks=len(tasks)
            ):
                page = await context.new_page()
                results: list[WorkerResult] = []
                for task in tasks:
                    results.append(await self._execute_task(task, page, worker_id))
                return results
        finally:
            try:
                await context.close()  # MUST close — prevent handle leak
            except Exception as exc:
                logger.warning(f"{worker_id}: context.close() failed: {exc!r}")

    async def _execute_task(
        self,
        task: TestHypothesis,
        page: Page,
        worker_id: str,
    ) -> WorkerResult:
        start = time.monotonic()
        passed = False
        error: str | None = None
        try:
            # LLM generation step — SERIALIZED. For pre-built hypotheses this is
            # a no-op pass-through; when a generator_fn is configured only ONE
            # Ollama call runs at a time across all workers.
            if self._generator_fn is not None:
                async with self._llm_semaphore:
                    task = await self._generator_fn(task)

            # Browser execution — fully parallel across workers.
            async with self._tracer.span(
                "parallel.task", worker_id=worker_id, task_id=task.hypothesis_id
            ):
                if self._task_runner is None:
                    raise RuntimeError(
                        "BrowserWorkerPool has no task_runner; call set_executor() "
                        "or pass task_runner=..."
                    )
                passed = await asyncio.wait_for(
                    self._task_runner(task, page, worker_id),
                    timeout=self._config.task_timeout_s,
                )
        except asyncio.TimeoutError:
            error = f"task timeout after {self._config.task_timeout_s}s"
            logger.warning(f"{worker_id}: {error} (task_id={task.hypothesis_id})")
        except Exception as exc:
            error = repr(exc)
            logger.warning(f"{worker_id}: task failed {task.hypothesis_id}: {exc!r}")
        duration_ms = (time.monotonic() - start) * 1000.0
        return WorkerResult(
            worker_id=worker_id,
            task_id=task.hypothesis_id,
            passed=passed,
            duration_ms=round(duration_ms, 3),
            error=error,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _context_guid(context: BrowserContext) -> str:
        """Best-effort stable identity for a BrowserContext (isolation proof)."""
        try:
            guid = context._impl_obj._guid  # type: ignore[attr-defined]
            if guid:
                return str(guid)
        except Exception:
            pass
        return f"ctx-{id(context)}"


__all__ = ["BrowserWorkerPool", "WorkerConfig", "WorkerResult"]
