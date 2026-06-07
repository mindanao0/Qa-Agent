"""Tests for BrowserWorkerPool (Sprint 15).

Uses lightweight fakes for Browser/Context/Page so fan-out, chunking, isolation
tracking, the serialized-LLM invariant, timeouts and guaranteed context.close()
are all verified without a real browser or LLM.
"""
from __future__ import annotations

import asyncio

import pytest

from src.explorer.hypothesis import TestHypothesis
from src.observability.tracer import OTelTracer
from src.parallel.worker_pool import (
    BrowserWorkerPool,
    WorkerConfig,
    WorkerResult,
    _chunk,
)


# ── Fakes ────────────────────────────────────────────────────────────────────


class _Impl:
    def __init__(self, guid: str) -> None:
        self._guid = guid


class FakeContext:
    def __init__(self, guid: str) -> None:
        self._impl_obj = _Impl(guid)
        self.closed = False

    async def new_page(self):
        return object()

    async def close(self) -> None:
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []
        self._n = 0

    async def new_context(self) -> FakeContext:
        self._n += 1
        ctx = FakeContext(f"guid-{self._n}")
        self.contexts.append(ctx)
        return ctx


def _tasks(n: int) -> list[TestHypothesis]:
    return [
        TestHypothesis(
            hypothesis_id=None,  # type: ignore[arg-type]
            goal=f"task {i}",
            start_url="https://example.test/#/",
            preconditions=[],
            steps=[f'Type "item {i}" and press enter'],
            expected_outcome="ok",
        )
        for i in range(n)
    ]


def _pool(tmp_path, **kwargs) -> BrowserWorkerPool:
    return BrowserWorkerPool(tracer=OTelTracer(traces_dir=tmp_path), **kwargs)


# ── _chunk ───────────────────────────────────────────────────────────────────


def test_chunk_even():
    chunks = _chunk(_tasks(12), 3)
    assert [len(c) for c in chunks] == [4, 4, 4]


def test_chunk_uneven_front_loaded():
    chunks = _chunk(_tasks(10), 3)
    assert [len(c) for c in chunks] == [4, 3, 3]


def test_chunk_more_workers_than_tasks_drops_empties():
    chunks = _chunk(_tasks(2), 5)
    assert [len(c) for c in chunks] == [1, 1]


def test_chunk_empty():
    assert _chunk([], 3) == []


# ── WorkerConfig ─────────────────────────────────────────────────────────────


def test_config_caps_max_workers_at_hard_cap():
    cfg = WorkerConfig(max_workers=10)
    assert cfg.max_workers == 5


def test_config_extra_forbidden():
    with pytest.raises(Exception):
        WorkerConfig(unknown_field=1)  # type: ignore[call-arg]


def test_worker_result_extra_forbidden():
    with pytest.raises(Exception):
        WorkerResult(  # type: ignore[call-arg]
            worker_id="w", task_id="t", passed=True, duration_ms=1.0, surprise=9
        )


# ── run_parallel ─────────────────────────────────────────────────────────────


async def test_run_parallel_distributes_all_tasks(tmp_path):
    async def runner(task, page, worker_id):
        return True

    pool = _pool(tmp_path, config=WorkerConfig(n_workers=3), task_runner=runner)
    browser = FakeBrowser()
    results = await pool.run_parallel(_tasks(12), browser)

    assert len(results) == 12
    assert pool.workers_used == 3
    assert all(r.passed for r in results)
    # every task id is represented exactly once
    assert len({r.task_id for r in results}) == 12


async def test_run_parallel_closes_every_context(tmp_path):
    async def runner(task, page, worker_id):
        return True

    pool = _pool(tmp_path, config=WorkerConfig(n_workers=3), task_runner=runner)
    browser = FakeBrowser()
    await pool.run_parallel(_tasks(9), browser)

    assert len(browser.contexts) == 3
    assert all(c.closed for c in browser.contexts)


async def test_run_parallel_isolation_distinct_context_ids(tmp_path):
    async def runner(task, page, worker_id):
        return True

    pool = _pool(tmp_path, config=WorkerConfig(n_workers=3), task_runner=runner)
    await pool.run_parallel(_tasks(9), FakeBrowser())

    assert len(pool.context_ids) == 3
    assert len(set(pool.context_ids)) == 3  # all isolated


async def test_run_parallel_empty_tasks(tmp_path):
    pool = _pool(tmp_path, task_runner=lambda *a: asyncio.sleep(0, result=True))
    results = await pool.run_parallel([], FakeBrowser())
    assert results == []
    assert pool.workers_used == 0


async def test_context_closed_even_when_task_raises(tmp_path):
    async def runner(task, page, worker_id):
        raise RuntimeError("boom")

    pool = _pool(tmp_path, config=WorkerConfig(n_workers=2), task_runner=runner)
    browser = FakeBrowser()
    results = await pool.run_parallel(_tasks(4), browser)

    assert all(not r.passed for r in results)
    assert all(r.error and "boom" in r.error for r in results)
    assert all(c.closed for c in browser.contexts)


async def test_task_timeout_recorded_as_error(tmp_path):
    async def slow(task, page, worker_id):
        await asyncio.sleep(0.3)
        return True

    pool = _pool(
        tmp_path,
        config=WorkerConfig(n_workers=2, task_timeout_s=0.05),
        task_runner=slow,
    )
    results = await pool.run_parallel(_tasks(2), FakeBrowser())
    assert all(not r.passed for r in results)
    assert all(r.error and "timeout" in r.error for r in results)


async def test_effective_worker_count_capped_by_task_count(tmp_path):
    async def runner(task, page, worker_id):
        return True

    pool = _pool(tmp_path, config=WorkerConfig(n_workers=3), task_runner=runner)
    await pool.run_parallel(_tasks(2), FakeBrowser())
    assert pool.workers_used == 2  # cannot use more workers than tasks


# ── Serialized-LLM invariant ─────────────────────────────────────────────────


async def test_generator_fn_is_serialized_by_llm_semaphore(tmp_path):
    """Generation (LLM) must never run concurrently even with parallel workers."""
    state = {"current": 0, "max": 0}

    async def generator(task):
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.02)  # widen the overlap window
        state["current"] -= 1
        return task

    async def runner(task, page, worker_id):
        await asyncio.sleep(0.02)  # browser work runs in parallel
        return True

    pool = _pool(
        tmp_path,
        config=WorkerConfig(n_workers=3),
        task_runner=runner,
        generator_fn=generator,
    )
    await pool.run_parallel(_tasks(9), FakeBrowser())

    assert state["max"] == 1  # serialized: never more than one LLM call at once


async def test_llm_semaphore_value_is_one(tmp_path):
    pool = _pool(tmp_path)
    assert pool.llm_semaphore._value == 1
