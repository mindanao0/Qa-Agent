"""Sprint 15 — Parallel Execution.

Browser-worker fan-out with the Ollama/LLM path kept strictly serialized
(``asyncio.Semaphore(1)``). See docs/specs/"Sprint 15 Parallel Execution.md".
"""
from __future__ import annotations

from src.parallel.throughput_meter import ThroughputMeter
from src.parallel.vram_monitor import VRAMExceededError, VRAMMonitor
from src.parallel.worker_pool import (
    BrowserWorkerPool,
    WorkerConfig,
    WorkerResult,
)

__all__ = [
    "BrowserWorkerPool",
    "WorkerConfig",
    "WorkerResult",
    "ThroughputMeter",
    "VRAMMonitor",
    "VRAMExceededError",
]
