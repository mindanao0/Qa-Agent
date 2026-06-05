"""QLoRA fine-tune with Unsloth — 6 GB VRAM optimised.

Run:
    uv run python scripts/finetune.py

Resumes from checkpoint automatically if models/finetune_output/
already contains a checkpoint-* subdirectory.

Install deps first:
    uv run python scripts/install_finetune_deps.py
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_JSONL = Path("data/training/train.jsonl")
VAL_JSONL = Path("data/training/val.jsonl")
METRICS_PATH = Path("models/finetune_output/training_metrics.json")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FinetuneConfig(BaseModel):
    """Spec-exact hyper-parameters for 6 GB VRAM QLoRA fine-tune."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 100
    learning_rate: float = 2e-4
    fp16: bool = True
    optim: str = "adamw_8bit"
    seed: int = 42
    output_dir: str = "models/finetune_output"
    save_steps: int = 50
    logging_steps: int = 10


# ---------------------------------------------------------------------------
# MemoryGuard — thread-based VRAM monitor
# ---------------------------------------------------------------------------


class MemoryGuard:
    """
    Background thread that polls GPU VRAM via VRAMMonitor every 0.5 s.

    If VRAM usage crosses max_growth_mb, it sets `exceeded=True` and sends
    SIGINT to the main thread so the SFTTrainer can save an emergency checkpoint.

    Degrades gracefully when NVML is unavailable (non-NVIDIA / CI).
    """

    def __init__(self, max_growth_mb: float = 5800.0) -> None:
        self.max_growth_mb = max_growth_mb
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._peak_mb: float | None = None
        self._exceeded = False

        try:
            from src.parallel.vram_monitor import VRAMMonitor  # noqa: PLC0415
            self._monitor: Any = VRAMMonitor()
        except Exception:
            self._monitor = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll, daemon=True, name="MemoryGuard"
        )
        self._thread.start()

    def stop(self) -> float | None:
        """Stop the guard and return peak VRAM in MB (or None)."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        return self._peak_mb

    def __enter__(self) -> "MemoryGuard":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def peak_mb(self) -> float | None:
        with self._lock:
            return self._peak_mb

    @property
    def exceeded(self) -> bool:
        with self._lock:
            return self._exceeded

    def _poll(self) -> None:
        while not self._stop.is_set():
            if self._monitor is not None:
                used = self._monitor.used_mb()
                if used is not None:
                    should_signal = False
                    with self._lock:
                        self._peak_mb = (
                            used if self._peak_mb is None else max(self._peak_mb, used)
                        )
                        if used > self.max_growth_mb and not self._exceeded:
                            self._exceeded = True
                            should_signal = True
                    if should_signal:
                        try:
                            os.kill(os.getpid(), signal.SIGINT)
                        except Exception:
                            pass
            self._stop.wait(timeout=0.5)
        # Final sample after stop
        if self._monitor is not None:
            used = self._monitor.used_mb()
            if used is not None:
                with self._lock:
                    self._peak_mb = (
                        used if self._peak_mb is None else max(self._peak_mb, used)
                    )
