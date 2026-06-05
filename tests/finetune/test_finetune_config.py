"""Tests for FinetuneConfig and MemoryGuard (no GPU required)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.finetune import FinetuneConfig, MemoryGuard


class TestFinetuneConfig:
    def test_default_values(self):
        cfg = FinetuneConfig()
        assert cfg.model_name == "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
        assert cfg.max_seq_length == 1024
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.0
        assert cfg.per_device_train_batch_size == 1
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.warmup_steps == 10
        assert cfg.max_steps == 100
        assert cfg.learning_rate == pytest.approx(2e-4)
        assert cfg.fp16 is True
        assert cfg.optim == "adamw_8bit"
        assert cfg.seed == 42
        assert cfg.output_dir == "models/finetune_output"
        assert cfg.save_steps == 50
        assert cfg.logging_steps == 10

    def test_all_target_modules_present(self):
        cfg = FinetuneConfig()
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert set(cfg.target_modules) == expected

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FinetuneConfig(unknown_param="bad")

    def test_load_in_4bit_true(self):
        cfg = FinetuneConfig()
        assert cfg.load_in_4bit is True

    def test_output_dir_is_string(self):
        cfg = FinetuneConfig()
        assert isinstance(cfg.output_dir, str)


class TestMemoryGuard:
    def test_start_stop_returns_peak_or_none(self):
        guard = MemoryGuard(max_growth_mb=5800.0)
        guard.start()
        time.sleep(0.15)
        peak = guard.stop()
        assert peak is None or isinstance(peak, float)

    def test_context_manager_does_not_raise(self):
        with MemoryGuard(max_growth_mb=5800.0) as guard:
            time.sleep(0.05)
        assert guard.peak_mb is None or isinstance(guard.peak_mb, float)

    def test_stop_without_start_is_safe(self):
        guard = MemoryGuard(max_growth_mb=5800.0)
        peak = guard.stop()
        assert peak is None

    def test_exceeded_flag_starts_false(self):
        guard = MemoryGuard(max_growth_mb=5800.0)
        assert guard.exceeded is False
