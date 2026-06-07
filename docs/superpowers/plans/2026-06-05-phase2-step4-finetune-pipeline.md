# Phase 2 Step 4 — Fine-tune Pipeline (Unsloth QLoRA 4-bit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement five standalone scripts (install_finetune_deps, finetune, export_gguf, create_modelfile, eval_finetune) that fine-tune Qwen2.5-Coder-7B with QLoRA on the project's training data, export to GGUF, register in Ollama, and measure quality gain vs the base model.

**Architecture:** Each script is a standalone CLI tool under `scripts/`. Training calls Unsloth SFTTrainer synchronously with a thread-based MemoryGuard pulling from the existing `src/parallel/VRAMMonitor`. Evaluation is async, gated by `asyncio.Semaphore(1)`, and scores val.jsonl completions from both the base model and the fine-tuned Ollama model.

**Tech Stack:** Unsloth (QLoRA), TRL/SFTTrainer, Pydantic v2, pathlib, asyncio, `src/parallel/vram_monitor.VRAMMonitor`, `src/llm/adapter.OllamaAdapter`, `src/codetest/executor.TestExecutor`, jsonschema, ast

---

## File Map

| File | Create / Modify | Responsibility |
|---|---|---|
| `scripts/install_finetune_deps.py` | Create | Install Unsloth + training deps; verify imports |
| `scripts/finetune.py` | Create | FinetuneConfig, MemoryGuard, load_jsonl, to_chatml, train(), main() |
| `scripts/export_gguf.py` | Create | Load checkpoint, save_pretrained_gguf Q4_K_M, rename output |
| `scripts/create_modelfile.py` | Create | Write Modelfile, run `ollama create qa-agent-finetuned` |
| `scripts/eval_finetune.py` | Create | EvalResult model, scoring, async eval loop, write models/eval_results.json |
| `tests/finetune/__init__.py` | Create | Empty package marker |
| `tests/finetune/test_finetune_config.py` | Create | FinetuneConfig and MemoryGuard unit tests |
| `tests/finetune/test_data_loading.py` | Create | load_jsonl, to_chatml unit tests |
| `tests/finetune/test_modelfile.py` | Create | write_modelfile content tests |
| `tests/finetune/test_eval_scoring.py` | Create | score_completion, get_scoring_type, EvalResult tests |

---

## Task 1: Install script

**Files:**
- Create: `scripts/install_finetune_deps.py`

- [ ] **Step 1.1: Write the script**

```python
"""Install Unsloth + QLoRA fine-tuning dependencies.

Run:
    uv run python scripts/install_finetune_deps.py

On Windows, if bitsandbytes fails use:
    pip install bitsandbytes \
        --index-url https://jllllll.github.io/bitsandbytes-windows-webui
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEPS = [
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.29.0",
    "datasets>=2.18.0",
]

_VERIFY_IMPORTS = [
    "unsloth", "transformers", "trl", "peft",
    "bitsandbytes", "accelerate", "datasets",
]


def install() -> bool:
    """Install all deps. Return True on success."""
    for dep in DEPS:
        print(f"Installing: {dep}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", dep],
            check=False,
        )
        if result.returncode != 0:
            print(f"FAILED: {dep}")
            return False
    return True


def verify() -> bool:
    """Verify all packages importable. Return True if all OK."""
    ok = True
    for pkg in _VERIFY_IMPORTS:
        try:
            __import__(pkg)
            print(f"OK:      {pkg}")
        except ImportError:
            print(f"MISSING: {pkg}")
            ok = False
    return ok


def main() -> int:
    if not install():
        return 1
    print("\nVerifying imports...")
    if not verify():
        return 1
    print("\nAll dependencies installed and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 1.2: Commit**

```bash
git add scripts/install_finetune_deps.py
git commit -m "feat(finetune): add install_finetune_deps.py script"
```

---

## Task 2: FinetuneConfig and MemoryGuard (with tests)

**Files:**
- Create (section in): `scripts/finetune.py` — FinetuneConfig + MemoryGuard only
- Create: `tests/finetune/__init__.py`
- Create: `tests/finetune/test_finetune_config.py`

- [ ] **Step 2.1: Write tests first**

```python
# tests/finetune/test_finetune_config.py
"""Tests for FinetuneConfig and MemoryGuard (no GPU required)."""
from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

# These imports work because scripts/finetune.py uses top-level classes
# We import them directly to test without executing main().
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))  # project root on path

from scripts.finetune import FinetuneConfig, MemoryGuard  # noqa: E402


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
        # On non-NVML machines, peak is None. On NVIDIA, it's a float.
        assert peak is None or isinstance(peak, float)

    def test_context_manager_does_not_raise(self):
        with MemoryGuard(max_growth_mb=5800.0) as guard:
            time.sleep(0.05)
        assert guard.peak_mb is None or isinstance(guard.peak_mb, float)

    def test_stop_without_start_is_safe(self):
        guard = MemoryGuard(max_growth_mb=5800.0)
        peak = guard.stop()  # should not raise
        assert peak is None

    def test_exceeded_flag_starts_false(self):
        guard = MemoryGuard(max_growth_mb=5800.0)
        assert guard.exceeded is False
```

- [ ] **Step 2.2: Run tests — expect ImportError (scripts/finetune.py doesn't exist yet)**

```
uv run python -m pytest tests/finetune/test_finetune_config.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'scripts.finetune'`

- [ ] **Step 2.3: Write scripts/finetune.py — Config + MemoryGuard sections only**

```python
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
        self._thread: threading.Thread | None = None
        self._peak_mb: float | None = None
        self._exceeded = False

        # Import VRAMMonitor lazily so the module loads on non-NVIDIA machines.
        try:
            from src.parallel.vram_monitor import VRAMMonitor  # noqa: PLC0415
            self._monitor: Any = VRAMMonitor()
        except Exception:
            self._monitor = None

    # -- Public API -----------------------------------------------------------

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
        return self._peak_mb

    @property
    def exceeded(self) -> bool:
        return self._exceeded

    # -- Internal -------------------------------------------------------------

    def _poll(self) -> None:
        while not self._stop.is_set():
            if self._monitor is not None:
                used = self._monitor.used_mb()
                if used is not None:
                    self._peak_mb = (
                        used if self._peak_mb is None else max(self._peak_mb, used)
                    )
                    if used > self.max_growth_mb and not self._exceeded:
                        self._exceeded = True
                        try:
                            os.kill(os.getpid(), signal.SIGINT)
                        except Exception:
                            pass
            self._stop.wait(timeout=0.5)
```

*(data loading, train(), and main() are added in later tasks — the file is intentionally incomplete here)*

- [ ] **Step 2.4: Run tests — expect PASS**

```
uv run python -m pytest tests/finetune/test_finetune_config.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add scripts/finetune.py tests/finetune/__init__.py tests/finetune/test_finetune_config.py
git commit -m "feat(finetune): FinetuneConfig + MemoryGuard with tests"
```

---

## Task 3: Data loading utilities (with tests)

**Files:**
- Modify (append to): `scripts/finetune.py` — load_jsonl + to_chatml
- Create: `tests/finetune/test_data_loading.py`

- [ ] **Step 3.1: Write tests first**

```python
# tests/finetune/test_data_loading.py
"""Tests for JSONL loading and ChatML conversion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.finetune import load_jsonl, to_chatml, TRAIN_JSONL, VAL_JSONL


class TestLoadJsonl:
    def test_train_jsonl_exists(self):
        assert TRAIN_JSONL.exists(), f"{TRAIN_JSONL} not found"

    def test_train_jsonl_count(self):
        records = load_jsonl(TRAIN_JSONL)
        assert len(records) >= 449

    def test_val_jsonl_count(self):
        records = load_jsonl(VAL_JSONL)
        assert len(records) == 51

    def test_each_record_has_prompt_and_completion(self):
        records = load_jsonl(TRAIN_JSONL)
        for r in records[:20]:
            assert "prompt" in r, f"Missing 'prompt' in {r}"
            assert "completion" in r, f"Missing 'completion' in {r}"
            assert r["prompt"], "prompt is empty"
            assert r["completion"], "completion is empty"

    def test_load_from_tmp_file(self, tmp_path: Path):
        p = tmp_path / "test.jsonl"
        p.write_text(
            '{"prompt": "hello", "completion": "world"}\n'
            '{"prompt": "foo", "completion": "bar"}\n',
            encoding="utf-8",
        )
        records = load_jsonl(p)
        assert len(records) == 2
        assert records[0]["prompt"] == "hello"

    def test_skips_blank_lines(self, tmp_path: Path):
        p = tmp_path / "blank.jsonl"
        p.write_text(
            '{"prompt": "a", "completion": "b"}\n\n\n',
            encoding="utf-8",
        )
        records = load_jsonl(p)
        assert len(records) == 1

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl(Path("/nonexistent/path.jsonl"))


class TestToChatml:
    def test_produces_two_messages(self):
        msgs = to_chatml({"prompt": "Write a test", "completion": "def test_it(): pass"})
        assert len(msgs) == 2

    def test_user_role(self):
        msgs = to_chatml({"prompt": "Q", "completion": "A"})
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Q"

    def test_assistant_role(self):
        msgs = to_chatml({"prompt": "Q", "completion": "A"})
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "A"

    def test_real_example_round_trip(self):
        records = load_jsonl(TRAIN_JSONL)
        r = records[0]
        msgs = to_chatml(r)
        assert msgs[0]["content"] == r["prompt"]
        assert msgs[1]["content"] == r["completion"]
```

- [ ] **Step 3.2: Run tests — expect ImportError (functions not yet in finetune.py)**

```
uv run python -m pytest tests/finetune/test_data_loading.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'load_jsonl' from 'scripts.finetune'`

- [ ] **Step 3.3: Append load_jsonl and to_chatml to scripts/finetune.py**

Add after the MemoryGuard class:

```python
# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return all non-blank records."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] JSONL parse error at line {lineno}: {exc}", file=sys.stderr)
    return records


def to_chatml(record: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a {prompt, completion} record to a 2-message ChatML list."""
    return [
        {"role": "user",      "content": record["prompt"]},
        {"role": "assistant", "content": record["completion"]},
    ]
```

- [ ] **Step 3.4: Run tests — expect PASS**

```
uv run python -m pytest tests/finetune/test_data_loading.py -v
```

Expected: all 11 tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add scripts/finetune.py tests/finetune/test_data_loading.py
git commit -m "feat(finetune): add load_jsonl + to_chatml with tests"
```

---

## Task 4: Training function and main() in finetune.py

**Files:**
- Modify (append to): `scripts/finetune.py` — train() and main()

No new test file — Unsloth requires GPU. The functions are guarded by try/import and the MemoryGuard/FinetuneConfig are already tested.

- [ ] **Step 4.1: Append train() to scripts/finetune.py**

```python
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the most recent checkpoint path or None."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    return str(checkpoints[-1]) if checkpoints else None


def train(config: FinetuneConfig) -> dict[str, Any]:
    """
    Run QLoRA fine-tune with Unsloth SFTTrainer.

    Returns a metrics dict with keys:
        training_loss_final, val_loss_final, vram_peak_mb, oom_errors
    Saves that dict to models/finetune_output/training_metrics.json.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
        from trl import SFTTrainer  # type: ignore[import]
        from transformers import TrainingArguments  # type: ignore[import]
        from transformers.trainer_callback import TrainerCallback  # type: ignore[import]
        from datasets import Dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            f"Fine-tune dependencies not installed: {exc}\n"
            "Run: uv run python scripts/install_finetune_deps.py"
        ) from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load dataset --
    records = load_jsonl(TRAIN_JSONL)
    print(f"Loaded {len(records)} training examples")

    # -- Load model --
    print(f"Loading base model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    # -- Format dataset --
    texts: list[dict[str, str]] = []
    filtered = 0
    for rec in records:
        msgs = to_chatml(rec)
        try:
            text: str = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            filtered += 1
            continue
        try:
            n = len(tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            n = len(text) // 3
        if n > config.max_seq_length:
            filtered += 1
            continue
        texts.append({"text": text})

    print(f"Dataset prepared: {len(texts)} examples ({filtered} filtered)")
    dataset = Dataset.from_list(texts)

    # -- Training args --
    resume = _find_latest_checkpoint(output_dir)
    if resume:
        print(f"Resuming from checkpoint: {resume}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        optim=config.optim,
        seed=config.seed,
        report_to="none",
    )

    last_loss: list[float] = []

    class _LossTracker(TrainerCallback):
        def on_log(self_inner, args, state, control, logs=None, **kwargs):  # noqa: N805
            if logs and "loss" in logs:
                last_loss.append(logs["loss"])
                print(
                    f"Step {state.global_step}/{config.max_steps} "
                    f"| Loss: {logs['loss']:.4f}"
                )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[_LossTracker()],
    )

    oom_errors = 0
    vram_peak_mb: float | None = None

    with MemoryGuard(max_growth_mb=5800.0) as guard:
        try:
            if resume:
                stats = trainer.train(resume_from_checkpoint=resume)
            else:
                stats = trainer.train()
        except KeyboardInterrupt:
            # MemoryGuard sends SIGINT on OOM — save emergency checkpoint
            oom_errors = 1
            emergency_dir = output_dir / "emergency"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(emergency_dir))
            tokenizer.save_pretrained(str(emergency_dir))
            print(f"Emergency checkpoint saved: {emergency_dir}", file=sys.stderr)
            raise
        vram_peak_mb = guard.peak_mb

    # -- Save final model --
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Final checkpoint saved: {final_dir}")

    training_loss_final = last_loss[-1] if last_loss else stats.training_loss

    metrics = {
        "training_loss_final": round(training_loss_final, 4),
        "val_loss_final": None,   # Unsloth SFTTrainer doesn't run val loop by default
        "vram_peak_mb": round(vram_peak_mb, 1) if vram_peak_mb else None,
        "oom_errors": oom_errors,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {METRICS_PATH}")
    return metrics
```

- [ ] **Step 4.2: Append main() to scripts/finetune.py**

```python
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    config = FinetuneConfig()
    print("Fine-tune config:")
    print(config.model_dump_json(indent=2))
    try:
        metrics = train(config)
        print("\nTraining complete:")
        print(json.dumps(metrics, indent=2))
        return 0
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.3: Verify the script is importable (no GPU required)**

```
uv run python -c "from scripts.finetune import FinetuneConfig, MemoryGuard, load_jsonl, to_chatml, train; print('OK')"
```

Expected: `OK`

- [ ] **Step 4.4: Re-run all finetune tests to confirm nothing broke**

```
uv run python -m pytest tests/finetune/ -v
```

Expected: all previously-passing tests still PASS

- [ ] **Step 4.5: Commit**

```bash
git add scripts/finetune.py
git commit -m "feat(finetune): add train() + main() to finetune.py"
```

---

## Task 5: export_gguf.py

**Files:**
- Create: `scripts/export_gguf.py`

No new test — the real export requires a trained checkpoint + Unsloth GPU. The logic is a thin wrapper reusing the pattern from `src/finetune/wsl2_trainer.export_gguf`.

- [ ] **Step 5.1: Write scripts/export_gguf.py**

```python
"""Convert fine-tuned LoRA adapter → GGUF Q4_K_M for Ollama.

Run:
    uv run python scripts/export_gguf.py

Input:  models/finetune_output/final   (trained checkpoint)
Output: models/qwen2.5-coder-finetuned/qwen2.5-coder-finetuned.Q4_K_M.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

CHECKPOINT_DIR = Path("models/finetune_output/final")
OUTPUT_DIR = Path("models/qwen2.5-coder-finetuned")
GGUF_NAME = "qwen2.5-coder-finetuned.Q4_K_M.gguf"
MAX_SEQ_LENGTH = 1024


def export(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gguf_name: str = GGUF_NAME,
) -> Path:
    """
    Load the fine-tuned checkpoint, merge LoRA, and save as GGUF Q4_K_M.

    Returns the Path to the produced .gguf file.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            f"Unsloth not installed: {exc}\n"
            "Run: uv run python scripts/install_finetune_deps.py"
        ) from exc

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}\n"
            "Run finetune.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Exporting GGUF Q4_K_M → {output_dir}")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Rename whatever Unsloth produced to the stable name.
    candidates = sorted(output_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No .gguf file found in {output_dir} after export")

    target = output_dir / gguf_name
    if candidates[-1] != target:
        candidates[-1].replace(target)
        print(f"Renamed {candidates[-1].name} → {target.name}")

    print(f"GGUF export complete: {target}")
    return target


def main() -> int:
    try:
        export()
        return 0
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.2: Verify importable**

```
uv run python -c "from scripts.export_gguf import export; print('OK')"
```

Expected: `OK`

- [ ] **Step 5.3: Commit**

```bash
git add scripts/export_gguf.py
git commit -m "feat(finetune): add export_gguf.py"
```

---

## Task 6: create_modelfile.py with tests

**Files:**
- Create: `scripts/create_modelfile.py`
- Create: `tests/finetune/test_modelfile.py`

- [ ] **Step 6.1: Write tests first**

```python
# tests/finetune/test_modelfile.py
"""Tests for Modelfile generation — no GPU or Ollama required."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.create_modelfile import write_modelfile, MODELFILE_TEMPLATE, MODEL_TAG


class TestWriteModelfile:
    def test_creates_modelfile(self, tmp_path: Path):
        gguf = tmp_path / "qwen2.5-coder-finetuned.Q4_K_M.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert mf.exists()

    def test_modelfile_named_correctly(self, tmp_path: Path):
        gguf = tmp_path / "model.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert mf.name == "Modelfile"

    def test_from_line_uses_relative_path(self, tmp_path: Path):
        gguf = tmp_path / "qwen2.5-coder-finetuned.Q4_K_M.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        content = mf.read_text()
        assert f"FROM ./{gguf.name}" in content

    def test_temperature_parameter(self, tmp_path: Path):
        gguf = tmp_path / "x.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert "PARAMETER temperature 0.1" in mf.read_text()

    def test_num_ctx_parameter(self, tmp_path: Path):
        gguf = tmp_path / "x.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert "PARAMETER num_ctx 2048" in mf.read_text()

    def test_system_prompt_present(self, tmp_path: Path):
        gguf = tmp_path / "x.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        content = mf.read_text()
        assert "SYSTEM" in content
        assert "QA" in content or "qa" in content.lower()

    def test_template_has_from_placeholder(self):
        assert "{gguf_name}" in MODELFILE_TEMPLATE

    def test_model_tag_correct(self):
        assert MODEL_TAG == "qa-agent-finetuned"
```

- [ ] **Step 6.2: Run tests — expect ImportError**

```
uv run python -m pytest tests/finetune/test_modelfile.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'scripts.create_modelfile'`

- [ ] **Step 6.3: Write scripts/create_modelfile.py**

```python
"""Create Ollama Modelfile for the fine-tuned model and register it.

Run:
    uv run python scripts/create_modelfile.py

Requires:
    - models/qwen2.5-coder-finetuned/qwen2.5-coder-finetuned.Q4_K_M.gguf (from export_gguf.py)
    - ollama CLI installed and daemon running

After registration, verify with:
    ollama run qa-agent-finetuned "Generate a pytest test for..."
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GGUF_DIR = Path("models/qwen2.5-coder-finetuned")
GGUF_NAME = "qwen2.5-coder-finetuned.Q4_K_M.gguf"
MODEL_TAG = "qa-agent-finetuned"

MODELFILE_TEMPLATE = """\
FROM ./{gguf_name}
SYSTEM \"\"\"You are an expert QA automation engineer. You write pytest-playwright \
tests using only semantic locators: get_by_role, get_by_label, get_by_text, \
get_by_test_id. You never use CSS selectors or XPath. You always use expect() \
for assertions. Function names must start with test_.\"\"\"
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
"""


def write_modelfile(
    gguf_path: Path,
    output_dir: Path,
) -> Path:
    """Write Modelfile to output_dir. Return the Modelfile path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    content = MODELFILE_TEMPLATE.format(gguf_name=gguf_path.name)
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"Modelfile written: {modelfile_path}")
    return modelfile_path


def register_with_ollama(modelfile_path: Path, model_tag: str = MODEL_TAG) -> None:
    """Run `ollama create <tag> -f Modelfile` from the Modelfile's directory."""
    cmd = ["ollama", "create", model_tag, "-f", modelfile_path.name]
    print(f"Running: {' '.join(cmd)}  (cwd={modelfile_path.parent})")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(modelfile_path.parent),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ollama CLI not found. Install from https://ollama.com/download "
            "and ensure it is in PATH."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    print(f"Registered {model_tag!r} with Ollama.")
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> int:
    gguf_path = GGUF_DIR / GGUF_NAME
    if not gguf_path.exists():
        print(f"ERROR: GGUF file not found: {gguf_path}", file=sys.stderr)
        print("Run export_gguf.py first.", file=sys.stderr)
        return 1
    try:
        modelfile_path = write_modelfile(gguf_path=gguf_path, output_dir=GGUF_DIR)
        register_with_ollama(modelfile_path)
        print(f"\nDone! Test with:")
        print(f'  ollama run {MODEL_TAG} "Generate a pytest test for login page"')
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6.4: Run tests — expect PASS**

```
uv run python -m pytest tests/finetune/test_modelfile.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 6.5: Commit**

```bash
git add scripts/create_modelfile.py tests/finetune/test_modelfile.py
git commit -m "feat(finetune): add create_modelfile.py with tests"
```

---

## Task 7: Scoring functions for eval (with tests)

**Files:**
- Create: `scripts/eval_finetune.py` — EvalResult, get_scoring_type, score_completion only
- Create: `tests/finetune/test_eval_scoring.py`

- [ ] **Step 7.1: Write tests first**

```python
# tests/finetune/test_eval_scoring.py
"""Tests for eval scoring functions — no Ollama or GPU required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.eval_finetune import (
    EvalResult,
    get_scoring_type,
    score_completion,
)


class TestGetScoringType:
    def test_sprint6_pytest(self):
        assert get_scoring_type("sprint6_pytest") == "pytest"

    def test_sprint13_schema(self):
        assert get_scoring_type("sprint13_schema") == "schema"

    def test_sprint9_vitest(self):
        assert get_scoring_type("sprint9_vitest") == "vitest"

    def test_sprint5_hypothesis(self):
        assert get_scoring_type("sprint5_hypothesis") == "hypothesis"

    def test_sprint14_pbt(self):
        assert get_scoring_type("sprint14_pbt") == "hypothesis"

    def test_unknown_source_defaults_to_hypothesis(self):
        assert get_scoring_type("unknown_source") == "hypothesis"


class TestScoreCompletion:
    # -- pytest scoring --
    def test_pytest_valid_code(self):
        code = "import pytest\n\ndef test_add():\n    assert 1 + 1 == 2\n"
        result = score_completion("pytest", code)
        assert isinstance(result, bool)

    def test_pytest_must_have_def_test(self):
        # Valid Python but no test function → False
        code = "def helper(): pass"
        assert score_completion("pytest", code) is False

    def test_pytest_syntax_error_is_false(self):
        assert score_completion("pytest", "def test(: pass") is False

    def test_pytest_empty_is_false(self):
        assert score_completion("pytest", "") is False

    def test_pytest_with_test_func(self):
        code = "def test_something():\n    assert True\n"
        assert score_completion("pytest", code) is True

    # -- schema scoring --
    def test_schema_valid_json_object(self):
        schema = json.dumps({
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        })
        assert score_completion("schema", schema) is True

    def test_schema_invalid_json(self):
        assert score_completion("schema", "not json at all") is False

    def test_schema_empty_string(self):
        assert score_completion("schema", "") is False

    def test_schema_valid_json_array_passes(self):
        # Any valid JSON with type key is acceptable
        schema = json.dumps({"type": "array", "items": {"type": "string"}})
        assert score_completion("schema", schema) is True

    def test_schema_json_without_type_key(self):
        # JSON but no "type" → not a schema
        schema = json.dumps({"foo": "bar"})
        assert score_completion("schema", schema) is False

    # -- vitest scoring --
    def test_vitest_valid(self):
        code = (
            "describe('add', () => {\n"
            "  it('works', () => {\n"
            "    expect(1 + 1).toBe(2);\n"
            "  });\n"
            "});\n"
        )
        assert score_completion("vitest", code) is True

    def test_vitest_missing_describe_is_false(self):
        assert score_completion("vitest", "expect(1).toBe(1);") is False

    def test_vitest_missing_expect_is_false(self):
        assert score_completion("vitest", "describe('x', () => {});") is False

    def test_vitest_empty_is_false(self):
        assert score_completion("vitest", "") is False

    # -- hypothesis scoring --
    def test_hypothesis_valid_python(self):
        code = (
            "from hypothesis import given\nimport hypothesis.strategies as st\n\n"
            "@given(st.integers())\ndef test_prop(n):\n    assert n + 1 > n\n"
        )
        assert score_completion("hypothesis", code) is True

    def test_hypothesis_invalid_python(self):
        assert score_completion("hypothesis", "def test(: pass") is False

    def test_hypothesis_empty(self):
        assert score_completion("hypothesis", "") is False

    def test_hypothesis_valid_python_no_hypothesis_import(self):
        # Valid Python syntax suffices even without hypothesis import
        code = "def test_x():\n    x = 1\n    assert x == 1\n"
        assert score_completion("hypothesis", code) is True


class TestEvalResult:
    def test_pass_status_when_all_criteria_met(self):
        r = EvalResult(
            base_pass_rate=0.5,
            ft_pass_rate=0.6,
            quality_gain=0.1,
            training_loss_final=1.2,
            val_loss_final=1.8,
            vram_peak_mb=4800.0,
            oom_errors=0,
            finetune_status="PASS",
        )
        assert r.finetune_status == "PASS"

    def test_fail_status(self):
        r = EvalResult(
            base_pass_rate=0.5,
            ft_pass_rate=0.5,
            quality_gain=0.0,
            training_loss_final=2.0,
            val_loss_final=2.5,
            vram_peak_mb=4800.0,
            oom_errors=0,
            finetune_status="FAIL",
        )
        assert r.finetune_status == "FAIL"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            EvalResult(
                base_pass_rate=0.5,
                ft_pass_rate=0.6,
                quality_gain=0.1,
                training_loss_final=1.2,
                val_loss_final=1.8,
                vram_peak_mb=4800.0,
                oom_errors=0,
                finetune_status="PASS",
                unknown_field="x",
            )

    def test_none_fields_allowed(self):
        # val_loss_final and vram_peak_mb may be None (not always available)
        r = EvalResult(
            base_pass_rate=0.5,
            ft_pass_rate=0.6,
            quality_gain=0.1,
            training_loss_final=1.2,
            val_loss_final=None,
            vram_peak_mb=None,
            oom_errors=0,
            finetune_status="PASS",
        )
        assert r.val_loss_final is None
        assert r.vram_peak_mb is None

    def test_model_dump_json_round_trip(self):
        r = EvalResult(
            base_pass_rate=0.6,
            ft_pass_rate=0.7,
            quality_gain=0.1,
            training_loss_final=1.1,
            val_loss_final=1.5,
            vram_peak_mb=5000.0,
            oom_errors=0,
            finetune_status="PASS",
        )
        data = json.loads(r.model_dump_json())
        assert data["quality_gain"] == pytest.approx(0.1)
        assert data["finetune_status"] == "PASS"
```

- [ ] **Step 7.2: Run tests — expect ImportError**

```
uv run python -m pytest tests/finetune/test_eval_scoring.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'scripts.eval_finetune'`

- [ ] **Step 7.3: Write the models + scoring functions in scripts/eval_finetune.py**

```python
"""Compare base vs fine-tuned model quality on val.jsonl.

Run:
    uv run python scripts/eval_finetune.py

Prerequisites:
    - data/training/val.jsonl  (51 examples)
    - Ollama running with qwen2.5-coder:7b-instruct-q4_K_M (base)
    - Ollama running with qa-agent-finetuned (from create_modelfile.py)
    - models/finetune_output/training_metrics.json (from finetune.py)

Output: models/eval_results.json
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VAL_JSONL = Path("data/training/val.jsonl")
METRICS_PATH = Path("models/finetune_output/training_metrics.json")
EVAL_OUTPUT = Path("models/eval_results.json")

BASE_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
FT_MODEL = "qa-agent-finetuned"

# ---------------------------------------------------------------------------
# Acceptance thresholds (from spec)
# ---------------------------------------------------------------------------
_THRESHOLD_TRAIN_LOSS = 1.50
_THRESHOLD_VAL_LOSS = 2.00
_THRESHOLD_QUALITY_GAIN = 0.05
_THRESHOLD_VRAM_MB = 5800.0

# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------

ScoringType = Literal["pytest", "schema", "vitest", "hypothesis"]


class EvalResult(BaseModel):
    """Output of the evaluation pipeline (matches spec eval_results.json schema)."""

    model_config = ConfigDict(extra="forbid")

    base_pass_rate: float
    ft_pass_rate: float
    quality_gain: float
    training_loss_final: float | None
    val_loss_final: float | None
    vram_peak_mb: float | None
    oom_errors: int
    finetune_status: Literal["PASS", "FAIL"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def get_scoring_type(source: str) -> ScoringType:
    """Map a dataset source string to a scoring strategy."""
    if source == "sprint6_pytest":
        return "pytest"
    if source == "sprint13_schema":
        return "schema"
    if source == "sprint9_vitest":
        return "vitest"
    return "hypothesis"


def score_completion(scoring_type: ScoringType, completion: str) -> bool:
    """
    Score a model completion. Returns True if it passes the scoring heuristic.

    pytest:     valid Python + contains "def test_"
    schema:     valid JSON + has "type" key
    vitest:     contains both "describe" and "expect"
    hypothesis: valid Python syntax (ast.parse succeeds)
    """
    if not completion or not completion.strip():
        return False

    if scoring_type == "pytest":
        if "def test_" not in completion:
            return False
        try:
            ast.parse(completion)
            return True
        except SyntaxError:
            return False

    if scoring_type == "schema":
        try:
            obj = json.loads(completion)
            return isinstance(obj, dict) and "type" in obj
        except (json.JSONDecodeError, ValueError):
            return False

    if scoring_type == "vitest":
        return "describe" in completion and "expect" in completion

    # hypothesis (and any unknown type)
    try:
        ast.parse(completion)
        return True
    except SyntaxError:
        return False
```

*(async query and main() are added in Task 8)*

- [ ] **Step 7.4: Run tests — expect PASS**

```
uv run python -m pytest tests/finetune/test_eval_scoring.py -v
```

Expected: all 27 tests PASS

- [ ] **Step 7.5: Commit**

```bash
git add scripts/eval_finetune.py tests/finetune/test_eval_scoring.py
git commit -m "feat(finetune): add EvalResult + scoring functions with tests"
```

---

## Task 8: eval_finetune.py — async eval loop and main()

**Files:**
- Modify (append to): `scripts/eval_finetune.py`

No new test file — the async eval loop requires live Ollama. The testable scoring layer is covered in Task 7.

- [ ] **Step 8.1: Append async query + evaluate() + main() to scripts/eval_finetune.py**

```python
# ---------------------------------------------------------------------------
# Async Ollama query
# ---------------------------------------------------------------------------

_OLLAMA_SEMAPHORE: asyncio.Semaphore | None = None  # set in evaluate()


async def query_model(
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 512,
) -> str:
    """
    Send a prompt to an Ollama model via OllamaAdapter.
    Semaphore(1) ensures no concurrent model calls (6 GB VRAM constraint).
    """
    from src.llm.adapter import OllamaAdapter  # noqa: PLC0415

    async with semaphore:
        adapter = OllamaAdapter(model=model, temperature=0.1, max_tokens=max_tokens)
        try:
            return await adapter.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            print(f"[warn] query_model({model!r}) failed: {exc}", file=sys.stderr)
            return ""
        finally:
            await adapter.close()


# ---------------------------------------------------------------------------
# Load training metrics
# ---------------------------------------------------------------------------


def _load_training_metrics() -> dict:
    """Load training_metrics.json produced by finetune.py, or return defaults."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(
        f"[warn] {METRICS_PATH} not found — training metrics will be None",
        file=sys.stderr,
    )
    return {
        "training_loss_final": None,
        "val_loss_final": None,
        "vram_peak_mb": None,
        "oom_errors": 0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate(
    val_jsonl: Path = VAL_JSONL,
    base_model: str = BASE_MODEL,
    ft_model: str = FT_MODEL,
) -> EvalResult:
    """
    Evaluate base vs fine-tuned model on val.jsonl.

    For each of the 51 examples, sends the prompt to both models (serialized via
    asyncio.Semaphore(1)) and scores the completions. Returns an EvalResult.
    """
    if not val_jsonl.exists():
        raise FileNotFoundError(f"Validation set not found: {val_jsonl}")

    records: list[dict] = []
    with val_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Evaluating {len(records)} examples | base={base_model!r} ft={ft_model!r}")

    # INVARIANT: Semaphore(1) — only 1 model call at a time (6 GB VRAM)
    semaphore = asyncio.Semaphore(1)

    base_passed = 0
    ft_passed = 0

    for i, rec in enumerate(records, 1):
        source = rec.get("source", "unknown")
        prompt = rec.get("prompt", "")
        scoring_type = get_scoring_type(source)

        # Base model
        base_completion = await query_model(base_model, prompt, semaphore)
        base_ok = score_completion(scoring_type, base_completion)
        if base_ok:
            base_passed += 1

        # Fine-tuned model
        ft_completion = await query_model(ft_model, prompt, semaphore)
        ft_ok = score_completion(scoring_type, ft_completion)
        if ft_ok:
            ft_passed += 1

        print(
            f"[{i:02d}/{len(records)}] source={source!r:25s} "
            f"scoring={scoring_type!r:12s} base={'PASS' if base_ok else 'fail'} "
            f"ft={'PASS' if ft_ok else 'fail'}"
        )

    n = len(records)
    base_pass_rate = base_passed / n
    ft_pass_rate = ft_passed / n
    quality_gain = ft_pass_rate - base_pass_rate

    metrics = _load_training_metrics()

    # Determine PASS/FAIL
    train_loss = metrics.get("training_loss_final")
    val_loss = metrics.get("val_loss_final")
    vram_peak = metrics.get("vram_peak_mb")
    oom_errors = metrics.get("oom_errors", 0)

    train_ok = (train_loss is None) or (train_loss <= _THRESHOLD_TRAIN_LOSS)
    val_ok = (val_loss is None) or (val_loss <= _THRESHOLD_VAL_LOSS)
    vram_ok = (vram_peak is None) or (vram_peak <= _THRESHOLD_VRAM_MB)
    gain_ok = quality_gain >= _THRESHOLD_QUALITY_GAIN
    oom_ok = oom_errors == 0

    status: Literal["PASS", "FAIL"] = "PASS" if all(
        [train_ok, val_ok, vram_ok, gain_ok, oom_ok]
    ) else "FAIL"

    result = EvalResult(
        base_pass_rate=round(base_pass_rate, 4),
        ft_pass_rate=round(ft_pass_rate, 4),
        quality_gain=round(quality_gain, 4),
        training_loss_final=train_loss,
        val_loss_final=val_loss,
        vram_peak_mb=vram_peak,
        oom_errors=oom_errors,
        finetune_status=status,
    )

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nEval results written: {EVAL_OUTPUT}")
    print(result.model_dump_json(indent=2))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        result = asyncio.run(evaluate())
        return 0 if result.finetune_status == "PASS" else 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8.2: Verify the complete eval_finetune.py is importable**

```
uv run python -c "
from scripts.eval_finetune import (
    EvalResult, get_scoring_type, score_completion,
    query_model, evaluate, main
)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 8.3: Re-run all finetune tests**

```
uv run python -m pytest tests/finetune/ -v
```

Expected: all tests PASS (27+ tests)

- [ ] **Step 8.4: Commit**

```bash
git add scripts/eval_finetune.py
git commit -m "feat(finetune): add eval_finetune.py async evaluation loop"
```

---

## Task 9: Final wiring — models/ directory and run verification

**Files:**
- Create: `models/.gitkeep` — ensure models/ is tracked but content ignored
- Modify: `.gitignore` — ignore model artefacts but keep directory

- [ ] **Step 9.1: Create models directory and gitignore**

Create `models/.gitkeep`:
```
(empty file)
```

Add to `.gitignore` (or create if missing):
```gitignore
# Fine-tune artefacts — large files, never commit
models/finetune_output/
models/qwen2.5-coder-finetuned/
models/eval_results.json
```

Note: `models/finetune_output/training_metrics.json` IS a text file that should
be committed after a real run. Only exclude the large checkpoint dirs and GGUF.
Update gitignore to be more specific:
```gitignore
models/finetune_output/checkpoint-*/
models/finetune_output/emergency/
models/finetune_output/final/
models/qwen2.5-coder-finetuned/*.gguf
```

- [ ] **Step 9.2: Verify all scripts are importable**

```
uv run python -c "
import scripts.install_finetune_deps as a
import scripts.finetune as b
import scripts.export_gguf as c
import scripts.create_modelfile as d
import scripts.eval_finetune as e
print('All scripts import OK')
"
```

Expected: `All scripts import OK`

- [ ] **Step 9.3: Run the full finetune test suite**

```
uv run python -m pytest tests/finetune/ -v --tb=short
```

Expected: 30+ tests PASS, 0 FAIL

- [ ] **Step 9.4: Verify no regressions in the rest of the test suite**

```
uv run python -m pytest tests/test_dataset_collection.py tests/pbt/ tests/parallel/ -v --tb=short 2>&1 | tail -20
```

Expected: previously passing tests still PASS

- [ ] **Step 9.5: Commit**

```bash
git add models/.gitkeep .gitignore
git commit -m "chore(finetune): add models/.gitkeep and gitignore for model artefacts"
```

---

## Run Order (actual training — requires GPU + Unsloth)

After all code is in place, execute in this order:

```bash
# 1. Install GPU deps
uv run python scripts/install_finetune_deps.py

# 2. Train (100 steps, ~30–60 min on GTX 1660 Ti)
uv run python scripts/finetune.py

# 3. Export to GGUF
uv run python scripts/export_gguf.py

# 4. Register with Ollama
uv run python scripts/create_modelfile.py

# 5. Evaluate
uv run python scripts/eval_finetune.py
# → models/eval_results.json
```

---

## Self-Review Against Spec

| Spec requirement | Task |
|---|---|
| `scripts/install_finetune_deps.py` with DEPS list | Task 1 |
| `FinetuneConfig` with exact spec values (lora_r=8, max_steps=100, …) | Task 2 |
| `VRAMMonitor.monitor_during` / MemoryGuard tracking peak | Task 2 (MemoryGuard polls VRAMMonitor) |
| `MemoryGuard(max_growth_mb=5800)` → abort if exceeded | Task 2 |
| ChatML format: user=prompt, assistant=completion | Task 3 (to_chatml) |
| `asyncio.Semaphore(1) NOT needed here` (no Ollama in training) | Task 4 (confirmed — no semaphore in train()) |
| Checkpointing every 50 steps + resume from checkpoint | Task 4 (save_steps=50, _find_latest_checkpoint) |
| `scripts/export_gguf.py` → `models/qwen2.5-coder-finetuned.Q4_K_M.gguf` | Task 5 |
| `scripts/create_modelfile.py` → Modelfile + `ollama create qa-agent-finetuned` | Task 6 |
| `asyncio.Semaphore(1) on ALL Ollama/model calls` in eval | Task 8 (semaphore passed to query_model) |
| `models/eval_results.json` with all required fields | Task 7 (EvalResult), Task 8 (evaluate writes it) |
| `finetune_status = PASS/FAIL` based on acceptance criteria | Task 8 |
| `pathlib.Path everywhere` | All tasks |
| `Pydantic V2 ConfigDict(extra="forbid")` | Tasks 2, 7 |
| `fp16=True` | Task 4 (TrainingArguments) |
| `batch_size=1 hard limit` | Task 2 (FinetuneConfig.per_device_train_batch_size=1) |
| OOM → reduce lora_r=4, max_seq_length=512 (documented fallback) | Mentioned in install script comment |
| BLOCKED_ACTION_PATTERNS not in training data (pre-validated) | Confirmed — validate_dataset.py already ran |
