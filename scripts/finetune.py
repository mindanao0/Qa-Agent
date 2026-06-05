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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(output_dir: Path) -> str | None:
    """Return the most recent checkpoint path string, or None if none exist."""
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

    metrics: dict[str, Any] = {
        "training_loss_final": round(training_loss_final, 4),
        "val_loss_final": None,
        "vram_peak_mb": round(vram_peak_mb, 1) if vram_peak_mb is not None else None,
        "oom_errors": oom_errors,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved: {METRICS_PATH}")
    return metrics


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
