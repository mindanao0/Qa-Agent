"""
QLoRA fine-tuning pipeline using Unsloth (UPDATE 3 specification).

Hardware requirements:
  - VRAM: >= 5.5 GB free before training starts.  The 4-bit base model occupies
    ~4.5 GB, leaving ~1.5 GB for activations + gradient buffers at
    batch_size=1 + grad_accum=4.
  - System RAM: >= 12 GB recommended.

Configuration (matches UPDATE 3 spec exactly):
  base_model            unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit
  max_seq_length        2048
  load_in_4bit          True
  lora_r                16
  lora_alpha            32
  target_modules        q_proj v_proj k_proj o_proj gate_proj up_proj down_proj
  lora_dropout          0.05
  bias                  none
  use_gradient_checkpointing  "unsloth"
  per_device_train_batch_size 1
  gradient_accumulation_steps 4
  warmup_steps          10
  max_steps             300
  learning_rate         2e-4
  fp16                  True   (broader GPU support than bf16)
  logging_steps         10
  save_steps            50

Install the optional fine-tune extras before running:
    pip install "qa-agent[finetune]"   # or: pip install unsloth[colab-new]

Usage:
    trainer = QLoRATrainer()
    trainer.run("~/.qa-agent/datasets/synthetic_universal.jsonl")
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Optional Unsloth import — deferred to .load_model() so the module is always
# importable on machines without CUDA / unsloth available.
# ──────────────────────────────────────────────────────────────────────────────

_UNSLOTH_AVAILABLE = False
try:
    import unsloth  # noqa: F401
    _UNSLOTH_AVAILABLE = True
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
_DEFAULT_CHECKPOINT_DIR = "~/.qa-agent/checkpoints"
_VRAM_MIN_GB = 5.5
_MAX_SEQ_LENGTH = 2048


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"JSONL parse error at line {lineno}: {exc}")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────


class QLoRATrainer:
    """
    Unsloth + TRL SFTTrainer wrapper for 4-bit QLoRA fine-tuning of
    Qwen2.5-Coder-7B on a 6 GB GPU.

    Public API:
      check_vram()         → returns free VRAM in GB (raises if < 5.5)
      load_model()         → loads the 4-bit base + applies LoRA adapters
      prepare_dataset(p)   → loads JSONL → HF Dataset using chat template
      train(dataset)       → runs SFTTrainer with the spec's hyper-parameters
      run(jsonl_path)      → orchestrates the four steps above
    """

    def __init__(
        self,
        base_model: str = _BASE_MODEL,
        checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
        max_seq_length: int = _MAX_SEQ_LENGTH,
    ) -> None:
        self.base_model = base_model
        self.checkpoint_dir = Path(os.path.expanduser(checkpoint_dir))
        self.max_seq_length = max_seq_length

        # Populated by load_model()
        self.model: Any = None
        self.tokenizer: Any = None

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — VRAM safety guard
    # ──────────────────────────────────────────────────────────────────────────

    def check_vram(self) -> float:
        """
        Return free VRAM (GB).  Raises RuntimeError if < 5.5 GB.

        Uses `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
        which returns free memory in MB; we convert to GB.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            logger.warning(
                "nvidia-smi not found — cannot verify VRAM, proceeding at your own risk"
            )
            return 0.0
        except Exception as exc:
            logger.warning(f"VRAM probe failed: {exc} — proceeding without guard")
            return 0.0

        if result.returncode != 0:
            logger.warning(
                f"nvidia-smi returned {result.returncode}: {result.stderr.strip()} — "
                "proceeding without VRAM guard"
            )
            return 0.0

        line = result.stdout.strip().splitlines()[0]
        try:
            free_mb = int(line.strip())
        except ValueError:
            logger.warning(f"Could not parse nvidia-smi output: {line!r}")
            return 0.0

        free_gb = free_mb / 1024.0
        logger.info(f"VRAM check: {free_gb:.2f} GB free (need {_VRAM_MIN_GB} GB)")
        if free_gb < _VRAM_MIN_GB:
            raise RuntimeError(
                f"Insufficient VRAM: {free_gb:.2f}GB free, need {_VRAM_MIN_GB}GB minimum. "
                "Close other GPU applications and retry."
            )
        return free_gb

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — model + LoRA
    # ──────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the 4-bit base model and attach LoRA adapters."""
        if not _UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Run:\n"
                '  pip install "unsloth[colab-new]>=2024.12.0"\n'
                "Requires CUDA 11.8+ and a compatible GPU."
            )

        from unsloth import FastLanguageModel  # type: ignore[import]

        logger.info(f"Loading base model: {self.base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            dtype=None,           # auto-detect bfloat16 / float16
            load_in_4bit=True,
        )

        logger.info("Applying LoRA (r=16, alpha=32, dropout=0.05)")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        self.model = model
        self.tokenizer = tokenizer

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — dataset preparation
    # ──────────────────────────────────────────────────────────────────────────

    def prepare_dataset(self, jsonl_path: str | Path) -> Any:
        """
        Load *jsonl_path*, convert ChatML to text via the tokenizer's chat
        template, filter out examples that exceed `max_seq_length` after
        tokenisation, and return a HuggingFace Dataset.
        """
        if self.tokenizer is None:
            raise RuntimeError("prepare_dataset called before load_model()")

        path = Path(os.path.expanduser(str(jsonl_path)))
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        records = _load_jsonl(path)
        if not records:
            raise ValueError(f"Dataset is empty: {path}")
        logger.info(f"Loaded {len(records)} raw examples from {path}")

        formatted: list[dict[str, str]] = []
        filtered_out = 0
        for rec in records:
            messages = rec.get("messages") or []
            if not messages:
                filtered_out += 1
                continue
            try:
                text: str = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                logger.debug(f"apply_chat_template failed: {exc}")
                filtered_out += 1
                continue

            # Length filter (tokenised)
            try:
                token_count = len(self.tokenizer(text, add_special_tokens=False).input_ids)
            except Exception:
                token_count = len(text) // 3   # rough fallback
            if token_count > self.max_seq_length:
                filtered_out += 1
                continue

            formatted.append({"text": text})

        logger.info(
            f"Dataset prepared: total={len(records)} "
            f"filtered_out={filtered_out} final={len(formatted)}"
        )

        from datasets import Dataset  # type: ignore[import]
        return Dataset.from_list(formatted)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, dataset: Any) -> Path:
        """
        Run SFTTrainer for max_steps and save a final checkpoint.

        Returns the path to the saved final checkpoint directory.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("train called before load_model()")

        from trl import SFTTrainer                      # type: ignore[import]
        from transformers import TrainingArguments      # type: ignore[import]
        from transformers.trainer_callback import (     # type: ignore[import]
            TrainerCallback,
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=300,
            learning_rate=2e-4,
            fp16=True,           # broader GPU support than bf16
            logging_steps=10,
            save_steps=50,
            save_total_limit=3,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        )

        # Lightweight progress logger per spec ("Step X/300 | Loss: … | LR: …")
        class _StepLogger(TrainerCallback):
            def on_log(self_inner, args, state, control, logs=None, **kwargs):  # noqa: N805
                if not logs:
                    return
                loss = logs.get("loss")
                lr = logs.get("learning_rate")
                if loss is not None and lr is not None:
                    logger.info(
                        f"Step {state.global_step}/{args.max_steps} | "
                        f"Loss: {loss:.4f} | LR: {lr:.2e}"
                    )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
            callbacks=[_StepLogger()],
        )

        logger.info("Starting training (max_steps=300)…")
        stats = trainer.train()
        logger.info(
            f"Training complete | loss={stats.training_loss:.4f} "
            f"steps={stats.global_step}"
        )

        final_dir = self.checkpoint_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Checkpoint saved: {final_dir}")
        return final_dir

    # ──────────────────────────────────────────────────────────────────────────
    # Orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, jsonl_path: str | Path) -> Path:
        """
        Full pipeline: VRAM guard → load model → prepare dataset → train.
        Returns the path of the saved final checkpoint.
        """
        self.check_vram()
        self.load_model()
        dataset = self.prepare_dataset(jsonl_path)
        final_dir = self.train(dataset)
        logger.info(
            "Training complete. Run finetune --step export to create the Ollama model."
        )
        return final_dir

    # ──────────────────────────────────────────────────────────────────────────
    # Backwards-compatibility shim (older main.py called `.train(dataset_path)`)
    # ──────────────────────────────────────────────────────────────────────────

    def train_from_path(self, jsonl_path: str | Path) -> Path:
        """Alias for .run(jsonl_path) — keeps older call sites working."""
        return self.run(jsonl_path)
