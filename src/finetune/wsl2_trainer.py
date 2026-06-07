#!/usr/bin/env python3
"""
WSL2 QLoRA trainer — 6 GB VRAM safe (UPDATE 4 specification).

This script is meant to run *inside* the WSL2 venv created by
`scripts/setup_wsl2_cuda.sh` (not the Windows project venv).  It is independent
of the async qa-agent runtime: only torch / unsloth / transformers / trl /
datasets are required.

Key design choices vs. the Windows `trainer.py`:

  * max_seq_length = 1024   (saves ~1 GB activation memory vs. 2048)
  * LoRA r = 8, alpha = 16
  * target_modules = ["q_proj", "v_proj"]   (minimal — fits in 6 GB)
  * gradient_accumulation_steps = 8        (effective batch 8)
  * fp16 = True                            (broader GPU support than bf16)
  * Background VRAMWatchdog thread that polls torch.cuda.mem_get_info() and
    forces a graceful stop if free VRAM drops below the critical threshold.
  * Catastrophic-forgetting mitigation: ~10% general instruction-following
    examples interleaved with the QA-specific examples (1-in-10 ratio).
  * KeyboardInterrupt handler saves an emergency checkpoint before exit.
  * --export-only mode: loads a checkpoint with FastLanguageModel and writes
    GGUF Q4_K_M plus the Modelfile that `register_ollama_windows.ps1` consumes.

CLI:

    python3 wsl2_trainer.py --dataset PATH --output PATH [--max-steps N]
    python3 wsl2_trainer.py --export-only --checkpoint PATH --output PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ── Logging (loguru is not available in the WSL2 training venv) ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wsl2_trainer")

# ── Constants (6 GB safe) ────────────────────────────────────────────────────
BASE_MODEL          = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"   # 7B fills all VRAM; 1.5B leaves ~4GB for compute
MAX_SEQ_LENGTH      = 1024
LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.0    # Unsloth recommendation for 4-bit
TARGET_MODULES      = ["q_proj", "v_proj"]
BATCH_SIZE          = 1
GRAD_ACCUM_STEPS    = 8
DEFAULT_MAX_STEPS   = 300
LEARNING_RATE       = 2e-4
WARMUP_STEPS        = 10
LOG_STEPS           = 10
SAVE_STEPS          = 50

VRAM_WARN_GB        = 0.15   # 150 MB free → warning (tighter — seq_len=512 uses less)
VRAM_CRIT_GB        = 0.05   # 50 MB free → emergency stop (PyTorch OOM fires before this)
VRAM_LOG_EVERY_SEC  = 30
VRAM_POLL_EVERY_SEC = 5

GGUF_FILE_NAME      = "qa-agent-coder-q4_k_m.gguf"
GGUF_QUANT_METHOD   = "q4_k_m"

MODELFILE_BODY = """\
FROM ./{gguf_name}
SYSTEM \"\"\"You are an expert Playwright Python QA engineer. Generate pytest-playwright
tests using only semantic locators: get_by_role, get_by_label, get_by_text,
get_by_test_id. Never use CSS selectors or XPath. Always use expect() for assertions.
Function names must start with test_.\"\"\"
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
"""

# ── Hard-coded general examples (catastrophic-forgetting mitigation) ─────────
GENERAL_EXAMPLES: list[dict[str, str]] = [
    {
        "instruction": "What is Playwright?",
        "response": (
            "Playwright is a browser automation library that drives Chromium, "
            "Firefox, and WebKit with a single API. It supports headless and "
            "headed modes, auto-waits for elements, and exposes semantic "
            "locators such as get_by_role and get_by_label for resilient tests."
        ),
    },
    {
        "instruction": "Explain pytest fixtures.",
        "response": (
            "Fixtures in pytest are functions decorated with @pytest.fixture "
            "that provide reusable setup and teardown for tests. They are "
            "requested by name as test parameters, can be scoped to function, "
            "class, module, or session, and run teardown via yield."
        ),
    },
    {
        "instruction": "What is get_by_role?",
        "response": (
            "get_by_role is a Playwright locator that selects elements by their "
            "ARIA role, optionally filtered by accessible name. It is the most "
            "user-facing and robust strategy because it survives CSS class and "
            "DOM structure changes."
        ),
    },
    {
        "instruction": "What is the difference between page.click and locator.click?",
        "response": (
            "page.click takes a selector string and performs a one-shot click, "
            "while locator.click operates on a Locator object that can be "
            "stored, chained, and asserted on. Locators are the recommended "
            "API because they support auto-waiting and retries."
        ),
    },
    {
        "instruction": "How do Playwright assertions work?",
        "response": (
            "Playwright's expect() function returns an assertion object with "
            "matchers like to_be_visible, to_have_text, and to_have_url. "
            "Each matcher polls the page until the condition holds or the "
            "timeout elapses, which removes the need for explicit waits."
        ),
    },
]

GENERAL_RATIO = 0.1  # 10% of final dataset is general examples


# ═════════════════════════════════════════════════════════════════════════════
# VRAM watchdog
# ═════════════════════════════════════════════════════════════════════════════


class VRAMWatchdog(threading.Thread):
    """
    Background thread that polls torch.cuda.mem_get_info() every few seconds.

    Behaviour:
      * Logs full usage every ``log_every_sec`` seconds.
      * Logs a warning when free VRAM drops below ``warn_gb``.
      * Sets an internal "should stop" flag and raises SIGINT on the main
        thread when free VRAM drops below ``crit_gb`` so the SFTTrainer can
        catch KeyboardInterrupt and save a checkpoint.

    The thread is a daemon and self-terminates when ``stop()`` is called.
    """

    def __init__(
        self,
        warn_gb: float = VRAM_WARN_GB,
        crit_gb: float = VRAM_CRIT_GB,
        poll_every_sec: float = VRAM_POLL_EVERY_SEC,
        log_every_sec: float = VRAM_LOG_EVERY_SEC,
    ) -> None:
        super().__init__(name="VRAMWatchdog", daemon=True)
        self.warn_gb = warn_gb
        self.crit_gb = crit_gb
        self.poll_every_sec = poll_every_sec
        self.log_every_sec = log_every_sec
        self._stop_event = threading.Event()
        self._critical = False

    @property
    def critical(self) -> bool:
        return self._critical

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            import torch
        except ImportError:
            logger.warning("VRAMWatchdog: torch not available — watchdog disabled")
            return

        if not torch.cuda.is_available():
            logger.warning("VRAMWatchdog: CUDA unavailable — watchdog disabled")
            return

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        total_gb    = total_bytes / 1e9
        last_log    = 0.0
        warned      = False

        logger.info(f"VRAMWatchdog started — total VRAM={total_gb:.1f} GB")

        while not self._stop_event.is_set():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except Exception as exc:
                logger.debug(f"VRAMWatchdog: mem_get_info failed: {exc}")
                self._stop_event.wait(self.poll_every_sec)
                continue

            free_gb = free_bytes / 1e9
            used_gb = total_gb - free_gb
            pct     = (used_gb / total_gb) * 100 if total_gb else 0.0

            now = time.monotonic()
            if now - last_log >= self.log_every_sec:
                logger.info(
                    f"VRAM: {used_gb:.1f}/{total_gb:.1f}GB ({pct:.0f}%) — free {free_gb:.2f}GB"
                )
                last_log = now

            if free_gb < self.crit_gb and not self._critical:
                self._critical = True
                logger.critical(
                    f"VRAM critical ({free_gb*1024:.0f} MB free < "
                    f"{self.crit_gb*1024:.0f} MB) — requesting graceful stop"
                )
                try:
                    os.kill(os.getpid(), signal.SIGINT)
                except Exception as exc:
                    logger.error(f"VRAMWatchdog: failed to raise SIGINT: {exc}")
                # Stay alive so we keep logging until the trainer exits.
            elif free_gb < self.warn_gb and not warned:
                logger.warning(
                    f"VRAM low ({free_gb*1024:.0f} MB free < "
                    f"{self.warn_gb*1024:.0f} MB) — close other GPU apps"
                )
                warned = True

            self._stop_event.wait(self.poll_every_sec)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset preparation
# ═════════════════════════════════════════════════════════════════════════════


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


def _general_as_chatml(example: dict[str, str]) -> dict[str, Any]:
    """Wrap a {instruction, response} pair as a ChatML record."""
    return {
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant who answers software-engineering "
                        "questions concisely."},
            {"role": "user",    "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ],
        "metadata": {"domain": "general", "type": "general"},
    }


def _interleave_general(
    qa_records: list[dict[str, Any]],
    ratio: float = GENERAL_RATIO,
) -> list[dict[str, Any]]:
    """
    Insert general instruction-following examples between QA examples so the
    fine-tuned model retains broad capability.

    Inserts 1 general example after every N=round(1/ratio) QA examples.
    """
    if not qa_records:
        return qa_records
    if ratio <= 0:
        return qa_records

    every_n = max(1, round(1.0 / ratio))
    out: list[dict[str, Any]] = []
    gi = 0
    for i, rec in enumerate(qa_records, 1):
        out.append(rec)
        if i % every_n == 0:
            out.append(_general_as_chatml(GENERAL_EXAMPLES[gi % len(GENERAL_EXAMPLES)]))
            gi += 1
    logger.info(
        f"Interleaved {gi} general examples into {len(qa_records)} QA examples "
        f"(1 per {every_n}) — final size {len(out)}"
    )
    return out


def prepare_dataset(jsonl_path: Path, tokenizer: Any, max_seq_length: int) -> Any:
    """
    Load JSONL, apply chat template, filter over-long examples, interleave
    general examples, return a HuggingFace Dataset with a single "text" field.
    """
    from datasets import Dataset  # type: ignore[import-not-found]

    raw = _load_jsonl(jsonl_path)
    if not raw:
        raise ValueError(f"Dataset empty: {jsonl_path}")
    logger.info(f"Loaded {len(raw)} raw examples from {jsonl_path}")

    # Support both ChatML {"messages": [...]} and prompt/completion {"prompt": ..., "completion": ...}
    def _to_messages(r: dict[str, Any]) -> list[dict[str, str]] | None:
        if r.get("messages"):
            return r["messages"]
        if r.get("prompt") and r.get("completion"):
            return [
                {"role": "user",      "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]},
            ]
        return None

    qa: list[dict[str, Any]] = []
    for r in raw:
        msgs = _to_messages(r)
        if msgs is not None:
            qa.append({**r, "messages": msgs})
    mixed = _interleave_general(qa)

    formatted: list[dict[str, str]] = []
    filtered_out = 0
    for rec in mixed:
        messages = rec.get("messages") or []
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            logger.debug(f"apply_chat_template failed: {exc}")
            filtered_out += 1
            continue

        try:
            n_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            n_tokens = len(text) // 3
        if n_tokens > max_seq_length:
            filtered_out += 1
            continue

        formatted.append({"text": text})

    logger.info(
        f"Dataset prepared | total_in={len(mixed)} filtered_out={filtered_out} "
        f"final={len(formatted)}"
    )
    if not formatted:
        raise ValueError("All examples were filtered out — check max_seq_length")
    return Dataset.from_list(formatted)


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════


def load_model_for_training() -> tuple[Any, Any]:
    """Load the 4-bit base model and apply minimal LoRA adapters."""
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    logger.info(f"Loading base model {BASE_MODEL} (max_seq_length={MAX_SEQ_LENGTH})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,            # auto-pick fp16
        load_in_4bit=True,
        device_map={"": 0},    # force all layers onto GPU 0; avoids conservative
                               # device_map="auto" splitting layers to CPU/disk
    )

    logger.info(
        f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, modules={TARGET_MODULES})"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Unsloth patches SFTTrainer to inject args.eos_token = "<EOS_TOKEN>" regardless
    # of what we pass. Qwen2's tokenizer doesn't have <EOS_TOKEN>, so TRL's
    # convert_tokens_to_ids returns None → ValueError.
    # Fix: register <EOS_TOKEN> in the tokenizer's added_tokens_encoder pointing to
    # the same ID as <|im_end|>, so TRL's validation passes and the correct EOS is used.
    _eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")  # 151645 for Qwen2
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = _eos_id
    # Make <EOS_TOKEN> resolve to the correct ID so TRL's check doesn't raise
    tokenizer.added_tokens_encoder["<EOS_TOKEN>"] = _eos_id
    logger.info(f"Registered <EOS_TOKEN>→{_eos_id} (<|im_end|>) in tokenizer vocab")

    return model, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════


def train(
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Path:
    """
    Full WSL2-safe training run.

    Returns the path of the final checkpoint directory.
    """
    import torch  # noqa: F401   (caller already imports indirectly)
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling  # type: ignore[import-not-found]
    from transformers.trainer_callback import TrainerCallback  # type: ignore[import-not-found]

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    watchdog = VRAMWatchdog()
    watchdog.start()

    model, tokenizer = load_model_for_training()

    # Tokenize the pre-formatted "text" dataset for the standard Trainer.
    # Using Trainer instead of SFTTrainer avoids Unsloth's <EOS_TOKEN> injection issue.
    raw_dataset = prepare_dataset(dataset_path.expanduser(), tokenizer, MAX_SEQ_LENGTH)

    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = raw_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], num_proc=2,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class _StepLogger(TrainerCallback):
        def on_log(self_inner, args, state, control, logs=None, **kw):  # noqa: N805
            if not logs:
                return
            loss = logs.get("loss")
            if loss is None:
                return
            try:
                import torch
                free_b, _ = torch.cuda.mem_get_info()
                total_b   = torch.cuda.get_device_properties(0).total_memory
                vram_used = (total_b - free_b) / 1e9
                vram_msg = f"{vram_used:.1f}GB"
            except Exception:
                vram_msg = "?"
            logger.info(
                f"Step {state.global_step}/{max_steps} | "
                f"Loss: {loss:.4f} | VRAM: {vram_msg}"
            )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[_StepLogger()],
    )

    final_dir = output_dir / "final"
    logger.info(f"Starting training (max_steps={max_steps}) — checkpoints to {output_dir}")
    try:
        stats = trainer.train()
        logger.info(
            f"Training complete | loss={stats.training_loss:.4f} "
            f"steps={stats.global_step}"
        )
    except KeyboardInterrupt:
        emergency_dir = output_dir / "emergency"
        logger.warning(
            f"KeyboardInterrupt received — saving emergency checkpoint to {emergency_dir}"
        )
        emergency_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(emergency_dir))
            tokenizer.save_pretrained(str(emergency_dir))
        except Exception as exc:
            logger.error(f"Emergency save failed: {exc}")
        watchdog.stop()
        raise

    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    watchdog.stop()
    logger.info(f"Final checkpoint saved: {final_dir}")
    return final_dir


# ═════════════════════════════════════════════════════════════════════════════
# Export to GGUF
# ═════════════════════════════════════════════════════════════════════════════


def export_gguf(checkpoint_dir: Path, output_dir: Path) -> Path:
    """
    Load a LoRA checkpoint with FastLanguageModel (which automatically merges
    the adapter), save it as GGUF Q4_K_M, and write the Modelfile alongside it.

    Returns the path to the produced GGUF file.
    """
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    checkpoint_dir = checkpoint_dir.expanduser()
    output_dir     = output_dir.expanduser()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint for export: {checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    logger.info(f"Saving GGUF ({GGUF_QUANT_METHOD}) to {output_dir}")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=GGUF_QUANT_METHOD,
    )

    # Locate the produced .gguf file (Unsloth name varies by version) and
    # normalise to qa-agent-coder-q4_k_m.gguf so the Ollama Modelfile is stable.
    candidates = sorted(output_dir.glob("*.gguf"))
    if not candidates:
        raise RuntimeError(f"No .gguf file produced in {output_dir}")
    final_gguf = output_dir / GGUF_FILE_NAME
    if candidates[0].name != GGUF_FILE_NAME:
        try:
            candidates[0].replace(final_gguf)
            logger.info(f"Renamed {candidates[0].name} -> {final_gguf.name}")
        except OSError:
            import shutil
            shutil.copy2(candidates[0], final_gguf)
            logger.info(f"Copied {candidates[0].name} -> {final_gguf.name}")

    modelfile = output_dir / "Modelfile"
    modelfile.write_text(
        MODELFILE_BODY.format(gguf_name=GGUF_FILE_NAME),
        encoding="utf-8",
    )
    logger.info(f"Modelfile written: {modelfile}")
    logger.info(f"GGUF export complete: {final_gguf}")
    return final_gguf


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsl2_trainer",
        description="WSL2 QLoRA trainer for qa-agent (6 GB VRAM safe)",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to ChatML JSONL training file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output dir (checkpoint dir during training, "
                             "GGUF dir during export)")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Total training steps (default 300)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, only export an existing checkpoint to GGUF")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[--export-only] Path to the trained checkpoint")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    out  = Path(args.output).expanduser()

    if args.export_only:
        if not args.checkpoint:
            logger.error("--export-only requires --checkpoint")
            return 2
        export_gguf(Path(args.checkpoint), out)
        return 0

    if not args.dataset:
        logger.error("--dataset is required for training (omit --export-only for that)")
        return 2

    try:
        train(Path(args.dataset), out, max_steps=args.max_steps)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130
    except Exception as exc:
        logger.exception(f"Training failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
