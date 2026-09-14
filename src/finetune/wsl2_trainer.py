#!/usr/bin/env python3
"""
QLoRA trainer for qa-agent — 6 GB VRAM safe.

NAME HISTORY: the "wsl2" in the filename is historical (born on Windows 11 + WSL2).
The project now runs on native Ubuntu; this script runs in the dedicated venv at
~/.qa-finetune-env (NOT the uv project venv). Only torch / unsloth / transformers /
trl / datasets / peft / bitsandbytes are required.

Key design choices:

  * 7B preset is the project mandate (never train a smaller model for real runs):
    seq=384, LoRA r=4 on q/v only, grad_accum=16 — see MODEL_PRESETS["7b"].
  * --cpu-offload: bitsandbytes + accelerate GPU↔CPU split — the only way 7B
    trains on the GTX 1660 Ti (5.61 GiB usable). ~4 min/step; a 300-step run is
    ~18-19 h. See load_model_for_training_cpu_offload for the three crash gotchas.
  * --resume auto|PATH: continue from the last checkpoint. Mandatory workflow for
    multi-hour runs on a desktop that gets power-cycled (the 2026-06-27 run died
    at step 6/300 with nothing saved — SAVE_STEPS now defaults to 20).
  * Background VRAMWatchdog polls torch.cuda.mem_get_info(); TWO consecutive
    critical polls (JIT embed/lm_head copies cause one-poll transients) raise
    SIGINT for a graceful stop, and the stop_reason is logged honestly.
  * Catastrophic-forgetting mitigation: ~10% general examples interleaved.
  * KeyboardInterrupt handler saves an emergency adapter before exit.
  * --export-only: LEGACY unsloth GGUF export — loads all-on-GPU and OOMs by
    ~34 MB with the GUI up. Prefer scripts/export_adapter_gguf.sh (CPU-only).

CLI:

    python3 wsl2_trainer.py --model 7b --cpu-offload --dataset PATH --output PATH \
        [--max-steps N] [--resume auto|CKPT_DIR]
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

# ── Model presets ─────────────────────────────────────────────────────────────
# Each preset tuned to fit inside 6 GB VRAM.
MODEL_PRESETS: dict[str, dict] = {
    "1.5b": {
        "base_model":    "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        # hf_base: the un-quantized HF repo used ONLY by the --cpu-offload path
        # (transformers quantizes it on the fly and can split layers GPU↔CPU; the
        # pre-quantized unsloth bnb-4bit repos above cannot be CPU-offloaded cleanly).
        "hf_base":       "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "max_seq_length": 1024,
        "lora_r":         8,
        "lora_alpha":     16,
        "grad_accum":     8,
    },
    "3b": {
        "base_model":    "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
        "hf_base":       "Qwen/Qwen2.5-Coder-3B-Instruct",
        "max_seq_length": 1024,
        "lora_r":         8,
        "lora_alpha":     16,
        "grad_accum":     8,
    },
    "7b": {
        "base_model":    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "hf_base":       "Qwen/Qwen2.5-Coder-7B-Instruct",
        "max_seq_length": 384,   # 256→384 (2026-07-12): validated by the 20-step headless
                                 # cpu-offload run at budget 2.9 — min free VRAM 0.83GB
                                 # across ~2h, zero watchdog strikes, keeps 93.3% of the
                                 # dataset vs 21.8% at 256 (p50=299 tok — see
                                 # scripts/measure_seq_fit.py). The old 256 came from the
                                 # 2026-06-27 GUI-up OOM at seq=512; headless training
                                 # (GUI down) is what buys the activation headroom.
                                 # FINETUNE_SEQ_LEN env still overrides for experiments,
                                 # but NEVER resume a checkpoint across a seq change —
                                 # prepare_dataset re-filters at the new length, so the
                                 # resumed run silently trains on a different dataset.
        "lora_r":         4,     # fewer trainable params → less optimizer VRAM
        "lora_alpha":     8,
        "grad_accum":     16,    # compensate for smaller seq_len
    },
}
DEFAULT_MODEL_SIZE  = "7b"   # project mandate: 7B-only for real runs (1.5b/3b presets
                             # kept for historical smoke tests — pass --model explicitly)

# ── Runtime constants (filled in from preset at startup) ──────────────────────
BASE_MODEL          = MODEL_PRESETS[DEFAULT_MODEL_SIZE]["base_model"]
HF_BASE             = MODEL_PRESETS[DEFAULT_MODEL_SIZE]["hf_base"]
MAX_SEQ_LENGTH      = MODEL_PRESETS[DEFAULT_MODEL_SIZE]["max_seq_length"]
LORA_R              = MODEL_PRESETS[DEFAULT_MODEL_SIZE]["lora_r"]
LORA_ALPHA          = MODEL_PRESETS[DEFAULT_MODEL_SIZE]["lora_alpha"]
LORA_DROPOUT        = 0.0    # Unsloth recommendation for 4-bit
TARGET_MODULES      = ["q_proj", "v_proj"]
BATCH_SIZE          = 1
GRAD_ACCUM_STEPS    = MODEL_PRESETS[DEFAULT_MODEL_SIZE]["grad_accum"]
DEFAULT_MAX_STEPS   = 300

# ── CPU-offload path defaults ─────────────────────────────────────────────────
# GPU VRAM ceiling for the decoder layers that stay on-GPU (embed_tokens + lm_head
# are always CPU-offloaded). Fewer GPU layers → more headroom but slower steps.
# MEASURED on the GTX 1660 Ti (5.61 GiB usable), 7B, seq=256, GUI up (2026-06-27):
#   budget 4.5 (28/28 layers GPU) → trains but OOM-spikes by ~140 MB
#   budget 3.2 (24/28 layers GPU) → trains 3 steps then a transient spike (24 MB
#                                   free) tripped the watchdog; ~236 s/step
# 2.9 (≈22/28 on GPU) trades ~2 more layers to CPU for the headroom those spikes
# need. NOTE: each CPU layer + embed + lm_head is copied GPU↔CPU every step, so this
# path is SLOW (~4 min/step here → a 300-step run ≈ ~20 h). It makes 7B *possible*
# on 6 GB, not fast. Raise the budget for speed (risk OOM), lower it for stability.
DEFAULT_GPU_BUDGET_GIB = 2.9
# Budgets >3.2 are measured OOM-spike territory on this card (see table above);
# main() clamps them back to the default unless FINETUNE_ALLOW_HIGH_BUDGET=1.
GPU_BUDGET_SAFE_MAX    = 3.2
LEARNING_RATE       = 2e-4
WARMUP_STEPS        = 10
# LOG_STEPS: env wins; otherwise resolved in train() — 1 in cpu-offload mode
# (~4 min/step, a line per step proves liveness in journalctl), 10 elsewhere.
_LOG_STEPS_ENV      = os.environ.get("LOG_STEPS")
# 20 steps ≈ 75 min between checkpoints in offload mode. Adapter checkpoints are
# a few MB — cheap insurance on a desktop that gets power-cycled.
SAVE_STEPS          = int(os.environ.get("SAVE_STEPS", "20"))

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
        self._crit_strikes = 0

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
                # The offload path JIT-copies embed/lm_head to the GPU each step —
                # a single poll can catch that transient. Require 2 consecutive
                # critical polls before stopping (real OOM fires from PyTorch first).
                self._crit_strikes += 1
                if self._crit_strikes >= 2:
                    self._critical = True
                    logger.critical(
                        f"VRAM critical ({free_gb*1024:.0f} MB free < "
                        f"{self.crit_gb*1024:.0f} MB, {self._crit_strikes} consecutive polls) "
                        f"— requesting graceful stop (stop_reason=vram_critical)"
                    )
                    try:
                        os.kill(os.getpid(), signal.SIGINT)
                    except Exception as exc:
                        logger.error(f"VRAMWatchdog: failed to raise SIGINT: {exc}")
                    # Stay alive so we keep logging until the trainer exits.
                else:
                    logger.warning(
                        f"VRAM below critical once ({free_gb*1024:.0f} MB free) — "
                        f"waiting one more poll to rule out a JIT-copy transient"
                    )
            elif free_gb >= self.crit_gb and self._crit_strikes and not self._critical:
                self._crit_strikes = 0
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

    kept_pct = 100.0 * len(formatted) / max(1, len(mixed))
    logger.info(
        f"Dataset prepared | total_in={len(mixed)} filtered_out={filtered_out} "
        f"final={len(formatted)} (kept {kept_pct:.1f}% at max_seq_length={max_seq_length})"
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


def load_model_for_training_cpu_offload(
    gpu_budget_gib: float = DEFAULT_GPU_BUDGET_GIB,
) -> tuple[Any, Any]:
    """
    CPU-offload load path — for models whose 4-bit weights won't leave enough
    VRAM headroom on this GPU to train all-on-device (e.g. 7B on a 6 GB card).

    Unsloth is NOT used here: it requires the whole model on one GPU and has no
    partial-CPU-offload mode. Instead we quantize the *un-quantized* HF base on
    the fly with bitsandbytes and let accelerate's device_map="auto" place as many
    layers on the GPU as fit under ``gpu_budget_gib`` and spill the rest to CPU RAM
    (kept fp32 via llm_int8_enable_fp32_cpu_offload). The offloaded layers run their
    forward/backward on CPU each step — correct, but several× slower than all-GPU.

    NOTE: this downloads the full fp16 HF base (~15 GB for 7B) on first use; the
    pre-quantized unsloth bnb-4bit repo cannot be CPU-offloaded cleanly.
    """
    import torch
    from transformers import (  # type: ignore[import-not-found]
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import (  # type: ignore[import-not-found]
        LoraConfig,
        get_peft_model,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # MUST stay False with CPU offload: double-quant adds a nested `offset`
        # tensor to each layer's quant_state. When accelerate dispatches an
        # offloaded layer it walks state_dict() → bitsandbytes calls offset.item(),
        # but the offloaded offset is a *meta* tensor → "Tensor.item() cannot be
        # called on meta tensors" (verified crash 2026-06-27). It only saves ~0.4GB.
        bnb_4bit_use_double_quant=False,
        # allow the 4-bit model to keep overflow layers on CPU instead of
        # refusing to dispatch — this is what makes the GPU↔CPU split legal.
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Explicit device_map instead of "auto" + max_memory. The two un-quantized fp16
    # weights — embed_tokens and lm_head (~1.1GB each) — are the VRAM hogs. accelerate
    # offload does NOT run an offloaded module on CPU; it copies the weight to the GPU
    # just-in-time for that module's forward, then evicts it. So an offloaded module
    # still needs ~its size free on the GPU while it runs — but embed (start of fwd)
    # and lm_head (end of fwd) execute at different times, so only ONE of them is on
    # the GPU at a peak (~1.1GB), and loss is still computed on CUDA (lm_head runs on
    # GPU after copy-in → fp16 GradScaler is happy). Offloading BOTH frees ~2.2GB of
    # permanent residency, leaving room for the JIT copy + activations. The 4-bit
    # decoder layers (bulk of compute, ~0.13GB each) stay GPU-pinned; gpu_budget_gib
    # caps how many fit (rest spill to CPU, slower). At 4.5 all 28 layers stay on GPU.
    cfg = AutoConfig.from_pretrained(HF_BASE)
    n_layers = int(getattr(cfg, "num_hidden_layers", 28))
    _GIB_PER_LAYER = 0.13   # ~4-bit footprint of one Qwen-7B decoder block
    max_gpu_layers = max(0, int(gpu_budget_gib / _GIB_PER_LAYER))
    layers_on_gpu = min(n_layers, max_gpu_layers)

    device_map: dict[str, Any] = {
        "model.embed_tokens": "cpu",
        "lm_head": "cpu",
        "model.norm": 0,
        "model.rotary_emb": 0,
    }
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = 0 if i < layers_on_gpu else "cpu"

    logger.info(
        f"[cpu-offload] loading {HF_BASE} (4-bit, explicit map: embed+lm_head→CPU "
        f"(JIT-copied to GPU per step), {layers_on_gpu}/{n_layers} decoder layers→GPU, "
        f"rest→CPU; gpu_budget={gpu_budget_gib}GiB) — first run downloads fp16 base (~15GB)"
    )
    model = AutoModelForCausalLM.from_pretrained(
        HF_BASE,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )

    devmap = getattr(model, "hf_device_map", {}) or {}
    on_cpu = sum(1 for d in devmap.values() if d in ("cpu", "disk"))
    on_gpu = sum(1 for d in devmap.values() if d not in ("cpu", "disk"))
    logger.info(
        f"[cpu-offload] device split — {on_gpu} module(s) on GPU, {on_cpu} on CPU/disk "
        f"({layers_on_gpu}/{n_layers} decoder layers on GPU)"
    )

    # Minimal k-bit prep WITHOUT peft.prepare_model_for_kbit_training's blanket
    # fp32 upcast — that upcast is what turned embed_tokens/lm_head (~1.1GB fp16)
    # into ~2GB fp32 and OOM'd the GPU. We only need: (1) freeze the base, (2) upcast
    # the tiny LayerNorms to fp32 for training stability, (3) gradient checkpointing,
    # (4) input grads so checkpointing has something to backprop into.
    model.config.use_cache = False
    for _p in model.parameters():
        _p.requires_grad = False
    _norms_upcast = 0
    for _name, _mod in model.named_modules():
        if "norm" in _name.lower():
            _w = getattr(_mod, "weight", None)
            if _w is not None and _w.dtype in (torch.float16, torch.bfloat16):
                _mod.weight.data = _mod.weight.data.to(torch.float32)
                _norms_upcast += 1
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    logger.info(
        f"[cpu-offload] minimal k-bit prep done (upcast {_norms_upcast} LayerNorms to "
        "fp32; embed/lm_head left fp16 to avoid the 2GB GPU OOM)"
    )

    # Restrict LoRA to the GPU-resident decoder layers. accelerate keeps an
    # offloaded layer's weights on the `meta` device between forwards; a trainable
    # LoRA adapter on such a layer makes autograd return a cuda:0 gradient for a
    # meta-device param → "MmBackward0 returned an invalid gradient ... expected
    # device meta but got cuda:0" (verified crash 2026-06-27). Offloaded layers run
    # frozen (base only, no adapter) so gradient just flows through them.
    layers_to_transform = (
        list(range(layers_on_gpu)) if layers_on_gpu < n_layers else None
    )
    logger.info(
        f"[cpu-offload] applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}, modules={TARGET_MODULES}, "
        f"layers={'all' if layers_to_transform is None else f'0..{layers_on_gpu-1} (GPU-resident only)'})"
    )
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(HF_BASE)
    _eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = _eos_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"[cpu-offload] EOS=<|im_end|>→{_eos_id}; pad_token={tokenizer.pad_token}")

    return model, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════


def restore_lora_adapter(model: Any, ckpt_dir: Path) -> int:
    """Load a checkpoint's LoRA weights into *model* WITHOUT re-dispatching it.

    Why this exists: transformers' Trainer._load_from_checkpoint() resumes a PeftModel by
    calling PeftModel.load_adapter(), which re-runs accelerate's dispatch_model() whenever
    hf_device_map contains "cpu"/"disk" — and it passes no offload dir. Our CPU-offload map
    keeps the offloaded base weights (embed_tokens, lm_head, the non-GPU decoder layers) on
    the meta device, so the re-dispatch has no value to place and dies with:
        ValueError: weight is on the meta device, we need a `value` to put in on 0.
    Both 2026-07-14 headless runs died there, 2 s in, on --resume auto.

    A resume only needs the LoRA weights out of the model checkpoint, and those live purely
    on the GPU-resident layers (layers_to_transform) — so load them straight into the adapter
    and never touch the dispatch. Optimizer / scheduler / scaler / RNG / step count are still
    restored by the stock Trainer code paths.

    Fail-closed: an adapter that does not fully land would train from scratch while *looking*
    like a resume, so a partial or no-op load raises instead of burning 27 h.

    Returns the number of tensors restored.
    """
    import torch
    from peft import set_peft_model_state_dict  # type: ignore[import-not-found]

    safetensors_file = ckpt_dir / "adapter_model.safetensors"
    bin_file = ckpt_dir / "adapter_model.bin"
    if safetensors_file.exists():
        from safetensors.torch import load_file  # type: ignore[import-not-found]
        adapter_sd = load_file(str(safetensors_file))
    elif bin_file.exists():
        adapter_sd = torch.load(str(bin_file), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"Resume from {ckpt_dir}: no adapter_model.safetensors / adapter_model.bin"
        )
    if not adapter_sd:
        raise ValueError(f"Resume from {ckpt_dir}: adapter checkpoint is empty")

    active = getattr(model, "active_adapters", None)
    adapter_name = active[0] if active else getattr(model, "active_adapter", "default")

    result = set_peft_model_state_dict(model, adapter_sd, adapter_name=adapter_name)
    # NOTE: missing_keys is meaningless here — it lists every frozen base weight, because
    # the adapter state dict only carries LoRA tensors. unexpected_keys is the real signal:
    # it means a checkpoint tensor matched no parameter in the model we just built.
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            f"Resume from {ckpt_dir}: {len(unexpected)} adapter tensor(s) matched no parameter "
            f"in the model (first: {unexpected[:3]}). The checkpoint was built with a different "
            f"LoRA config — r / lora_alpha / target_modules / layers_to_transform (which follows "
            f"--gpu-budget-gib). Refusing to train on a half-restored adapter."
        )

    # lora_B is zero-initialised when the adapter is built, so a restored adapter cannot be
    # all-zero. This is the guard against a silent no-op load (right keys, nothing copied).
    b_norm = 0.0
    for name, param in model.named_parameters():
        if "lora_B" in name and param.device.type != "meta":
            b_norm += float(param.detach().float().norm())
    if b_norm == 0.0:
        raise RuntimeError(
            f"Resume from {ckpt_dir}: every lora_B weight is still zero after the load — the "
            f"adapter did NOT restore, and this run would silently train from scratch."
        )

    logger.info(
        f"[resume] restored {len(adapter_sd)} LoRA tensors from {ckpt_dir} "
        f"(adapter='{adapter_name}', ||lora_B||={b_norm:.4f}) — no dispatch_model, "
        f"offloaded base weights left on meta"
    )
    return len(adapter_sd)


def resync_lr_after_resume(optimizer: Any, lr_scheduler: Any) -> list[float] | None:
    """Re-point the restored optimizer LRs at the CURRENT schedule horizon.

    LambdaLR.load_state_dict() restores last_epoch / _step_count / base_lrs / _last_lr but NOT
    the lambdas — a functools.partial's __dict__ is empty, so the freshly built partial (with
    num_training_steps=max_steps) survives the load. Verified 2026-07-14 against the real
    scheduler.pt: the horizon does come back as 300.

    The optimizer, however, carries its own per-group 'lr' inside optimizer.pt, and HF applies
    THAT to the first post-resume update — it only calls lr_scheduler.step() *after* the step
    (and logs the pre-update LR, which is why checkpoint-20's own log shows 4.89e-06 at step 20
    while its scheduler.pt holds _last_lr=0.0).

    Resuming checkpoint-20 — a max_steps=20 run whose cosine had decayed to exactly 0.0 — into a
    300-step run therefore burns step 21 on a no-op update at lr=0.0 and logs "learning_rate:
    0.0", which is indistinguishable from a dead scheduler. Recomputing the LR from the fresh
    lambdas at last_epoch fixes both: step 21 runs at 1.994e-04 — the LR an uninterrupted
    300-step run would use there.

    Same-horizon resumes recompute the value they just restored → no-op. Returns the new LRs.
    """
    sched = getattr(lr_scheduler, "scheduler", lr_scheduler)  # unwrap accelerate's wrapper
    lambdas = getattr(sched, "lr_lambdas", None)
    base_lrs = getattr(sched, "base_lrs", None)
    if not lambdas or not base_lrs:
        return None  # not a LambdaLR (ReduceLROnPlateau &c.) — nothing to re-sync
    fresh = [base * fn(sched.last_epoch) for fn, base in zip(lambdas, base_lrs)]
    stale = [group["lr"] for group in optimizer.param_groups]
    for group, lr in zip(optimizer.param_groups, fresh):
        group["lr"] = lr
    sched._last_lr = list(fresh)
    if any(abs(a - b) > 1e-12 for a, b in zip(stale, fresh)):
        logger.info(
            f"[resume] LR re-synced to the current schedule at step {sched.last_epoch}: "
            f"{stale[0]:.3e} (from optimizer.pt) → {fresh[0]:.3e}. The checkpoint's schedule "
            f"ended on a different horizon, so its LR was stale."
        )
    return fresh


def train(
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    cpu_offload: bool = False,
    gpu_budget_gib: float = DEFAULT_GPU_BUDGET_GIB,
    resume: str | None = None,
) -> Path:
    """
    Full WSL2-safe training run.

    When ``cpu_offload`` is True the model is loaded via the bitsandbytes +
    accelerate GPU↔CPU split path (load_model_for_training_cpu_offload) instead
    of unsloth — slower per step, but fits models whose 4-bit weights would
    otherwise leave no VRAM headroom for the backward pass.

    Returns the path of the final checkpoint directory.
    """
    import torch  # noqa: F401   (caller already imports indirectly)
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling  # type: ignore[import-not-found]
    from transformers.trainer_callback import TrainerCallback  # type: ignore[import-not-found]

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume resolution — fail fast BEFORE any heavy loading ───────────────
    from transformers.trainer_utils import get_last_checkpoint  # type: ignore[import-not-found]

    last_ckpt = get_last_checkpoint(str(output_dir))
    resume_from: str | None = None
    if resume:
        if resume == "auto":
            if last_ckpt is None:
                logger.warning(f"--resume auto: no checkpoint in {output_dir} — starting fresh")
            else:
                resume_from = last_ckpt
        else:
            ckpt = Path(resume).expanduser()
            if not (ckpt / "trainer_state.json").exists():
                raise FileNotFoundError(
                    f"--resume {resume}: not a trainer checkpoint (missing trainer_state.json)"
                )
            resume_from = str(ckpt)
        if resume_from:
            state = json.loads(
                (Path(resume_from) / "trainer_state.json").read_text(encoding="utf-8")
            )
            done_steps = int(state.get("global_step", 0))
            if done_steps >= max_steps:
                raise ValueError(
                    f"Checkpoint {resume_from} is already at step {done_steps} >= "
                    f"max_steps={max_steps} — raise --max-steps or start a fresh output dir"
                )
            logger.info(f"Resuming from {resume_from} (step {done_steps}/{max_steps})")
    elif last_ckpt is not None:
        raise ValueError(
            f"Found existing checkpoint {last_ckpt} but --resume was not given. "
            f"Pass --resume auto to continue it, or move the checkpoint-* dirs aside to "
            f"start fresh (otherwise save_total_limit rotation would silently delete them)."
        )

    watchdog = VRAMWatchdog()
    watchdog.start()

    if cpu_offload:
        model, tokenizer = load_model_for_training_cpu_offload(gpu_budget_gib)
    else:
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

    # num_proc=1: map() with num_proc>1 forks a CUDA-initialized, watchdog-threaded
    # parent — a known intermittent-hang class. The dataset is small (~1 s anyway).
    tokenized = raw_dataset.map(
        tokenize_fn, batched=True, remove_columns=["text"], num_proc=1,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class _StepLogger(TrainerCallback):
        """Per-log line with loss, VRAM, measured s/step and ETA (resume-aware)."""

        def __init__(self) -> None:
            self._t0: float | None = None
            self._step0 = 0

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            loss = logs.get("loss")
            if loss is None:
                return
            now = time.monotonic()
            if self._t0 is None:
                # Baseline at the first logged step so model-load time and the
                # resume offset don't pollute the s/step estimate.
                self._t0, self._step0 = now, state.global_step
                eta_msg = "ETA: measuring"
            else:
                done = state.global_step - self._step0
                if done > 0:
                    per_step = (now - self._t0) / done
                    remain_s = int(per_step * max(0, max_steps - state.global_step))
                    eta_msg = (
                        f"{per_step:.0f}s/step | "
                        f"ETA {remain_s // 3600}h{(remain_s % 3600) // 60:02d}m"
                    )
                else:
                    eta_msg = "ETA: measuring"
            try:
                import torch
                free_b, _ = torch.cuda.mem_get_info()
                total_b   = torch.cuda.get_device_properties(0).total_memory
                vram_msg  = f"{(total_b - free_b) / 1e9:.1f}GB"
            except Exception:
                vram_msg = "?"
            logger.info(
                f"Step {state.global_step}/{max_steps} | "
                f"Loss: {loss:.4f} | VRAM: {vram_msg} | {eta_msg}"
            )

    if _LOG_STEPS_ENV is not None:
        log_steps = int(_LOG_STEPS_ENV)
    else:
        # Offload steps take ~4 min each — a line per step keeps journalctl alive.
        log_steps = 1 if cpu_offload else 10

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS if cpu_offload else 2,
        # In the offload path prepare_model_for_kbit_training already enabled
        # gradient checkpointing (with use_reentrant=False); enabling it again here
        # would double-wrap. The unsloth path enables it via get_peft_model instead.
        gradient_checkpointing=False if cpu_offload else True,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=log_steps,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        # paged_adamw_8bit pages optimizer state to host RAM under VRAM pressure —
        # the right choice for the already-tight offload path; adamw_8bit elsewhere.
        optim="paged_adamw_8bit" if cpu_offload else "adamw_8bit",
        seed=42,
        report_to="none",
        # tqdm's \r-updates are noise in journald/tee'd log files; the per-step
        # logger line replaces it in offload mode.
        disable_tqdm=cpu_offload,
    )

    class _OffloadSafeTrainer(Trainer):
        """Trainer whose PEFT resume never re-dispatches a CPU-offloaded model.

        Only the offloaded path needs this (see restore_lora_adapter); a fully GPU-resident
        model — the unsloth path, hf_device_map={"": 0} — keeps the stock loader.
        """

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            target = model if model is not None else self.model
            device_map = getattr(target, "hf_device_map", None) or {}
            if not any(d in ("cpu", "disk") for d in device_map.values()):
                return super()._load_from_checkpoint(resume_from_checkpoint, model)
            restore_lora_adapter(target, Path(resume_from_checkpoint))

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if checkpoint is not None:
                resync_lr_after_resume(self.optimizer, self.lr_scheduler)

    trainer = _OffloadSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[_StepLogger()],
    )

    final_dir = output_dir / "final"
    logger.info(
        f"Starting training (max_steps={max_steps}, "
        f"resume={resume_from if resume_from else 'fresh'}) — "
        f"checkpoints to {output_dir} every {SAVE_STEPS} steps"
    )
    try:
        stats = trainer.train(resume_from_checkpoint=resume_from)
        logger.info(
            f"Training complete | loss={stats.training_loss:.4f} "
            f"steps={stats.global_step}"
        )
    except KeyboardInterrupt:
        # Honest stop reason: the watchdog raises SIGINT too — don't blame the user.
        reason = (
            "vram_critical (watchdog)" if watchdog.critical
            else "user_interrupt (SIGINT/Ctrl+C/systemctl stop)"
        )
        emergency_dir = output_dir / "emergency"
        logger.warning(
            f"Training stopped early — stop_reason={reason}; saving emergency adapter "
            f"to {emergency_dir} (to CONTINUE training, use the checkpoint-* dirs via "
            f"--resume auto — the emergency save is adapter-only, not resumable)"
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

    # Unsloth may save to output_dir or output_dir_gguf depending on version.
    # Check both and move files to output_dir so callers have a stable location.
    gguf_sibling = Path(str(output_dir) + "_gguf")
    if gguf_sibling.exists():
        import shutil as _shutil
        for f in gguf_sibling.iterdir():
            dest = output_dir / f.name
            if not dest.exists():
                _shutil.move(str(f), str(dest))
                logger.info(f"Moved {f.name} from _gguf dir → {output_dir}")
    candidates = sorted(output_dir.glob("*.gguf"))
    if not candidates:
        raise RuntimeError(f"No .gguf file produced in {output_dir} or {gguf_sibling}")
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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SIZE,
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model size preset (default 7b — the project mandate; "
                             "1.5b/3b exist only for historical smoke tests)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Train via bitsandbytes+accelerate GPU↔CPU split (no unsloth). "
                             "Fits 7B on a 6GB card by spilling layers to CPU RAM; "
                             "several× slower per step. Downloads the full fp16 base on first use.")
    parser.add_argument("--gpu-budget-gib", type=float, default=DEFAULT_GPU_BUDGET_GIB,
                        help=f"[--cpu-offload] VRAM ceiling for layer placement "
                             f"(default {DEFAULT_GPU_BUDGET_GIB}; raise for speed, lower if OOM)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training: 'auto' = latest checkpoint-* in --output, "
                             "or an explicit checkpoint dir. Without this flag a run "
                             "REFUSES to start if checkpoints already exist in --output "
                             "(protects them from save_total_limit rotation).")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip training, only export an existing checkpoint to GGUF "
                             "(LEGACY all-on-GPU path — prefer scripts/export_adapter_gguf.sh)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[--export-only] Path to the trained checkpoint")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    out  = Path(args.output).expanduser()

    # Apply model preset — mutates module-level globals so all functions pick it up
    preset = MODEL_PRESETS[args.model]
    global BASE_MODEL, HF_BASE, MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA, GRAD_ACCUM_STEPS
    BASE_MODEL       = preset["base_model"]
    HF_BASE          = preset["hf_base"]
    MAX_SEQ_LENGTH   = preset["max_seq_length"]
    LORA_R           = preset["lora_r"]
    LORA_ALPHA       = preset["lora_alpha"]
    GRAD_ACCUM_STEPS = preset["grad_accum"]
    # Seq-length experiment knob (measured 2026-07-12 on the deduped dataset:
    # seq=256 keeps only 21.8%, 320→67.2%, 384→92.6%). Longer seq costs
    # activation VRAM — validate with a short run before a long one.
    _seq_env = os.environ.get("FINETUNE_SEQ_LEN")
    if _seq_env:
        MAX_SEQ_LENGTH = int(_seq_env)
        logger.info(f"MAX_SEQ_LENGTH overridden via FINETUNE_SEQ_LEN → {MAX_SEQ_LENGTH}")
    logger.info(f"Model preset: {args.model} → {BASE_MODEL} | seq={MAX_SEQ_LENGTH} | r={LORA_R} | grad_accum={GRAD_ACCUM_STEPS}")
    if args.cpu_offload:
        logger.info(f"CPU-offload ENABLED → base={HF_BASE} | gpu_budget={args.gpu_budget_gib}GiB")

    if args.export_only:
        if not args.checkpoint:
            logger.error("--export-only requires --checkpoint")
            return 2
        export_gguf(Path(args.checkpoint), out)
        return 0

    if not args.dataset:
        logger.error("--dataset is required for training (omit --export-only for that)")
        return 2

    gpu_budget = args.gpu_budget_gib
    if (
        args.cpu_offload
        and gpu_budget > GPU_BUDGET_SAFE_MAX
        and os.environ.get("FINETUNE_ALLOW_HIGH_BUDGET") != "1"
    ):
        logger.warning(
            f"--gpu-budget-gib {gpu_budget} > {GPU_BUDGET_SAFE_MAX}: measured OOM-spike "
            f"territory on this card (2026-06-27: 4.5 → free dipped to 0.41GB; 3.2 → "
            f"watchdog trip) and barely faster (~222 vs ~236 s/step). Clamping to "
            f"{DEFAULT_GPU_BUDGET_GIB} — set FINETUNE_ALLOW_HIGH_BUDGET=1 to override."
        )
        gpu_budget = DEFAULT_GPU_BUDGET_GIB

    try:
        train(
            Path(args.dataset), out,
            max_steps=args.max_steps,
            cpu_offload=args.cpu_offload,
            gpu_budget_gib=gpu_budget,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted (stop_reason logged above by the trainer)")
        return 130
    except Exception as exc:
        logger.exception(f"Training failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
