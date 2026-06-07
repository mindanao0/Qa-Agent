## CONTEXT
PROJECT: D:\Code\qa-agent
GPU: GTX 1660 Ti 6GB VRAM (WDDM TDR fixed: TdrDelay=60)
BASE MODEL: qwen2.5-coder:7b-instruct-q4_K_M (via Ollama)
DATASET: data/training/train.jsonl (449 examples), val.jsonl (51 examples)
TARGET: fine-tuned adapter → load via Ollama

## OBJECTIVE
GOAL: QLoRA 4-bit fine-tune บน 6GB VRAM
      produce GGUF adapter → load ใน Ollama
      วัด quality gain vs base model

## ACCEPTANCE CRITERIA
  training_loss_final   ≤ 1.50   (converged)
  val_loss_final        ≤ 2.00
  quality_gain          ≥ 0.05   (fine-tuned pass_rate - base pass_rate)
  vram_peak_mb          ≤ 5800   (headroom บน 6GB)
  oom_errors            = 0

## STEP 1 — Install Unsloth + dependencies

# Create scripts/install_finetune_deps.py:
import subprocess, sys

DEPS = [
    # Unsloth for QLoRA (Windows-compatible)
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    # Core training
    "transformers>=4.40.0",
    "trl>=0.8.0",           # SFTTrainer
    "peft>=0.10.0",         # LoRA
    "bitsandbytes>=0.43.0", # 4-bit quantization (Windows build)
    "accelerate>=0.29.0",
    "datasets>=2.18.0",
    # GGUF export
    "llama-cpp-python",
]

# Run: uv run python scripts/install_finetune_deps.py
# Verify each import after install

## STEP 2 — Create fine-tune script

### scripts/finetune.py
# QLoRA fine-tune with Unsloth
#
# CONFIG (6GB VRAM optimized):
FINETUNE_CONFIG = {
    "model_name":        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    "max_seq_length":    1024,    # keep short for VRAM
    "load_in_4bit":      True,
    "lora_r":            8,       # rank — lower = less VRAM
    "lora_alpha":        16,
    "lora_dropout":      0.0,     # Unsloth recommendation
    "target_modules":    ["q_proj","k_proj","v_proj","o_proj",
                          "gate_proj","up_proj","down_proj"],
    "per_device_train_batch_size": 1,    # VRAM constraint
    "gradient_accumulation_steps": 4,   # effective batch = 4
    "warmup_steps":      10,
    "max_steps":         100,     # start conservative
    "learning_rate":     2e-4,
    "fp16":              True,    # GTX 1660 Ti = no bf16
    "optim":             "adamw_8bit",
    "seed":              42,
    "output_dir":        "models/finetune_output",
    "save_steps":        50,
    "logging_steps":     10,
}
#
# Training format (ChatML):
#   {"role": "user",      "content": example["prompt"]}
#   {"role": "assistant", "content": example["completion"]}
#
# VRAMMonitor.monitor_during(training_loop) → track peak
# MemoryGuard(max_growth_mb=5800) → abort if exceeded
# asyncio.Semaphore(1) NOT needed here (no Ollama during training)
#
# Checkpointing: save every 50 steps to models/finetune_output/
# Resume: if checkpoint exists → resume_from_checkpoint=True

## STEP 3 — Export to GGUF

### scripts/export_gguf.py
# Convert fine-tuned adapter → GGUF for Ollama
#
# from unsloth import FastLanguageModel
# model.save_pretrained_gguf(
#     "models/qwen2.5-coder-finetuned",
#     tokenizer,
#     quantization_method="q4_k_m",  # match base model quantization
# )
# Output: models/qwen2.5-coder-finetuned.Q4_K_M.gguf

## STEP 4 — Load in Ollama

### scripts/create_modelfile.py
# Create Ollama Modelfile for fine-tuned model:
#
# MODELFILE content:
# FROM ./models/qwen2.5-coder-finetuned.Q4_K_M.gguf
# PARAMETER temperature 0.1
# PARAMETER num_ctx 2048
# SYSTEM "You are an expert QA automation engineer..."
#
# Run:
# ollama create qa-agent-finetuned -f Modelfile
# ollama run qa-agent-finetuned "Generate a pytest test for..."

## STEP 5 — Quality evaluation

### scripts/eval_finetune.py
# Compare base vs fine-tuned on val.jsonl (51 examples)
#
# For each example in val.jsonl:
#   1. Send prompt to BASE model (qwen2.5-coder:7b-instruct-q4_K_M)
#   2. Send same prompt to FINETUNED model (qa-agent-finetuned)
#   3. Score both completions:
#      - pytest test: run via TestExecutor → pass/fail
#      - schema: validate with jsonschema → valid/invalid
#      - invariant: check hypothesis_strategy is valid Python
#   4. base_pass_rate    = base_passed / 51
#   5. ft_pass_rate      = ft_passed / 51
#   6. quality_gain      = ft_pass_rate - base_pass_rate
#
# asyncio.Semaphore(1) on ALL model calls (base + finetuned)
# Output: models/eval_results.json

## OUTPUT models/eval_results.json:
{
  "base_pass_rate":     <float>,
  "ft_pass_rate":       <float>,
  "quality_gain":       <float>,
  "training_loss_final": <float>,
  "val_loss_final":     <float>,
  "vram_peak_mb":       <float>,
  "oom_errors":         <int>,
  "finetune_status":    "PASS" | "FAIL"
}

## RULES:
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - VRAMMonitor throughout training
# - asyncio.Semaphore(1) on ALL Ollama/model calls
# - fp16=True (GTX 1660 Ti ไม่รองรับ bf16)
# - batch_size=1 hard limit (VRAM constraint)
# - ถ้า OOM → reduce lora_r=4, max_seq_length=512 แล้วลองใหม่
# - NEVER train with load_in_4bit=False (จะ OOM ทันที)
# - Resume from checkpoint ถ้า training interrupted
# - BLOCKED_ACTION_PATTERNS ต้องไม่อยู่ใน training data
#   (validate_dataset.py ผ่านแล้ว — confirmed)

## INSTALL ORDER (สำคัญ — Windows):
# 1. uv run python scripts/install_finetune_deps.py
# 2. ถ้า bitsandbytes fail → pip install bitsandbytes
#    --index-url https://jllllll.github.io/bitsandbytes-windows-webui
# 3. verify: uv run python -c "import unsloth; print('OK')"
# 4. uv run python scripts/finetune.py
# 5. uv run python scripts/export_gguf.py
# 6. uv run python scripts/create_modelfile.py
# 7. uv run python scripts/eval_finetune.py