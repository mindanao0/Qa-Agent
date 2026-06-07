## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Fix scripts/eval_finetune.py only — ห้าม modify อื่น
CONFIRMED:
  training = real (100 steps, loss 2.71→~1.5)
  model    = qa-agent-finetuned:latest (986MB, real GGUF)
  base     = qwen2.5-coder:7b-instruct-q4_K_M (responding normally)

## FIX 1 — trainer_state.json path

BEFORE: pathlib.Path("models/finetune_output/trainer_state.json")
AFTER:
  def _load_trainer_state() -> dict:
      output_dir = pathlib.Path("models/finetune_output")
      # find latest checkpoint
      checkpoints = sorted(
          [d for d in output_dir.iterdir()
           if d.is_dir() and d.name.startswith("checkpoint-")],
          key=lambda d: int(d.name.split("-")[1])
      )
      if not checkpoints:
          return {}
      latest = checkpoints[-1] / "trainer_state.json"
      return json.loads(latest.read_text()) if latest.exists() else {}

  # Extract loss from log_history:
  state = _load_trainer_state()
  log_history = state.get("log_history", [])
  train_losses = [e["loss"] for e in log_history if "loss" in e]
  eval_losses  = [e["eval_loss"] for e in log_history if "eval_loss" in e]
  training_loss_final = train_losses[-1] if train_losses else None
  val_loss_final      = eval_losses[-1]  if eval_losses  else None

## FIX 2 — Ollama call error handling

CURRENT problem: failed Ollama call → counts as fail silently
→ base_pass_rate non-deterministic

FIX in _call_model():
  async def _call_model(
      model: str,
      prompt: str,
      semaphore: asyncio.Semaphore,
      timeout_s: float = 60.0,
  ) -> str | None:
      async with semaphore:
          try:
              async with httpx.AsyncClient(timeout=timeout_s) as client:
                  resp = await client.post(
                      "http://localhost:11434/api/generate",
                      json={
                          "model":  model,
                          "prompt": prompt,
                          "stream": False,
                          "options": {"temperature": 0.1},
                      }
                  )
                  resp.raise_for_status()
                  return resp.json()["response"]
          except Exception as e:
              # log error but do NOT count as pass=False silently
              print(f"[WARN] model={model} call failed: {e}")
              return None  # None = skip this example, don't count

  # In scoring loop:
  response = await _call_model(model, prompt, semaphore)
  if response is None:
      skipped += 1
      continue   # ← skip, don't penalize pass_rate
  # else: score normally

  # Report skipped count in results:
  # "base_skipped": <int>
  # "ft_skipped":   <int>

## FIX 3 — pass_rate denominator

BEFORE: pass_rate = passed / total
AFTER:  pass_rate = passed / max(1, total - skipped)
# Only count examples that got a real response

## FIX 4 — vram_peak via torch (not NVML)

BEFORE: vram_peak_mb = None  (NVML not available during eval)
AFTER:
  def _vram_peak_mb() -> float | None:
      try:
          import torch
          if torch.cuda.is_available():
              return torch.cuda.max_memory_allocated() / 1024 / 1024
      except ImportError:
          pass
      return None

  # Call before + after eval loop:
  torch.cuda.reset_peak_memory_stats() if torch available
  # ... eval loop ...
  vram_peak_mb = _vram_peak_mb()

## VERIFICATION — run fixed eval:

uv run python scripts/eval_finetune.py

# Expected:
# - training_loss_final: ~1.5  (from checkpoint-100)
# - val_loss_final: float or null (only if eval_loss logged)
# - base_pass_rate: stable ~0.20 (consistent with earlier run)
# - ft_pass_rate:   ≥ base_pass_rate
# - base_skipped + ft_skipped: 0 (if Ollama stable)
# - vram_peak_mb: float or null

## MUST NOT
# - ห้าม modify finetune.py, export_gguf.py, dataset files
# - ห้าม re-run training
# - asyncio.Semaphore(1) on ALL model calls
# - pathlib.Path everywhere