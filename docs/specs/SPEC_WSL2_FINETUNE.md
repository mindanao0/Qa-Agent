# SPEC_WSL2_FINETUNE — WSL2 Fine-tuning Pipeline + VRAM Safety (UPDATE 4)

Implement 4 components required for safe QLoRA fine-tuning on 6GB VRAM Windows 11 + WSL2.

---

### COMPONENT 1: scripts/fix_wddm_tdr.ps1 (Windows PowerShell)

Create a PowerShell script that fixes GPU timeout during long training runs.
Must run as Administrator.

The script must:

1. Set these registry values under HKLM\System\CurrentControlSet\Control\GraphicsDrivers:
   - TdrLevel = 3 (DWORD)
   - TdrDelay = 60 (DWORD)
   - TdrDdiDelay = 60 (DWORD)

2. Print confirmation of each registry write

3. Print warning:
   "RESTART REQUIRED: Changes take effect after reboot.
    Without this fix, GPU driver may timeout during training (training will crash after ~2 min)."

4. Ask user: "Restart now? (y/n)"
   If y: Restart-Computer -Force

---

### COMPONENT 2: scripts/setup_wsl2_cuda.sh (WSL2 Bash)

Create a complete WSL2 CUDA + Unsloth setup script.

The script must execute these steps in order:

Step 1 — Verify WSL2 has GPU access:
  nvidia-smi
  If fails: print "GPU not visible in WSL2. Enable in Windows: nvidia-smi -pm 1" and exit 1

Step 2 — Install CUDA toolkit 12.x (WSL2 Ubuntu):
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-4

Step 3 — Install Python 3.11 and uv:
  sudo add-apt-repository ppa:deadsnakes/ppa -y
  sudo apt install -y python3.11 python3.11-venv python3.11-pip
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc

Step 4 — Create WSL2 venv for training (separate from Windows project):
  cd ~
  python3.11 -m venv ~/.qa-finetune-env
  source ~/.qa-finetune-env/bin/activate

Step 5 — Install PyTorch with CUDA 12.4:
  pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Step 6 — Install Unsloth:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes
  pip install transformers datasets

Step 7 — Verify GPU is accessible from Python:
  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
  Expected output: CUDA: True, VRAM: ~6.0GB

Step 8 — Copy project files to WSL2 home:
  cp -r /mnt/d/Code/qa-agent/src/finetune ~/qa-finetune/
  cp -r /mnt/c/Users/Da00/.qa-agent/datasets ~/qa-finetune/

Print final message:
  "WSL2 CUDA setup complete.
   Run training with: bash scripts/run_training_wsl2.sh"

---

### COMPONENT 3: scripts/run_training_wsl2.sh (WSL2 Bash)

Create the actual training runner script for WSL2.

The script must:

Step 1 — Activate venv:
  source ~/.qa-finetune-env/bin/activate

Step 2 — Check dataset exists and has enough examples:
  DATASET="$HOME/qa-finetune/datasets/synthetic_universal.jsonl"
  COUNT=$(wc -l < "$DATASET")
  If COUNT < 200: print "WARNING: Only $COUNT examples. Recommended 500+. Continue? (y/n)"

Step 3 — Check free VRAM:
  python3 -c "
  import torch
  free = torch.cuda.mem_get_info()[0] / 1e9
  print(f'Free VRAM: {free:.1f}GB')
  if free < 5.5:
      print('ERROR: Need 5.5GB free VRAM')
      exit(1)
  "

Step 4 — Run training script:
  python3 ~/qa-finetune/finetune/wsl2_trainer.py \
    --dataset "$DATASET" \
    --output "$HOME/.qa-agent/checkpoints" \
    --max-steps 300

Step 5 — After training completes, export GGUF:
  python3 ~/qa-finetune/finetune/wsl2_trainer.py \
    --export-only \
    --checkpoint "$HOME/.qa-agent/checkpoints" \
    --output "$HOME/.qa-agent/gguf"

Step 6 — Copy GGUF to Windows accessible path:
  cp "$HOME/.qa-agent/gguf/qa-agent-coder-q4_k_m.gguf" "/mnt/d/Code/qa-agent/models/"
  cp "$HOME/.qa-agent/gguf/Modelfile" "/mnt/d/Code/qa-agent/models/"
  Print: "GGUF copied to D:\Code\qa-agent\models\ — register in Ollama from Windows PowerShell"

---

### COMPONENT 4: src/finetune/wsl2_trainer.py (Python — runs inside WSL2)

Create a standalone Python training script that works inside WSL2 with Unsloth.
This is separate from trainer.py (which was for Windows) — this one actually works.

The script must have:

VRAM Watchdog class:
  - Background thread checking VRAM every 5 seconds using torch.cuda.mem_get_info()
  - If free VRAM < 400MB: log critical warning
  - If free VRAM < 200MB: call trainer.stop() to checkpoint and exit gracefully
  - Log VRAM usage every 30 seconds: "VRAM: {used:.1f}/{total:.1f}GB ({pct:.0f}%)"

Training config (hardcoded for 6GB VRAM safety):
  max_seq_length = 1024  (NOT 2048 — saves ~1GB VRAM)
  load_in_4bit = True
  r = 8  (NOT 16 — saves VRAM on 6GB)
  lora_alpha = 16
  target_modules = ["q_proj", "v_proj"]  (minimal targets for 6GB)
  per_device_train_batch_size = 1
  gradient_accumulation_steps = 8  (effective batch = 8)
  max_steps = 300
  learning_rate = 2e-4
  fp16 = True
  save_steps = 50
  logging_steps = 10

Dataset preparation:
  Load JSONL file
  Filter examples where len(tokens) > max_seq_length (discard too-long examples)
  Add 10% general instruction-following examples to prevent catastrophic forgetting:
    Use these hardcoded general examples (5 examples repeated proportionally):
    - {"instruction": "What is Playwright?", "response": "Playwright is a browser automation library..."}
    - {"instruction": "Explain pytest fixtures", "response": "Fixtures in pytest are functions that..."}
    - {"instruction": "What is get_by_role?", "response": "get_by_role is a Playwright locator..."}
    Interleave with QA examples at 1 general per 9 QA examples ratio

Training loop:
  Log every 10 steps: "Step {step}/{max_steps} | Loss: {loss:.4f} | VRAM: {vram_used:.1f}GB"
  Save checkpoint every 50 steps
  On KeyboardInterrupt: save checkpoint immediately before exit

Export to GGUF (--export-only mode):
  Load checkpoint with FastLanguageModel
  Save as GGUF Q4_K_M to output path
  Create Modelfile with this exact content:
  ---
  FROM ./qa-agent-coder-q4_k_m.gguf
  SYSTEM "You are an expert Playwright Python QA engineer. Generate pytest-playwright
  tests using only semantic locators: get_by_role, get_by_label, get_by_text,
  get_by_test_id. Never use CSS selectors or XPath. Always use expect() for assertions.
  Function names must start with test_."
  PARAMETER temperature 0.1
  PARAMETER top_p 0.9
  PARAMETER num_ctx 2048
  ---

CLI interface (argparse):
  --dataset PATH
  --output PATH
  --max-steps INT (default 300)
  --export-only (skip training, just export existing checkpoint)
  --checkpoint PATH (for export-only mode)

---

### COMPONENT 5: scripts/register_ollama_windows.ps1 (Windows PowerShell)

Create script to register the exported model in Ollama after training.

Steps:
1. Check models\ folder exists and has .gguf file
   If not: print "Run WSL2 training first: bash scripts/run_training_wsl2.sh"

2. cd to D:\Code\qa-agent\models\

3. Register model:
   ollama create qa-agent-coder -f Modelfile

4. Smoke test:
   ollama run qa-agent-coder "Write a simple pytest-playwright test for login page"

5. If response contains "def test_":
   Print "Model registered successfully as qa-agent-coder"
   Print "Update config\agent.yaml: change llm.model to qa-agent-coder"

6. Auto-update config\agent.yaml:
   Read the file, replace the model line, write back

---

### COMPONENT 6: main.py — Update finetune mode

Update --mode finetune --step generate to run FULL dataset generation (not smoke test):

When --step generate is called:
  Call generate_synthetic_dataset() with NO max_per_domain limit
  Expected output: 500-800 examples across all 13 domains
  Show rich progress bar per domain
  At end show table:
  | Domain | Happy Path | Edge Cases | Healing | Total |
  Print: "Dataset ready at ~/.qa-agent/datasets/synthetic_universal.jsonl"
  Print: "Next: Run WSL2 training with: bash scripts/run_training_wsl2.sh"

---

## FULL WORKFLOW INSTRUCTIONS

Add a section to README.md under a new heading "## Fine-tuning on WSL2 (6GB VRAM)"
that documents the complete workflow:

Step 1 — Fix GPU timeout (Windows PowerShell as Admin, ONE TIME ONLY):
  powershell -ExecutionPolicy Bypass -File scripts\fix_wddm_tdr.ps1
  [Restart PC]

Step 2 — Generate full dataset (Windows PowerShell):
  uv run python main.py --mode finetune --step generate
  [Wait 30-60 minutes]

Step 3 — Setup WSL2 CUDA (WSL2 terminal, ONE TIME ONLY):
  bash scripts/setup_wsl2_cuda.sh
  [Wait 10-15 minutes]

Step 4 — Run training (WSL2 terminal):
  bash scripts/run_training_wsl2.sh
  [Wait 1-2 hours]

Step 5 — Register model (Windows PowerShell):
  powershell -ExecutionPolicy Bypass -File scripts\register_ollama_windows.ps1

Step 6 — Verify:
  ollama list  ← should show qa-agent-coder
  uv run python main.py --mode generate --requirement "test checkout flow" --url http://localhost:3000 --role admin
  ← should use qa-agent-coder model now

---

## VERIFICATION

After creating all files, verify structure:
  dir scripts\
  Expected: fix_wddm_tdr.ps1, setup_wsl2_cuda.sh, run_training_wsl2.sh, register_ollama_windows.ps1

  dir src\finetune\
  Expected: __init__.py, trainer.py, wsl2_trainer.py, export.py

Syntax check wsl2_trainer.py:
  python -c "import ast; ast.parse(open('src/finetune/wsl2_trainer.py').read()); print('Syntax OK')"
