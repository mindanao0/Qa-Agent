<#
.SYNOPSIS
    Register the fine-tuned GGUF model in Ollama and point config\agent.yaml at it.

.DESCRIPTION
    After WSL2 training completes, the GGUF file and its Modelfile land in
    D:\Code\qa-agent\models\. This script:

      1. Verifies models\ exists and contains qa-agent-coder-q4_k_m.gguf + Modelfile.
      2. Runs `ollama create qa-agent-coder -f Modelfile` to register the model.
      3. Smoke-tests with a short prompt; passes if the response contains "def test_".
      4. Rewrites the `model:` line in config\agent.yaml to "qa-agent-coder".

.NOTES
    Run from the project root (D:\Code\qa-agent) or any directory — the script
    resolves paths relative to its own location.
#>

[CmdletBinding()]
param(
    [string]$ModelTag = 'qa-agent-coder'
)

$ErrorActionPreference = 'Stop'

function Write-Section([string]$Title) {
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan
}

# ── Resolve paths relative to script location ────────────────────────────────
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir '..')
$modelsDir   = Join-Path $projectRoot 'models'
$ggufFile    = Join-Path $modelsDir   'qa-agent-coder-q4_k_m.gguf'
$modelfile   = Join-Path $modelsDir   'Modelfile'
$agentYaml   = Join-Path $projectRoot 'config\agent.yaml'

Write-Section "Ollama model registration — $ModelTag"
Write-Host "Project root: $projectRoot"
Write-Host "Models dir:   $modelsDir"

# ── Step 1 — Verify artifacts exist ──────────────────────────────────────────
if (-not (Test-Path $modelsDir)) {
    Write-Host "ERROR: models\ directory not found at $modelsDir" -ForegroundColor Red
    Write-Host 'Run WSL2 training first: bash scripts/run_training_wsl2.sh' -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $ggufFile)) {
    Write-Host "ERROR: GGUF file not found at $ggufFile" -ForegroundColor Red
    Write-Host 'Run WSL2 training first: bash scripts/run_training_wsl2.sh' -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $modelfile)) {
    Write-Host "ERROR: Modelfile not found at $modelfile" -ForegroundColor Red
    Write-Host 'Re-run WSL2 export: python3 wsl2_trainer.py --export-only ...' -ForegroundColor Yellow
    exit 1
}
Write-Host '  [ok] Found GGUF + Modelfile' -ForegroundColor Green

# ── Step 2 — Verify ollama is installed and running ──────────────────────────
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host 'ERROR: `ollama` CLI not found in PATH. Install from https://ollama.com/' -ForegroundColor Red
    exit 1
}

# ── Step 3 — Register model (run from models\ so FROM ./*.gguf resolves) ─────
Write-Section "Registering $ModelTag with Ollama"
Push-Location $modelsDir
try {
    & ollama create $ModelTag -f Modelfile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: ollama create failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
Write-Host "  [ok] Model registered as $ModelTag" -ForegroundColor Green

# ── Step 4 — Smoke test ──────────────────────────────────────────────────────
Write-Section 'Smoke test'
$prompt = 'Write a simple pytest-playwright test for login page'
Write-Host "Prompt: $prompt"
try {
    $response = & ollama run $ModelTag $prompt 2>&1 | Out-String
} catch {
    Write-Host "ERROR: ollama run failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Write-Host ''
Write-Host '--- Response (first 1000 chars) ---' -ForegroundColor DarkGray
$snippet = if ($response.Length -gt 1000) { $response.Substring(0, 1000) + '...' } else { $response }
Write-Host $snippet
Write-Host '-----------------------------------' -ForegroundColor DarkGray

if ($response -match 'def\s+test_') {
    Write-Host "  [ok] Response contains 'def test_' — smoke test PASSED" -ForegroundColor Green
} else {
    Write-Host "  [warn] Response does NOT contain 'def test_' — model may need more training" -ForegroundColor Yellow
}

# ── Step 5 — Update config\agent.yaml ────────────────────────────────────────
Write-Section "Updating $agentYaml -> llm.model = $ModelTag"
if (-not (Test-Path $agentYaml)) {
    Write-Host "WARN: $agentYaml not found. Skipping config update." -ForegroundColor Yellow
} else {
    $original = Get-Content $agentYaml -Raw -Encoding UTF8
    # Rewrite the first `  model: "..."` line. The llm: block is at the top of the file,
    # so the first quoted model: line belongs to llm:.
    $updated = [regex]::Replace(
        $original,
        '(?m)^(\s*model:\s*)"[^"]*"\s*$',
        ('${1}"' + $ModelTag + '"'),
        1
    )
    if ($updated -eq $original) {
        # Fall back to unquoted form
        $updated = [regex]::Replace(
            $original,
            '(?m)^(\s*model:\s*)\S+\s*$',
            ('${1}"' + $ModelTag + '"'),
            1
        )
    }
    if ($updated -eq $original) {
        Write-Host "WARN: Could not locate 'model:' line in agent.yaml. Edit manually." -ForegroundColor Yellow
    } else {
        # Write back as UTF-8 *without* BOM so PyYAML on Linux stays happy.
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($agentYaml, $updated, $utf8NoBom)
        Write-Host "  [ok] config\agent.yaml llm.model -> $ModelTag" -ForegroundColor Green
    }
}

Write-Host ''
Write-Host "Model registered successfully as $ModelTag" -ForegroundColor Green
Write-Host 'Verify with:' -ForegroundColor Cyan
Write-Host '    ollama list'
Write-Host '    uv run python main.py --mode generate --requirement "test login flow" --url http://localhost:3000 --role admin'
