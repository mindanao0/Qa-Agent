#!/usr/bin/env bash
# ==============================================================================
#  pull_models.sh — Pull required Ollama models and verify them
#
#  Models:
#    - qwen2.5-coder:7b-instruct-q4_K_M  (primary LLM, ~4.5GB VRAM)
#    - nomic-embed-text                   (embedding model, ~300MB)
#
#  Usage (from WSL2 Ubuntu terminal):
#    chmod +x scripts/pull_models.sh
#    ./scripts/pull_models.sh
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

OLLAMA_API="http://localhost:11434"

# ── Ensure Ollama is running ───────────────────────────────────────────────────
ensure_ollama_running() {
    if ! curl -sf "${OLLAMA_API}/api/tags" >/dev/null 2>&1; then
        info "Ollama not responding — attempting to start…"
        if command -v ollama &>/dev/null; then
            nohup ollama serve >/tmp/ollama.log 2>&1 &
            local attempts=0
            while ! curl -sf "${OLLAMA_API}/api/tags" >/dev/null 2>&1; do
                attempts=$((attempts + 1))
                if [ "$attempts" -ge 15 ]; then
                    die "Ollama did not start after 45 seconds. Check /tmp/ollama.log"
                fi
                info "  Waiting for Ollama… ($attempts/15)"
                sleep 3
            done
            success "Ollama started"
        else
            die "Ollama is not installed. Run scripts/setup_wsl2.sh first."
        fi
    else
        success "Ollama is running"
    fi
}

# ── Print VRAM usage via nvidia-smi ───────────────────────────────────────────
print_vram_usage() {
    local label="${1:-Current}"
    if command -v nvidia-smi &>/dev/null; then
        local used free total
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
        echo -e "  ${BLUE}VRAM${NC} [${label}]: used=${used}MB  free=${free}MB  total=${total}MB"
    else
        warn "nvidia-smi not available — skipping VRAM report"
    fi
}

# ── Pull a single model ────────────────────────────────────────────────────────
pull_model() {
    local model="$1"
    info "Pulling model: ${model}"
    print_vram_usage "before pull"

    if ollama pull "$model"; then
        success "Pulled: ${model}"
    else
        die "Failed to pull ${model}"
    fi

    print_vram_usage "after pull"
}

# ── Verify a model with a test inference call ─────────────────────────────────
verify_model() {
    local model="$1"
    local prompt="$2"
    info "Verifying model: ${model}"

    local response
    response=$(curl -sf "${OLLAMA_API}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${model}\", \"prompt\": \"${prompt}\", \"stream\": false, \"options\": {\"temperature\": 0.1, \"num_predict\": 32}}" \
        2>/dev/null || echo "")

    if [ -z "$response" ]; then
        warn "No response from model ${model} — Ollama may still be loading it"
        return 1
    fi

    # Check that the response field is non-empty
    local text
    text=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response',''))" 2>/dev/null || echo "")

    if [ -n "$text" ]; then
        success "Model ${model} responded: \"${text:0:80}…\""
        return 0
    else
        warn "Model ${model} returned an empty response"
        return 1
    fi
}

# ── Verify embedding model ────────────────────────────────────────────────────
verify_embedding_model() {
    local model="$1"
    info "Verifying embedding model: ${model}"

    local response
    response=$(curl -sf "${OLLAMA_API}/api/embeddings" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${model}\", \"prompt\": \"test embedding\"}" \
        2>/dev/null || echo "")

    if [ -z "$response" ]; then
        warn "No response from embedding model ${model}"
        return 1
    fi

    local dim
    dim=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d.get('embedding',[]); print(len(e))" 2>/dev/null || echo "0")

    if [ "$dim" -gt 0 ] 2>/dev/null; then
        success "Embedding model ${model} produced vector of dimension ${dim}"
        return 0
    else
        warn "Embedding model ${model} returned zero-length embedding"
        return 1
    fi
}

# ── List currently available models ──────────────────────────────────────────
list_models() {
    info "Currently available models:"
    curl -sf "${OLLAMA_API}/api/tags" 2>/dev/null \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if not models:
    print('  (none)')
else:
    for m in models:
        size_gb = m.get('size', 0) / 1e9
        print(f\"  - {m['name']}  ({size_gb:.1f} GB)\")
" 2>/dev/null || warn "Could not parse model list"
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║      Universal Local AI QA Agent            ║${NC}"
    echo -e "${BLUE}║      Model Pull & Verification               ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    ensure_ollama_running
    echo ""

    print_vram_usage "baseline"
    echo ""

    list_models
    echo ""

    # ── Pull primary LLM ──────────────────────────────────────────────────────
    LLM_MODEL="qwen2.5-coder:7b-instruct-q4_K_M"
    pull_model "$LLM_MODEL"
    echo ""

    # ── Pull embedding model ──────────────────────────────────────────────────
    EMBED_MODEL="nomic-embed-text"
    pull_model "$EMBED_MODEL"
    echo ""

    print_vram_usage "after all pulls"
    echo ""

    # ── Verification ─────────────────────────────────────────────────────────
    echo -e "${BLUE}── Verification ─────────────────────────────────────────────${NC}"

    # Wait a moment for models to be ready
    sleep 2

    LLM_OK=0
    EMBED_OK=0

    verify_model "$LLM_MODEL" \
        "Write a one-line Python comment describing what a Playwright test does." \
        && LLM_OK=1 || true

    verify_embedding_model "$EMBED_MODEL" && EMBED_OK=1 || true

    echo ""

    # ── Final summary ─────────────────────────────────────────────────────────
    echo -e "${BLUE}── Summary ──────────────────────────────────────────────────${NC}"
    if [ "$LLM_OK" -eq 1 ]; then
        echo -e "  ${GREEN}✓${NC}  ${LLM_MODEL}"
    else
        echo -e "  ${YELLOW}⚠${NC}  ${LLM_MODEL}  (pulled but inference check failed — may still be loading)"
    fi

    if [ "$EMBED_OK" -eq 1 ]; then
        echo -e "  ${GREEN}✓${NC}  ${EMBED_MODEL}"
    else
        echo -e "  ${YELLOW}⚠${NC}  ${EMBED_MODEL}  (pulled but embedding check failed — may still be loading)"
    fi

    echo ""

    if [ "$LLM_OK" -eq 1 ] && [ "$EMBED_OK" -eq 1 ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║       All models ready. System is good ✓    ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
    else
        echo -e "${YELLOW}╔══════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  Models pulled. Re-run verification later.  ║${NC}"
        echo -e "${YELLOW}╚══════════════════════════════════════════════╝${NC}"
    fi

    echo ""
    echo "Next step — generate your first test:"
    echo "  python main.py --mode generate \\"
    echo "    --requirement 'Test the monthly payroll calculation' \\"
    echo "    --url http://localhost:3000 --role payroll_officer --domain payroll_calculation"
    echo ""
}

main "$@"
