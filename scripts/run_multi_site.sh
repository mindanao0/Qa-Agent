#!/usr/bin/env bash
# run_multi_site.sh — รัน Universal QA Agent บน 14 sites + collect training data
# ทำงานต่อเนื่องจนครบ MAX_HOURS ชั่วโมง (default 6)
#
# Usage (จากที่ไหนก็ได้ — สคริปต์ cd เข้า repo root เอง):
#   bash scripts/run_multi_site.sh           # รัน 6 ชั่วโมง
#   bash scripts/run_multi_site.sh 3         # รัน 3 ชั่วโมง
#   bash scripts/run_multi_site.sh 0 saucedemo  # รัน 1 รอบ เริ่มจาก site ที่ระบุ
#
# Credentials: อ่านจาก eval/test_credentials.json (gitignored) — ห้ามฝัง user/pass
# ในสคริปต์นี้ (แม่แบบอยู่ที่ eval/test_credentials.example.json)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source "$HOME/.local/bin/env" 2>/dev/null || true
export PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu24.04-x64

MAX_HOURS="${1:-6}"
START_FROM="${2:-}"
MAX_SECONDS=$(( MAX_HOURS * 3600 ))
START_TIME=$(date +%s)

REPORT_DIR="reports/multi_site"
DETAIL_DIR="reports/multi_site/detail"
TRAIN_FILE="data/training/train.jsonl"
CRED_FILE="eval/test_credentials.json"
mkdir -p "$REPORT_DIR" "$DETAIL_DIR" "$(dirname "$TRAIN_FILE")"

# 14 sites: label|url — sites ที่ต้อง login ต้องมี key ตรงกับ label ใน CRED_FILE
SITES=(
    "saucedemo|https://www.saucedemo.com"
    "practicesoftwaretesting|https://practicesoftwaretesting.com"
    "automationexercise|https://automationexercise.com"
    "the-internet|https://the-internet.herokuapp.com"
    "demoqa|https://demoqa.com"
    "orangehrm|https://opensource-demo.orangehrmlive.com"
    "parabank|https://parabank.parasoft.com"
    "nopcommerce|https://demo.nopcommerce.com"
    "magento|https://magento.softwaretestingboard.com"
    "moodle|https://school.moodledemo.net"
    "phptravels|https://phptravels.net/demo"
    "redmine|https://demo.redmine.org"
    "kanboard|https://demo.kanboard.org"
    "prestashop|https://demo.prestashop.com/#/en"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# get_cred <label> <username|password> — ว่าง = site นั้นไม่ใช้ login
get_cred() {
    python3 - "$CRED_FILE" "$1" "$2" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
    print(data.get(sys.argv[2], {}).get(sys.argv[3], ""))
except (FileNotFoundError, json.JSONDecodeError):
    print("")
PY
}

elapsed_seconds() { echo $(( $(date +%s) - START_TIME )); }

remaining_seconds() { echo $(( MAX_SECONDS - $(elapsed_seconds) )); }

run_site() {
    local label="$1" url="$2"
    local username password
    username="$(get_cred "$label" username)"
    password="$(get_cred "$label" password)"
    local detail_out="$DETAIL_DIR/${label}_$(date '+%Y%m%d_%H%M%S').json"
    local log_out="$REPORT_DIR/${label}.txt"

    log "==============================="
    log "SITE: $label"
    log "URL:  $url"
    log "==============================="

    local cmd=(uv run python -m src.universal_qa
        --url "$url"
        --max-pages 30
        --explore-timeout 4
        --headless
        --detail-output "$detail_out"
    )
    [[ -n "$username" ]] && cmd+=(--username "$username" --password "$password")

    # Run agent, capture output
    if "${cmd[@]}" 2>&1 | tee "$log_out"; then
        log "Agent finished for $label"
    else
        log "Agent failed/timed-out for $label — continuing"
    fi

    # Collect training data from successful run
    if [[ -f "$detail_out" ]]; then
        uv run python scripts/collect_test_training_data.py \
            --results "$detail_out" \
            --site-label "$label" \
            --output "$TRAIN_FILE" 2>&1 || true
    fi
}

# ─── main loop ────────────────────────────────────────────────────────────────
ROUND=0
while true; do
    ROUND=$(( ROUND + 1 ))
    REMAINING=$(remaining_seconds)
    if (( REMAINING <= 0 )); then
        log "Time limit reached (${MAX_HOURS}h). Stopping."
        break
    fi
    log "=== Round $ROUND — ${REMAINING}s remaining ==="

    ACTIVE=false
    [[ -z "$START_FROM" ]] && ACTIVE=true

    for entry in "${SITES[@]}"; do
        IFS='|' read -r label url <<< "$entry"

        # --start-from: skip until we reach that label (only affects Round 1)
        if [[ $ROUND -eq 1 && -n "$START_FROM" && ! "$ACTIVE" == "true" ]]; then
            [[ "$label" == "$START_FROM" ]] && ACTIVE=true || continue
        fi

        REMAINING=$(remaining_seconds)
        if (( REMAINING <= 60 )); then
            log "Less than 60s remaining — stopping."
            break 2
        fi

        run_site "$label" "$url"
        log "Training data: $(wc -l < "$TRAIN_FILE" 2>/dev/null || echo 0) examples total"
    done

    # After first round, always start from the first site
    START_FROM=""

    REMAINING=$(remaining_seconds)
    log "Round $ROUND complete. ${REMAINING}s remaining."
    (( REMAINING <= 0 )) && break
done

ELAPSED=$(elapsed_seconds)
TOTAL_EXAMPLES=$(wc -l < "$TRAIN_FILE" 2>/dev/null || echo 0)
log "=== ALL DONE === elapsed=${ELAPSED}s | training_examples=${TOTAL_EXAMPLES}"
echo "{\"elapsed_seconds\": $ELAPSED, \"training_examples\": $TOTAL_EXAMPLES, \"rounds\": $ROUND}"
