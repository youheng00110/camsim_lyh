#!/usr/bin/env bash
set -u

BASE="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh"
GPU_CHECK_SCRIPT="$BASE/run_gpu_check_every_30min.sh"

GPU_IDS="0,1"
UTIL_THRESHOLD=30
IDLE_SECONDS_REQUIRED=$((2 * 60 * 60))
MONITOR_INTERVAL=300

LOCK_DIR="$BASE/gpu_idle_monitor.lock"

cd "$BASE" || exit 1
mkdir -p "$BASE/gpu_check_logs"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[WARN] gpu idle monitor already running, exit."
    exit 0
fi

trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found."
    exit 1
fi

if [ ! -f "$GPU_CHECK_SCRIPT" ]; then
    echo "[ERROR] gpu check script not found: $GPU_CHECK_SCRIPT"
    exit 1
fi

echo "===== $(date '+%Y-%m-%d %H:%M:%S') start gpu idle monitor ====="
echo "[INFO] monitor gpu ids: $GPU_IDS"
echo "[INFO] condition: all monitored GPUs < ${UTIL_THRESHOLD}% for ${IDLE_SECONDS_REQUIRED}s"
echo "[INFO] monitor interval: ${MONITOR_INTERVAL}s"

LOW_START=0
LOW_SECONDS=0

while true
do
    NOW_TS=$(date +%s)
    NOW_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    UTIL_LINES="$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"

    ALL_LOW=1
    UTIL_TEXT=""

    IFS=',' read -ra WATCH_GPU_ARR <<< "$GPU_IDS"

    for gpu_id in "${WATCH_GPU_ARR[@]}"
    do
        gpu_id="$(echo "$gpu_id" | tr -d '[:space:]')"

        util="$(echo "$UTIL_LINES" | awk -F',' -v id="$gpu_id" '
            {
                idx=$1
                val=$2
                gsub(/ /, "", idx)
                gsub(/ /, "", val)
                if (idx == id) {
                    print val
                    exit
                }
            }
        ')"

        if [ -z "$util" ]; then
            echo "[WARN] cannot read gpu${gpu_id} utilization, reset timer."
            util=999
        fi

        UTIL_TEXT="${UTIL_TEXT} gpu${gpu_id}=${util}%"

        if [ "$util" -ge "$UTIL_THRESHOLD" ]; then
            ALL_LOW=0
        fi
    done

    if [ "$ALL_LOW" -eq 1 ]; then
        if [ "$LOW_START" -eq 0 ]; then
            LOW_START="$NOW_TS"
        fi

        LOW_SECONDS=$((NOW_TS - LOW_START))
        REMAIN_SECONDS=$((IDLE_SECONDS_REQUIRED - LOW_SECONDS))

        if [ "$REMAIN_SECONDS" -lt 0 ]; then
            REMAIN_SECONDS=0
        fi

        echo "[INFO] ${NOW_TIME}${UTIL_TEXT}; idle ${LOW_SECONDS}s, remain ${REMAIN_SECONDS}s"
    else
        LOW_START=0
        LOW_SECONDS=0
        echo "[INFO] ${NOW_TIME}${UTIL_TEXT}; not idle enough, reset timer."
    fi

    if [ "$LOW_START" -ne 0 ] && [ "$LOW_SECONDS" -ge "$IDLE_SECONDS_REQUIRED" ]; then
        echo "[INFO] GPUs have been below ${UTIL_THRESHOLD}% for 2 hours."
        echo "[INFO] start gpu hold script: $GPU_CHECK_SCRIPT"
        bash "$GPU_CHECK_SCRIPT"
        exit $?
    fi

    sleep "$MONITOR_INTERVAL"
done
