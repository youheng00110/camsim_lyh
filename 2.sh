#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_pvonly_preview"
PREVIEW_TMUX_SESSION="pvonly_preview_8gpu"
GPU_CHECK_SCRIPT="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/run_gpu_check_every_30min.sh"

TARGET_MP4=200
CHECK_INTERVAL=30

mkdir -p "${OUT_DIR}/logs"

echo "[watch] waiting for mp4 count >= ${TARGET_MP4}"
echo "[watch] output dir: ${OUT_DIR}"

while true; do
    MP4_COUNT=$(find "${OUT_DIR}" -type f -iname "*.mp4" | wc -l | tr -d ' ')

    date
    echo "[watch] mp4 count: ${MP4_COUNT}/${TARGET_MP4}"

    if [[ "${MP4_COUNT}" -ge "${TARGET_MP4}" ]]; then
        sleep 10
        MP4_COUNT_2=$(find "${OUT_DIR}" -type f -iname "*.mp4" | wc -l | tr -d ' ')
        echo "[watch] mp4 count recheck: ${MP4_COUNT_2}/${TARGET_MP4}"

        if [[ "${MP4_COUNT_2}" -ge "${TARGET_MP4}" ]]; then
            break
        fi
    fi

    sleep "${CHECK_INTERVAL}"
done

echo "[watch] target reached, stopping preview"

if tmux has-session -t "${PREVIEW_TMUX_SESSION}" 2>/dev/null; then
    tmux send-keys -t "${PREVIEW_TMUX_SESSION}" C-c
    sleep 10
    tmux send-keys -t "${PREVIEW_TMUX_SESSION}" C-c
    sleep 5
    tmux kill-session -t "${PREVIEW_TMUX_SESSION}" || true
else
    echo "[warn] preview tmux session not found, try killing torchrun preview process"

    PIDS=$(ps -eo pid,args | awk '/torchrun/ && /dwm.preview/ && /nuplanpvonlypreview.json/ && !/awk/ {print $1}')

    if [[ -n "${PIDS}" ]]; then
        echo "${PIDS}" | xargs -r kill -TERM
        sleep 10
        echo "${PIDS}" | xargs -r kill -KILL || true
    fi
fi

echo "[watch] starting gpu check script"

LOG_FILE="${OUT_DIR}/logs/gpu_check_$(date +%Y%m%d_%H%M%S).log"

bash "${GPU_CHECK_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
