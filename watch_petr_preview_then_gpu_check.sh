#!/usr/bin/env bash
set -euo pipefail

BASE="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh"

WATCH_DIR="$BASE/output/debug_token_preview/preview"
TMUX_TARGET="0:0.0"

TARGET_COUNT=200
TARGET_LAST_INDEX=199
CHECK_INTERVAL=10

GPU_IDLE_MONITOR_SCRIPT="$BASE/watch_gpu_idle_then_hold.sh"
GPU_IDLE_MONITOR_LOG="$BASE/gpu_check_logs/gpu_idle_monitor.log"

mkdir -p "$BASE/gpu_check_logs"

echo "[INFO] watching dir: ${WATCH_DIR}"
echo "[INFO] target mp4 count: ${TARGET_COUNT}"
echo "[INFO] target last file: ${TARGET_LAST_INDEX}.mp4"
echo "[INFO] target tmux pane: ${TMUX_TARGET}"
echo "[INFO] gpu idle monitor script: ${GPU_IDLE_MONITOR_SCRIPT}"
echo "[INFO] gpu idle monitor log: ${GPU_IDLE_MONITOR_LOG}"

if ! tmux has-session -t 0 2>/dev/null; then
  echo "[ERROR] tmux session 0 not found."
  exit 1
fi

if [[ -n "${TMUX:-}" ]]; then
  CURRENT_PANE="$(tmux display-message -p '#{session_name}:#{window_index}.#{pane_index}' 2>/dev/null || true)"
  if [[ "${CURRENT_PANE}" == "${TMUX_TARGET}" ]]; then
    echo "[ERROR] Do not run this script inside target pane ${TMUX_TARGET}; it would Ctrl-C itself."
    exit 1
  fi
fi

if [[ ! -d "${WATCH_DIR}" ]]; then
  echo "[ERROR] watch dir not found: ${WATCH_DIR}"
  exit 1
fi

if [[ ! -f "${GPU_IDLE_MONITOR_SCRIPT}" ]]; then
  echo "[ERROR] gpu idle monitor script not found: ${GPU_IDLE_MONITOR_SCRIPT}"
  exit 1
fi

echo "[INFO] starting gpu idle monitor in background"
nohup bash "${GPU_IDLE_MONITOR_SCRIPT}" >> "${GPU_IDLE_MONITOR_LOG}" 2>&1 &

GPU_IDLE_MONITOR_PID=$!
echo "[INFO] gpu idle monitor pid: ${GPU_IDLE_MONITOR_PID}"

while true; do
  mp4_count="$(find "${WATCH_DIR}" -maxdepth 1 -type f -iname "*.mp4" 2>/dev/null | wc -l)"
  mp4_count="$(echo "${mp4_count}" | tr -d '[:space:]')"

  last_file="${WATCH_DIR}/${TARGET_LAST_INDEX}.mp4"

  if [[ -f "${last_file}" ]]; then
    last_exists="yes"
  else
    last_exists="no"
  fi

  echo "[INFO] current mp4 count: ${mp4_count}; ${TARGET_LAST_INDEX}.mp4 exists: ${last_exists}"

  if (( mp4_count >= TARGET_COUNT )); then
    if [[ -f "${last_file}" ]]; then
      echo "[INFO] reached ${TARGET_COUNT} videos and found ${TARGET_LAST_INDEX}.mp4"
    else
      echo "[INFO] reached ${TARGET_COUNT} videos, but ${TARGET_LAST_INDEX}.mp4 not found. Still stop."
    fi
    break
  fi

  sleep "${CHECK_INTERVAL}"
done

echo "[INFO] sending Ctrl-C to tmux pane ${TMUX_TARGET}"
tmux send-keys -t "${TMUX_TARGET}" C-c

sleep 10

PANE_PID="$(tmux display-message -p -t "${TMUX_TARGET}" '#{pane_pid}' 2>/dev/null || true)"
if [[ -n "${PANE_PID}" ]]; then
  mapfile -t CHILD_PIDS < <(pgrep -P "${PANE_PID}" || true)

  if (( ${#CHILD_PIDS[@]} > 0 )); then
    echo "[WARN] target pane still has child processes; sending TERM to their process groups."

    for pid in "${CHILD_PIDS[@]}"; do
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)
      if [[ -n "${pgid}" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
      fi
    done

    sleep 8
  fi
fi

echo "[INFO] preview monitor done."
echo "[INFO] gpu idle monitor is independent. Check log:"
echo "[INFO] ${GPU_IDLE_MONITOR_LOG}"
