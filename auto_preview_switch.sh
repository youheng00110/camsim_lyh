#!/usr/bin/env bash
set -euo pipefail

WATCH_DIR="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_implicit_preview/preview"
TMUX_TARGET="1:0.0"
TARGET_COUNT=199
CHECK_INTERVAL=10

GPU_CHECK_SCRIPT="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/run_gpu_check_every_30min.sh"

echo "[INFO] watching dir: ${WATCH_DIR}"
echo "[INFO] target mp4 count: ${TARGET_COUNT}"
echo "[INFO] target tmux pane: ${TMUX_TARGET}"
echo "[INFO] gpu check script: ${GPU_CHECK_SCRIPT}"

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

if [[ ! -f "${GPU_CHECK_SCRIPT}" ]]; then
  echo "[ERROR] gpu check script not found: ${GPU_CHECK_SCRIPT}"
  exit 1
fi

while true; do
  mp4_count="$(find "${WATCH_DIR}" -maxdepth 1 -type f -iname "*.mp4" 2>/dev/null | wc -l)"
  mp4_count="$(echo "${mp4_count}" | tr -d ' ')"

  echo "[INFO] current mp4 count: ${mp4_count}"

  if (( mp4_count >= TARGET_COUNT )); then
    echo "[INFO] mp4 count reached ${TARGET_COUNT}; stopping old preview."
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
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
      if [[ -n "${pgid}" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
      fi
    done

    sleep 8
  fi
fi

echo "[INFO] running gpu check script"
bash "${GPU_CHECK_SCRIPT}"

echo "[INFO] all done."