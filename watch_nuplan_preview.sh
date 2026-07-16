#!/usr/bin/env bash
set -euo pipefail

WATCH_DIR="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_nuplantv24000/preview"
TMUX_TARGET="0:0.0"

THRESHOLD=200
POLL_SECONDS=5
STOP_TIMEOUT=30

if [[ "${1:-}" == "--run-next" ]]; then
    source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

    cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src

    unset ENABLE_DEBUGPY
    unset DEBUGPY_PORT

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1

    export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
    export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
    export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
    export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"

    exec torchrun \
        --standalone \
        --nproc_per_node=4 \
        -m dwm.preview \
        -c /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim/nuplanboxpreview.json \
        -o /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_nuplanbox30000
fi

if ! tmux display-message -p -t "$TMUX_TARGET" '#{pane_id}' >/dev/null 2>&1; then
    echo "[ERROR] 找不到 tmux pane $TMUX_TARGET"
    exit 1
fi

SCRIPT_PATH="$(readlink -f "$0")"
printf -v START_COMMAND 'bash %q --run-next' "$SCRIPT_PATH"

echo "[INFO] 正在监控 $WATCH_DIR"
echo "[INFO] mp4 数量大于 $THRESHOLD 时切换任务"

LAST_COUNT=-1

while true; do
    if [[ -d "$WATCH_DIR" ]]; then
        FILE_COUNT="$(
            find "$WATCH_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l
        )"
    else
        FILE_COUNT=0
    fi

    if (( FILE_COUNT != LAST_COUNT )); then
        echo "[INFO] 当前 mp4 数量 $FILE_COUNT"
        LAST_COUNT="$FILE_COUNT"
    fi

    if (( FILE_COUNT > THRESHOLD )); then
        echo "[INFO] 已达到触发条件，停止原任务"

        tmux send-keys -t "$TMUX_TARGET" C-c

        for ((WAITED = 0; WAITED < STOP_TIMEOUT; WAITED++)); do
            sleep 1

            CURRENT_COMMAND="$(
                tmux display-message \
                    -p \
                    -t "$TMUX_TARGET" \
                    '#{pane_current_command}'
            )"

            if [[ "$CURRENT_COMMAND" == "bash" ||
                  "$CURRENT_COMMAND" == "zsh" ||
                  "$CURRENT_COMMAND" == "sh" ]]; then
                break
            fi
        done

        CURRENT_COMMAND="$(
            tmux display-message \
                -p \
                -t "$TMUX_TARGET" \
                '#{pane_current_command}'
        )"

        if [[ "$CURRENT_COMMAND" != "bash" &&
              "$CURRENT_COMMAND" != "zsh" &&
              "$CURRENT_COMMAND" != "sh" ]]; then
            echo "[WARN] 原任务未正常退出，强制重启 pane"
            tmux respawn-pane -k -t "$TMUX_TARGET" bash
            sleep 2
        fi

        echo "[INFO] 启动 nuplanbox30000"

        tmux send-keys -t "$TMUX_TARGET" -l "$START_COMMAND"
        tmux send-keys -t "$TMUX_TARGET" Enter

        echo "[INFO] 新任务已提交"
        exit 0
    fi

    sleep "$POLL_SECONDS"
done
