#!/usr/bin/env bash
set -euo pipefail

CKPT="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/train_nuplanpvonly/checkpoints/18000.pth"
CONFIG="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim/nuplanpvonlypreview.json"
OUT_DIR="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_pvonly_preview"

TRAIN_TMUX_TARGET="0:0"
PREVIEW_TMUX_SESSION="pvonly_preview_8gpu"

CHECK_INTERVAL=60
STABLE_WAIT=20

echo "[watch] waiting for checkpoint: ${CKPT}"

while [[ ! -f "${CKPT}" ]]; do
    date
    echo "[watch] checkpoint not found, sleep ${CHECK_INTERVAL}s"
    sleep "${CHECK_INTERVAL}"
done

echo "[watch] checkpoint found, checking file size stability"

while true; do
    SIZE_1=$(stat -c%s "${CKPT}")
    sleep "${STABLE_WAIT}"
    SIZE_2=$(stat -c%s "${CKPT}")

    echo "[watch] ckpt size: ${SIZE_1} -> ${SIZE_2}"

    if [[ "${SIZE_1}" == "${SIZE_2}" && "${SIZE_2}" -gt 0 ]]; then
        break
    fi
done

echo "[watch] checkpoint looks stable"

if ! grep -q '"all_rank_preview"[[:space:]]*:[[:space:]]*true' "${CONFIG}"; then
    echo "[warn] ${CONFIG} may not contain \"all_rank_preview\": true"
fi

echo "[watch] stopping training tmux target: ${TRAIN_TMUX_TARGET}"

if tmux has-session -t "0" 2>/dev/null; then
    tmux send-keys -t "${TRAIN_TMUX_TARGET}" C-c
    sleep 10
    tmux send-keys -t "${TRAIN_TMUX_TARGET}" C-c
    sleep 20
else
    echo "[warn] tmux session 0 not found"
fi

mkdir -p "${OUT_DIR}/logs"

RUNNER="${OUT_DIR}/run_pvonly_preview_8gpu.sh"

cat > "${RUNNER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src

unset ENABLE_DEBUGPY
unset DEBUGPY_PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH
export PYTHONPATH=$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH
export PYTHONPATH=$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH

torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m dwm.preview \
  -c /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim/nuplanpvonlypreview.json \
  -o /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_pvonly_preview
EOF

chmod +x "${RUNNER}"

if tmux has-session -t "${PREVIEW_TMUX_SESSION}" 2>/dev/null; then
    echo "[watch] old preview tmux exists, killing: ${PREVIEW_TMUX_SESSION}"
    tmux kill-session -t "${PREVIEW_TMUX_SESSION}"
fi

LOG_FILE="${OUT_DIR}/logs/preview_$(date +%Y%m%d_%H%M%S).log"

echo "[watch] starting preview tmux session: ${PREVIEW_TMUX_SESSION}"
tmux new-session -d -s "${PREVIEW_TMUX_SESSION}" "bash '${RUNNER}' 2>&1 | tee '${LOG_FILE}'"

echo "[done] preview started"
echo "[done] attach by: tmux attach -t ${PREVIEW_TMUX_SESSION}"