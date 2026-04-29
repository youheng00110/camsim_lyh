#!/usr/bin/env bash
set -u

BASE="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh"
ENV="/inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate"
RUN_MINUTES=10
SLEEP_SECONDS=300
MEM_GB=16
MATMUL_SIZE=16384
INNER_ITERS=20

cd "$BASE"

if [ -f "$ENV" ]; then
    source "$ENV"
else
    echo "[WARN] env not found: $ENV"
    echo "[WARN] use current python: $(which python)"
fi

mkdir -p "$BASE/gpu_check_logs"

while true
do
    NOW=$(date '+%Y%m%d_%H%M%S')

    echo "===== $(date '+%Y-%m-%d %H:%M:%S') start gpu check ====="

    CUDA_VISIBLE_DEVICES=0 python "$BASE/gpu_hold_test.py" \
        --device 0 \
        --mem_gb "$MEM_GB" \
        --matmul_size "$MATMUL_SIZE" \
        --inner_iters "$INNER_ITERS" \
        --minutes "$RUN_MINUTES" \
        > /dev/null 2>&1 &

    PID0=$!

    CUDA_VISIBLE_DEVICES=1 python "$BASE/gpu_hold_test.py" \
        --device 0 \
        --mem_gb "$MEM_GB" \
        --matmul_size "$MATMUL_SIZE" \
        --inner_iters "$INNER_ITERS" \
        --minutes "$RUN_MINUTES" \
        > /dev/null 2>&1 &

    PID1=$!

    wait "$PID0"
    STATUS0=$?

    wait "$PID1"
    STATUS1=$?

    echo "gpu0 status: $STATUS0"
    echo "gpu1 status: $STATUS1"

    if [ "$STATUS0" -ne 0 ]; then
        echo "----- gpu0 error log -----"
        tail -n 40 "$BASE/gpu_check_logs/gpu0_${NOW}.log" || true
    fi

    if [ "$STATUS1" -ne 0 ]; then
        echo "----- gpu1 error log -----"
        tail -n 40 "$BASE/gpu_check_logs/gpu1_${NOW}.log" || true
    fi

    echo "===== $(date '+%Y-%m-%d %H:%M:%S') gpu check finished ====="
    echo "sleep ${SLEEP_SECONDS}s"

    sleep "$SLEEP_SECONDS"
done