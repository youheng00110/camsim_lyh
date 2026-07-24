#!/bin/bash

set -eo pipefail


# ============================================================
# 0. 环境
# ============================================================

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src


export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1


export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh


export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH
export PYTHONPATH=$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH
export PYTHONPATH=$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH


# PyTorch 权重缓存目录
export TORCH_HOME=/root/.cache/torch


# ============================================================
# 1. 本地已有权重
# ============================================================

export CKPT_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt

export LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt
export RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
export I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt


# 检查权重是否存在
test -f "$LOFTR_CHECKPOINT" || {
    echo "LoFTR checkpoint not found: $LOFTR_CHECKPOINT"
    exit 1
}

test -f "$RAFT_CHECKPOINT" || {
    echo "RAFT checkpoint not found: $RAFT_CHECKPOINT"
    exit 1
}

test -f "$I3D_CHECKPOINT" || {
    echo "I3D checkpoint not found: $I3D_CHECKPOINT"
    exit 1
}


# ============================================================
# 2. 已合并的 full 评测目录
# ============================================================

ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard/nuplanfull_preview_paired_200_merged200

MANIFEST=$ROOT/stflow_manifest.jsonl


test -f "$MANIFEST" || {
    echo "Manifest not found: $MANIFEST"
    exit 1
}


VIDEO_COUNT=$(wc -l < "$MANIFEST")

echo "================================"
echo "Evaluation root: $ROOT"
echo "Manifest videos: $VIDEO_COUNT"
echo "LoFTR checkpoint: $LOFTR_CHECKPOINT"
echo "RAFT checkpoint: $RAFT_CHECKPOINT"
echo "I3D checkpoint: $I3D_CHECKPOINT"
echo "================================"


# ============================================================
# 3. 读取视频帧数
# ============================================================

SEQ_COUNT=$(python - <<PY
import json

manifest_path = "$MANIFEST"

with open(manifest_path, "r", encoding="utf-8") as file:
    item = json.loads(file.readline())

print(len(item["frames"]))
PY
)

echo "sequence_count=$SEQ_COUNT"


# ============================================================
# 4. ST-Flow / Traj：只跑 gate16
# ============================================================

echo "================================"
echo "Run ST-Flow / Traj Gate16"
echo "================================"

python -m dwm.tools.evaluate_stflow \
    --manifest "$MANIFEST" \
    --output "$ROOT/stflow_traj_result_gate16.json" \
    --device cuda \
    --max-videos 200 \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px 16


# ============================================================
# 5. FVD
# ============================================================

echo "================================"
echo "Run FVD"
echo "================================"

python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$MANIFEST" \
    --output "$ROOT/paired_fvd_result_all${SEQ_COUNT}.json" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos 200 \
    --sequence-count "$SEQ_COUNT" \
    --batch-size 2


echo "================================"
echo "DONE"
echo "ST-Flow: $ROOT/stflow_traj_result_gate16.json"
echo "FVD: $ROOT/paired_fvd_result_all${SEQ_COUNT}.json"
echo "================================"