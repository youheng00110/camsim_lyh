source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
set -uo pipefail
cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src || exit 1


# ============================================================
# 环境
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"

export TORCH_HOME=/root/.cache/torch


# ============================================================
# 本地权重
# ============================================================

export CKPT_ROOT=$CAMSIM_ROOT/ckpt
export RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
export LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt
export I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt

for CHECKPOINT in \
    "$RAFT_CHECKPOINT" \
    "$LOFTR_CHECKPOINT" \
    "$I3D_CHECKPOINT"
do
    if [ ! -f "$CHECKPOINT" ]
    then
        echo "Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
done

mkdir -p "$TORCH_HOME/hub/checkpoints"

cp -f \
    "$RAFT_CHECKPOINT" \
    "$TORCH_HOME/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
    "$LOFTR_CHECKPOINT" \
    "$TORCH_HOME/hub/checkpoints/loftr_outdoor.ckpt"

echo "================================"
echo "Local checkpoints ready"
echo "================================"
ls -lh "$RAFT_CHECKPOINT"
ls -lh "$LOFTR_CHECKPOINT"
ls -lh "$I3D_CHECKPOINT"


# ============================================================
# 输入和输出
# ============================================================

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000

SRC=$BASE/tokenearly24000_preview_paired_200
ROOT=$BASE/tokenearly24000_preview_paired_200_merged1000

MANIFEST=$ROOT/stflow_manifest.jsonl
MAX_VIDEOS=1000


if [ ! -d "$SRC" ]
then
    echo "Source directory not found:"
    echo "$SRC"
    exit 1
fi


# ============================================================
# 检查是否已经合并
# ============================================================

NEED_MERGE=1

if [ -f "$MANIFEST" ]
then
    VIDEO_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

    echo "Existing merged videos: $VIDEO_COUNT"

    if [ "$VIDEO_COUNT" -eq "$MAX_VIDEOS" ]
    then
        NEED_MERGE=0
        echo "Merged1000 already exists. Skip merge."
    fi
fi


# ============================================================
# 合并前 1000 个
# ============================================================

if [ "$NEED_MERGE" -eq 1 ]
then
    echo
    echo "================================"
    echo "Merge tokenearly24000 to 1000 videos"
    echo "================================"

    if ! python -m dwm.tools.merge_rank_preview_manifests_interleave \
        --input-root "$SRC" \
        --output-root "$ROOT" \
        --dataset-name nuplan \
        --max-videos "$MAX_VIDEOS" \
        --overwrite
    then
        echo "Hard-link merge failed."
        echo "Retry with --copy."

        python -m dwm.tools.merge_rank_preview_manifests_interleave \
            --input-root "$SRC" \
            --output-root "$ROOT" \
            --dataset-name nuplan \
            --max-videos "$MAX_VIDEOS" \
            --overwrite \
            --copy

        MERGE_EXIT=$?

        if [ "$MERGE_EXIT" -ne 0 ]
        then
            echo "Merge failed: $MERGE_EXIT"
            exit "$MERGE_EXIT"
        fi
    fi
fi


# ============================================================
# 检查合并结果
# ============================================================

if [ ! -f "$MANIFEST" ]
then
    echo "Manifest not found:"
    echo "$MANIFEST"
    exit 1
fi

VIDEO_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

echo
echo "================================"
echo "Merged manifest check"
echo "================================"
echo "Manifest: $MANIFEST"
echo "Videos: $VIDEO_COUNT"

if [ "$VIDEO_COUNT" -ne "$MAX_VIDEOS" ]
then
    echo "Expected $MAX_VIDEOS videos, found $VIDEO_COUNT."
    exit 1
fi


SEQ_COUNT=$(python - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter

manifest_path = sys.argv[1]
frame_counts = []

with open(manifest_path, "r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        item = json.loads(line)
        frame_counts.append(len(item["frames"]))

distribution = Counter(frame_counts)

print("frame-count distribution:", dict(distribution), file=sys.stderr)

if not frame_counts:
    print("Manifest is empty.", file=sys.stderr)
    sys.exit(1)

if len(distribution) != 1:
    print("Frame counts are inconsistent.", file=sys.stderr)
    sys.exit(1)

print(frame_counts[0])
PY
)

SEQ_EXIT=$?

if [ "$SEQ_EXIT" -ne 0 ]
then
    echo "Failed to determine sequence count."
    exit "$SEQ_EXIT"
fi

echo "sequence_count=$SEQ_COUNT"


# ============================================================
# ST-Flow / Traj gate16
# 不设置 startframe
# ============================================================

STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16.json
STFLOW_LOG=$ROOT/stflow_gate16.log

echo
echo "================================"
echo "Run ST-Flow / Traj gate16"
echo "================================"

python -m dwm.tools.evaluate_stflow \
    --manifest "$MANIFEST" \
    --output "$STFLOW_OUTPUT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --frame-stride 2 \
    --min-matches 16 \
    --max-matches 256 \
    --loftr-confidence 0.1 \
    --pair-policy dataset \
    --cross-gate-px 16 \
    2>&1 | tee "$STFLOW_LOG"

STFLOW_EXIT=${PIPESTATUS[0]}

if [ "$STFLOW_EXIT" -ne 0 ]
then
    echo "ST-Flow failed: $STFLOW_EXIT"
    exit "$STFLOW_EXIT"
fi


# ============================================================
# FVD
# 不设置 startframe
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_all${SEQ_COUNT}.json
FVD_LOG=$ROOT/fvd_all${SEQ_COUNT}.log

echo
echo "================================"
echo "Run FVD"
echo "sequence_count=$SEQ_COUNT"
echo "================================"

python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$MANIFEST" \
    --output "$FVD_OUTPUT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos "$MAX_VIDEOS" \
    --sequence-count "$SEQ_COUNT" \
    --batch-size 2 \
    2>&1 | tee "$FVD_LOG"

FVD_EXIT=${PIPESTATUS[0]}

if [ "$FVD_EXIT" -ne 0 ]
then
    echo "FVD batch-size=2 failed."
    echo "Retry with batch-size=1."

    python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$MANIFEST" \
        --output "$FVD_OUTPUT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$MAX_VIDEOS" \
        --sequence-count "$SEQ_COUNT" \
        --batch-size 1 \
        2>&1 | tee -a "$FVD_LOG"

    FVD_EXIT=${PIPESTATUS[0]}

    if [ "$FVD_EXIT" -ne 0 ]
    then
        echo "FVD failed: $FVD_EXIT"
        exit "$FVD_EXIT"
    fi
fi


# ============================================================
# 读取并打印结果
# ============================================================

python - "$STFLOW_OUTPUT" "$FVD_OUTPUT" <<'PY'
import json
import math
import os
import sys

stflow_path = sys.argv[1]
fvd_path = sys.argv[2]

print()
print("=" * 80)
print("tokenearly24000 evaluation result")
print("=" * 80)

if not os.path.isfile(stflow_path):
    print("ST-Flow result missing:", stflow_path)
    sys.exit(1)

if not os.path.isfile(fvd_path):
    print("FVD result missing:", fvd_path)
    sys.exit(1)

with open(stflow_path, "r", encoding="utf-8") as file:
    stflow_data = json.load(file)

with open(fvd_path, "r", encoding="utf-8") as file:
    fvd_data = json.load(file)

mean = stflow_data.get("mean", {})
fvd = fvd_data.get("fvd")

stflow_score = mean.get("stflow_score")
stflow_d_score = mean.get("stflow_d_score")
temporal_l1 = mean.get("temporal_l1")
cross_raw_epi = mean.get("cross_raw_epi_px")
cross_epi = mean.get("cross_epi_px")
cross_inlier = mean.get("cross_inlier_ratio")
traj_epi = mean.get("traj_epi_px")
traj_inlier2 = mean.get("traj_inlier2")
traj_inlier4 = mean.get("traj_inlier4")

raw_edges = mean.get("num_cross_raw_edges")
gated_edges = mean.get("num_cross_edges")

stflow_c = None

if (
    isinstance(stflow_score, (int, float))
    and isinstance(cross_inlier, (int, float))
    and isinstance(raw_edges, (int, float))
    and isinstance(gated_edges, (int, float))
    and raw_edges > 0
):
    edge_coverage = gated_edges / raw_edges
    stflow_c = stflow_score * edge_coverage * cross_inlier

print("ST-Flow file:", stflow_path)
print("FVD file:    ", fvd_path)
print()
print("ST-Flow:          ", stflow_score)
print("ST-Flow-D:        ", stflow_d_score)
print("ST-Flow-C:        ", stflow_c)
print("Temporal-L1:      ", temporal_l1)
print("Cross-Raw-Epi:    ", cross_raw_epi)
print("Cross-Epi:        ", cross_epi)
print("Cross-Inlier:     ", cross_inlier)
print("Traj-Epi:         ", traj_epi)
print("Traj-Inlier@2:    ", traj_inlier2)
print("Traj-Inlier@4:    ", traj_inlier4)
print("FVD:              ", fvd)
print("=" * 80)

if not isinstance(fvd, (int, float)) or not math.isfinite(float(fvd)):
    print("Invalid FVD value.")
    sys.exit(1)
PY


echo
echo "================================"
echo "DONE"
echo "================================"
echo "Merged root:"
echo "$ROOT"
echo
echo "ST-Flow:"
echo "$STFLOW_OUTPUT"
echo
echo "FVD:"
echo "$FVD_OUTPUT"