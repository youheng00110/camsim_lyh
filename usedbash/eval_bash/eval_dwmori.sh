source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

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

mkdir -p "$TORCH_HOME/hub/checkpoints"

cp -f \
  "$RAFT_CHECKPOINT" \
  "$TORCH_HOME/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

cp -f \
  "$LOFTR_CHECKPOINT" \
  "$TORCH_HOME/hub/checkpoints/loftr_outdoor.ckpt"

echo "===== CHECKPOINTS ====="
ls -lh "$RAFT_CHECKPOINT"
ls -lh "$LOFTR_CHECKPOINT"
ls -lh "$I3D_CHECKPOINT"


# ============================================================
# 路径
# ============================================================

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

SRC=$BASE/dwmori
ROOT=$BASE/dwmori_merged1000

MANIFEST=$ROOT/stflow_manifest.jsonl


# ============================================================
# 合并前1000
# ============================================================

echo
echo "============================================================"
echo "MERGE dwmori -> dwmori_merged1000"
echo "============================================================"

python -m dwm.tools.merge_rank_preview_manifests_interleave \
  --input-root "$SRC" \
  --output-root "$ROOT" \
  --dataset-name nuplan \
  --max-videos 1000 \
  --overwrite

MERGE_EXIT=$?

if [ "$MERGE_EXIT" -ne 0 ]; then
  echo "Hard link failed, retry with --copy"

  python -m dwm.tools.merge_rank_preview_manifests_interleave \
    --input-root "$SRC" \
    --output-root "$ROOT" \
    --dataset-name nuplan \
    --max-videos 1000 \
    --overwrite \
    --copy

  if [ "$?" -ne 0 ]; then
    echo "MERGE FAILED"
    exit 1
  fi
fi


# ============================================================
# 检查合并
# ============================================================

echo
echo "============================================================"
echo "CHECK MERGED MANIFEST"
echo "============================================================"

python - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter

manifest = sys.argv[1]

items = []

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))

print("videos:", len(items))

if len(items) != 1000:
    raise RuntimeError(
        f"Expected 1000 videos, got {len(items)}"
    )

first = items[0]

print("video_id:", first.get("video_id"))
print("dataset:", first.get("dataset_name"))
print("frames:", len(first["frames"]))
print("views:", len(first["frames"][0]["views"]))
print(
    "cameras:",
    [v["camera"] for v in first["frames"][0]["views"]],
)

frame_counts = Counter(
    len(item["frames"])
    for item in items
)

print("frame count distribution:", dict(frame_counts))

if len(frame_counts) != 1:
    raise RuntimeError(
        f"Inconsistent frame counts: {dict(frame_counts)}"
    )
PY

if [ "$?" -ne 0 ]; then
  echo "MANIFEST CHECK FAILED"
  exit 1
fi


# ============================================================
# ST-Flow / Traj gate16
# 完整 nuPlan 8 cameras
# 不额外设置 startframe
# ============================================================

echo
echo "============================================================"
echo "RUN ST-FLOW / TRAJ GATE16"
echo "============================================================"

python -m dwm.tools.evaluate_stflow \
  --manifest "$MANIFEST" \
  --output "$ROOT/stflow_traj_result_gate16.json" \
  --device cuda \
  --max-videos 1000 \
  --frame-stride 2 \
  --min-matches 16 \
  --max-matches 256 \
  --loftr-confidence 0.1 \
  --pair-policy dataset \
  --cross-gate-px 16 \
  2>&1 | tee "$ROOT/stflow_gate16.log"

STFLOW_EXIT=${PIPESTATUS[0]}

if [ "$STFLOW_EXIT" -ne 0 ]; then
  echo "ST-FLOW FAILED: $STFLOW_EXIT"
  exit "$STFLOW_EXIT"
fi


# ============================================================
# 获取完整 sequence count
# ============================================================

SEQ_COUNT=$(python - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter

manifest = sys.argv[1]
counts = []

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            counts.append(len(item["frames"]))

dist = Counter(counts)

if len(dist) != 1:
    raise RuntimeError(
        f"Inconsistent sequence lengths: {dict(dist)}"
    )

print(counts[0])
PY
)

echo
echo "sequence_count=$SEQ_COUNT"


# ============================================================
# FVD
# ============================================================

FVD_OUTPUT=$ROOT/paired_fvd_result_all${SEQ_COUNT}.json

echo
echo "============================================================"
echo "RUN FVD"
echo "sequence_count=$SEQ_COUNT"
echo "============================================================"

python -m dwm.tools.evaluate_fvd_from_paired_manifest \
  --manifest "$MANIFEST" \
  --output "$FVD_OUTPUT" \
  --i3d-checkpoint "$I3D_CHECKPOINT" \
  --device cuda \
  --max-videos 1000 \
  --sequence-count "$SEQ_COUNT" \
  --batch-size 2 \
  2>&1 | tee "$ROOT/fvd_all${SEQ_COUNT}.log"

FVD_EXIT=${PIPESTATUS[0]}

if [ "$FVD_EXIT" -ne 0 ]; then
  echo "FVD batch-size=2 failed, retry batch-size=1"

  python -m dwm.tools.evaluate_fvd_from_paired_manifest \
    --manifest "$MANIFEST" \
    --output "$FVD_OUTPUT" \
    --i3d-checkpoint "$I3D_CHECKPOINT" \
    --device cuda \
    --max-videos 1000 \
    --sequence-count "$SEQ_COUNT" \
    --batch-size 1 \
    2>&1 | tee -a "$ROOT/fvd_all${SEQ_COUNT}.log"

  FVD_EXIT=${PIPESTATUS[0]}

  if [ "$FVD_EXIT" -ne 0 ]; then
    echo "FVD FAILED: $FVD_EXIT"
    exit "$FVD_EXIT"
  fi
fi


# ============================================================
# 最后检查结果
# ============================================================

echo
echo "============================================================"
echo "RESULT CHECK"
echo "============================================================"

python - "$ROOT/stflow_traj_result_gate16.json" "$FVD_OUTPUT" <<'PY'
import json
import sys

st_path = sys.argv[1]
fvd_path = sys.argv[2]

with open(st_path, "r", encoding="utf-8") as f:
    st = json.load(f)

with open(fvd_path, "r", encoding="utf-8") as f:
    fv = json.load(f)

mean = st.get("mean", {})

print("Temporal-L1      :", mean.get("temporal_l1"))
print("Cross-Raw-Epi   :", mean.get("cross_raw_epi_px"))
print("Cross-Inlier    :", mean.get("cross_inlier_ratio"))
print("Traj-Epi        :", mean.get("traj_epi_px"))
print("Traj-Inlier@2   :", mean.get("traj_inlier2"))
print("ST-Flow-D       :", mean.get("stflow_d_score"))
print("FVD             :", fv.get("fvd"))
PY


echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "ROOT:"
echo "$ROOT"
echo
echo "ST-Flow:"
echo "$ROOT/stflow_traj_result_gate16.json"
echo
echo "FVD:"
echo "$FVD_OUTPUT"