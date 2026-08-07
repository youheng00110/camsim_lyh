#!/bin/bash

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

set -uo pipefail


# ============================================================
# 0. 环境
# ============================================================

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src || exit 1

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"


# ============================================================
# 1. 本地权重
# ============================================================

export TORCH_HOME=/root/.cache/torch

CKPT_ROOT=$CAMSIM_ROOT/ckpt
CACHE_DIR=$TORCH_HOME/hub/checkpoints

export RAFT_CHECKPOINT=$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth
export LOFTR_CHECKPOINT=$CKPT_ROOT/loftr_outdoor.ckpt
export I3D_CHECKPOINT=$CKPT_ROOT/i3d_pretrained_400.pt

RAFT_CACHE=$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth
LOFTR_CACHE=$CACHE_DIR/loftr_outdoor.ckpt


for CHECKPOINT in \
    "$RAFT_CHECKPOINT" \
    "$LOFTR_CHECKPOINT" \
    "$I3D_CHECKPOINT"
do
    if [ ! -f "$CHECKPOINT" ]
    then
        echo "Checkpoint not found:"
        echo "$CHECKPOINT"
        exit 1
    fi
done


mkdir -p "$CACHE_DIR"

cp -f "$RAFT_CHECKPOINT" "$RAFT_CACHE"
cp -f "$LOFTR_CHECKPOINT" "$LOFTR_CACHE"

echo "============================================================"
echo "Local checkpoints"
echo "============================================================"

ls -lh "$RAFT_CACHE"
ls -lh "$LOFTR_CACHE"
ls -lh "$I3D_CHECKPOINT"


# ============================================================
# 2. 离线验证 RAFT / LoFTR
# ============================================================

echo
echo "============================================================"
echo "Verify local RAFT and LoFTR checkpoints"
echo "============================================================"

CUDA_VISIBLE_DEVICES="" python - <<'PY'
from kornia.feature import LoFTR
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    raft_large,
)

print("Loading RAFT from local Torch Hub cache...")

raft_model = raft_large(
    weights=Raft_Large_Weights.DEFAULT,
    progress=False,
)

print("Loading LoFTR outdoor from local Torch Hub cache...")

loftr_model = LoFTR(
    pretrained="outdoor",
)

print("Local checkpoints loaded successfully.")

del raft_model
del loftr_model
PY

VERIFY_EXIT=$?

if [ "$VERIFY_EXIT" -ne 0 ]
then
    echo "Local checkpoint verification failed."
    exit "$VERIFY_EXIT"
fi


# ============================================================
# 3. 只配置两个 nuScenes 实验
# ============================================================

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

MAX_VIDEOS=1000
GATE=16

NAMES=(
    nuscenesablationori
    nuscenesablationori3
)

# nuscenesablationori：
#   实际 6 摄，但填充到 8 个槽位
#   0 = Front Left
#   2 = Front
#   4 = Front Right
#
# nuscenesablationori3：
#   只有 3 摄
#   0 = Front Left
#   1 = Front
#   2 = Front Right
VIEW_INDEX_SETS=(
    "0,2,4"
    "0,1,2"
)

CANONICAL_CAMERAS="CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT"

CAMERA_PAIRS="CAM_FRONT_LEFT__CAM_FRONT,CAM_FRONT__CAM_FRONT_RIGHT"


if [ ! -d "$BASE" ]
then
    echo "Base directory not found:"
    echo "$BASE"
    exit 1
fi


echo
echo "============================================================"
echo "Configured nuScenes experiments"
echo "============================================================"

for INDEX in "${!NAMES[@]}"
do
    echo "[$INDEX] GPU $INDEX"
    echo "  method: ${NAMES[$INDEX]}"
    echo "  selected view indices: ${VIEW_INDEX_SETS[$INDEX]}"
done


# ============================================================
# 4. 合并两个实验
#
# 只处理：
#   nuscenesablationori
#   nuscenesablationori3
#
# 不会处理 dwmori。
# ============================================================

ROOTS=()

for INDEX in "${!NAMES[@]}"
do
    NAME=${NAMES[$INDEX]}
    INDEX_SET=${VIEW_INDEX_SETS[$INDEX]}

    SRC=$BASE/$NAME
    ROOT=$BASE/${NAME}_merged1000

    MANIFEST=$ROOT/stflow_manifest.jsonl
    FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl
    FRONT_CONFIG=$ROOT/front3_eval_config.json

    ROOTS+=("$ROOT")


    echo
    echo "============================================================"
    echo "Process experiment"
    echo "NAME: $NAME"
    echo "SRC: $SRC"
    echo "ROOT: $ROOT"
    echo "VIEW INDICES: $INDEX_SET"
    echo "============================================================"


    if [ ! -d "$SRC" ]
    then
        echo "Source directory not found:"
        echo "$SRC"
        exit 1
    fi


    # --------------------------------------------------------
    # 统计 rank 输出总数，最多取 1000
    # --------------------------------------------------------

    EXPECTED_COUNT=$(python - "$SRC" "$MAX_VIDEOS" <<'PY'
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
max_videos = int(sys.argv[2])

count = 0

rank_dirs = sorted(
    path
    for path in source_root.iterdir()
    if path.is_dir() and path.name.startswith("rank_")
)

for rank_dir in rank_dirs:
    manifest = rank_dir / "stflow_manifest.jsonl"

    if not manifest.is_file():
        continue

    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                json.loads(line)
                count += 1

print(min(count, max_videos))
PY
    )

    COUNT_EXIT=$?

    if [ "$COUNT_EXIT" -ne 0 ] || [ "$EXPECTED_COUNT" -le 0 ]
    then
        echo "No valid rank manifest items found:"
        echo "$SRC"
        exit 1
    fi

    echo "Expected merged videos: $EXPECTED_COUNT"


    # --------------------------------------------------------
    # 检查已有合并结果
    # --------------------------------------------------------

    MERGED_VALID=0

    if [ -f "$MANIFEST" ]
    then
        EXISTING_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

        if [ "$EXISTING_COUNT" -eq "$EXPECTED_COUNT" ]
        then
            MERGED_VALID=1
            echo "Existing merged manifest is valid."
            echo "Videos: $EXISTING_COUNT"
        else
            echo "Existing merged count mismatch:"
            echo "existing=$EXISTING_COUNT"
            echo "expected=$EXPECTED_COUNT"
        fi
    fi


    # --------------------------------------------------------
    # 不存在或数量不符时重新合并
    # --------------------------------------------------------

    if [ "$MERGED_VALID" -eq 0 ]
    then
        echo "Merge rank outputs..."

        python -m dwm.tools.merge_rank_preview_manifests_interleave \
            --input-root "$SRC" \
            --output-root "$ROOT" \
            --dataset-name nuscenes \
            --max-videos "$MAX_VIDEOS" \
            --overwrite

        MERGE_EXIT=$?

        if [ "$MERGE_EXIT" -ne 0 ]
        then
            echo "Default merge failed."
            echo "Retry with --copy."

            python -m dwm.tools.merge_rank_preview_manifests_interleave \
                --input-root "$SRC" \
                --output-root "$ROOT" \
                --dataset-name nuscenes \
                --max-videos "$MAX_VIDEOS" \
                --overwrite \
                --copy

            MERGE_EXIT=$?

            if [ "$MERGE_EXIT" -ne 0 ]
            then
                echo "Merge failed:"
                echo "$NAME"
                exit "$MERGE_EXIT"
            fi
        fi
    else
        echo "Skip rank merge."
    fi


    if [ ! -f "$MANIFEST" ]
    then
        echo "Merged manifest not found:"
        echo "$MANIFEST"
        exit 1
    fi


    MERGED_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

    if [ "$MERGED_COUNT" -ne "$EXPECTED_COUNT" ]
    then
        echo "Merged count is incorrect."
        echo "expected=$EXPECTED_COUNT"
        echo "actual=$MERGED_COUNT"
        exit 1
    fi


    # --------------------------------------------------------
    # 生成只包含三个前视视角的 manifest
    #
    # 按视角位置选取，不依赖原始 camera 字段。
    # 选出后统一重命名为：
    #   CAM_FRONT_LEFT
    #   CAM_FRONT
    #   CAM_FRONT_RIGHT
    # --------------------------------------------------------

    python - \
        "$MANIFEST" \
        "$FRONT_MANIFEST" \
        "$FRONT_CONFIG" \
        "$INDEX_SET" <<'PY'
import copy
import json
import sys
from collections import Counter


manifest_path = sys.argv[1]
front_manifest_path = sys.argv[2]
config_path = sys.argv[3]

selected_indices = [
    int(value)
    for value in sys.argv[4].split(",")
]

canonical_names = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
]

if len(selected_indices) != 3:
    print(
        f"Expected 3 selected indices, got {selected_indices}",
        file=sys.stderr,
    )
    sys.exit(1)


items = []

with open(manifest_path, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as error:
            print(
                f"Invalid JSON at line {line_number}: {error}",
                file=sys.stderr,
            )
            sys.exit(1)


if not items:
    print("Merged manifest is empty.", file=sys.stderr)
    sys.exit(1)


filtered_items = []
frame_count_distribution = Counter()
original_view_count_distribution = Counter()
filtered_view_count_distribution = Counter()
first_original_cameras = None


for item_index, item in enumerate(items):
    filtered_item = copy.deepcopy(item)
    frames = filtered_item.get("frames", [])

    if not frames:
        print(
            f"Video {item_index} has no frames.",
            file=sys.stderr,
        )
        sys.exit(1)

    frame_count_distribution[len(frames)] += 1


    for frame_index, frame in enumerate(frames):
        views = frame.get("views", [])
        original_view_count_distribution[len(views)] += 1

        if first_original_cameras is None:
            first_original_cameras = [
                view.get("camera")
                for view in views
            ]

        maximum_index = max(selected_indices)

        if len(views) <= maximum_index:
            print(
                f"Not enough views at video={item_index}, "
                f"frame={frame_index}. "
                f"Need index {maximum_index}, "
                f"but only have {len(views)} views.",
                file=sys.stderr,
            )
            print(
                "Available cameras:",
                [view.get("camera") for view in views],
                file=sys.stderr,
            )
            sys.exit(1)


        selected_views = []

        for output_index, source_index in enumerate(selected_indices):
            selected_view = copy.deepcopy(
                views[source_index]
            )

            selected_view["source_camera_name"] = (
                selected_view.get("camera")
            )
            selected_view["source_view_index"] = source_index
            selected_view["camera"] = canonical_names[output_index]

            selected_views.append(selected_view)


        frame["views"] = selected_views
        filtered_view_count_distribution[len(selected_views)] += 1


    filtered_items.append(filtered_item)


with open(front_manifest_path, "w", encoding="utf-8") as file:
    for item in filtered_items:
        file.write(
            json.dumps(item, ensure_ascii=False)
            + "\n"
        )


config = {
    "source_manifest": manifest_path,
    "filtered_manifest": front_manifest_path,
    "num_videos": len(filtered_items),
    "selected_view_indices": selected_indices,
    "canonical_camera_names": canonical_names,
    "camera_names_csv": ",".join(canonical_names),
    "camera_pairs": [
        "CAM_FRONT_LEFT__CAM_FRONT",
        "CAM_FRONT__CAM_FRONT_RIGHT",
    ],
    "camera_pairs_csv": (
        "CAM_FRONT_LEFT__CAM_FRONT,"
        "CAM_FRONT__CAM_FRONT_RIGHT"
    ),
    "first_original_cameras": first_original_cameras,
    "frame_count_distribution": {
        str(key): value
        for key, value in sorted(
            frame_count_distribution.items()
        )
    },
    "original_view_count_distribution": {
        str(key): value
        for key, value in sorted(
            original_view_count_distribution.items()
        )
    },
    "filtered_view_count_distribution": {
        str(key): value
        for key, value in sorted(
            filtered_view_count_distribution.items()
        )
    },
}


with open(config_path, "w", encoding="utf-8") as file:
    json.dump(
        config,
        file,
        indent=2,
        ensure_ascii=False,
    )


print("=" * 80)
print("Front3 manifest created")
print("source:", manifest_path)
print("output:", front_manifest_path)
print("videos:", len(filtered_items))
print("selected indices:", selected_indices)
print("original cameras:", first_original_cameras)
print("canonical cameras:", canonical_names)
print("frame counts:", dict(frame_count_distribution))
print("original view counts:", dict(original_view_count_distribution))
print("filtered view counts:", dict(filtered_view_count_distribution))
print("=" * 80)
PY

    FILTER_EXIT=$?

    if [ "$FILTER_EXIT" -ne 0 ]
    then
        echo "Front3 manifest generation failed:"
        echo "$NAME"
        exit "$FILTER_EXIT"
    fi

    echo
    echo "Front3 configuration:"
    cat "$FRONT_CONFIG"
done


# ============================================================
# 5. 两张卡并行评测
#
# GPU 0：nuscenesablationori
# GPU 1：nuscenesablationori3
#
# 不设置 start-frame。
# ============================================================

echo
echo "============================================================"
echo "Run Front3 evaluations on two GPUs"
echo "============================================================"

PIDS=()


for INDEX in "${!ROOTS[@]}"
do
(
    set -euo pipefail

    GPU=$INDEX
    export CUDA_VISIBLE_DEVICES="$GPU"

    NAME=${NAMES[$INDEX]}
    ROOT=${ROOTS[$INDEX]}

    FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl
    FRONT_CONFIG=$ROOT/front3_eval_config.json

    LOG_DIR=$ROOT/eval_logs_front3
    mkdir -p "$LOG_DIR"


    VIDEO_COUNT=$(python - "$FRONT_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    config = json.load(file)

print(config["num_videos"])
PY
    )


    SEQ_COUNT=$(python - "$FRONT_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    config = json.load(file)

distribution = config["frame_count_distribution"]

if len(distribution) != 1:
    print(
        f"Inconsistent frame counts: {distribution}",
        file=sys.stderr,
    )
    sys.exit(1)

print(next(iter(distribution.keys())))
PY
    )


    CAMERA_NAMES=$(python - "$FRONT_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    config = json.load(file)

print(config["camera_names_csv"])
PY
    )


    CAMERA_PAIRS=$(python - "$FRONT_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    config = json.load(file)

print(config["camera_pairs_csv"])
PY
    )


    echo
    echo "============================================================"
    echo "[GPU $GPU] Front3 evaluation"
    echo "METHOD: $NAME"
    echo "VIDEOS: $VIDEO_COUNT"
    echo "SEQUENCE COUNT: $SEQ_COUNT"
    echo "CAMERAS: $CAMERA_NAMES"
    echo "CAMERA PAIRS: $CAMERA_PAIRS"
    echo "============================================================"


    # --------------------------------------------------------
    # ST-Flow / Traj
    #
    # filtered manifest 中只有三个前视相机。
    # Temporal 和 Traj 只会计算这三个视角。
    #
    # Cross-view 只计算：
    #   Front Left -> Front
    #   Front -> Front Right
    # --------------------------------------------------------

    STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16_front3.json
    STFLOW_LOG=$LOG_DIR/stflow_gate16_front3.log


    python -m dwm.tools.evaluate_stflow \
        --manifest "$FRONT_MANIFEST" \
        --output "$STFLOW_OUTPUT" \
        --device cuda \
        --max-videos "$VIDEO_COUNT" \
        --frame-stride 2 \
        --min-matches 16 \
        --max-matches 256 \
        --loftr-confidence 0.1 \
        --camera-pairs "$CAMERA_PAIRS" \
        --pair-policy ring \
        --cross-gate-px "$GATE" \
        2>&1 | tee "$STFLOW_LOG"


    # --------------------------------------------------------
    # FVD
    # --------------------------------------------------------

    FVD_OUTPUT=$ROOT/paired_fvd_result_front3_all${SEQ_COUNT}.json
    FVD_LOG=$LOG_DIR/fvd_front3_all${SEQ_COUNT}.log


    if ! python -m dwm.tools.evaluate_fvd_from_paired_manifest \
        --manifest "$FRONT_MANIFEST" \
        --output "$FVD_OUTPUT" \
        --i3d-checkpoint "$I3D_CHECKPOINT" \
        --device cuda \
        --max-videos "$VIDEO_COUNT" \
        --sequence-count "$SEQ_COUNT" \
        --camera-names "$CAMERA_NAMES" \
        --batch-size 2 \
        2>&1 | tee "$FVD_LOG"
    then
        echo "[GPU $GPU] FVD batch-size=2 failed."
        echo "[GPU $GPU] Retry with batch-size=1."

        python -m dwm.tools.evaluate_fvd_from_paired_manifest \
            --manifest "$FRONT_MANIFEST" \
            --output "$FVD_OUTPUT" \
            --i3d-checkpoint "$I3D_CHECKPOINT" \
            --device cuda \
            --max-videos "$VIDEO_COUNT" \
            --sequence-count "$SEQ_COUNT" \
            --camera-names "$CAMERA_NAMES" \
            --batch-size 1 \
            2>&1 | tee -a "$FVD_LOG"
    fi


    echo "[GPU $GPU] Completed: $NAME"
) &

    PIDS+=("$!")
done


# ============================================================
# 6. 等待两个任务结束
# ============================================================

STATUS=0

for PID in "${PIDS[@]}"
do
    if ! wait "$PID"
    then
        echo "Evaluation worker failed: PID=$PID"
        STATUS=1
    fi
done


if [ "$STATUS" -ne 0 ]
then
    echo "At least one evaluation failed."
    exit 1
fi


# ============================================================
# 7. 汇总结果
# ============================================================

echo
echo "============================================================"
echo "Front3 evaluation summary"
echo "============================================================"


python - "$BASE" "${ROOTS[@]}" <<'PY'
import glob
import json
import math
import os
import sys


base = sys.argv[1]
roots = sys.argv[2:]

summary = {}
failed = False


print()
print(
    "| Method | Indices | Videos | Temporal-L1 ↓ | "
    "Cross-Raw-Epi ↓ | Cross-Inlier ↑ | "
    "Traj-Epi ↓ | Traj-Inlier@2 ↑ | "
    "ST-Flow-D ↑ | ST-Flow-C ↑ | FVD ↓ | Status |"
)
print(
    "|---|---|---:|---:|---:|---:|---:|---:|"
    "---:|---:|---:|---|"
)


for root in roots:
    method = os.path.basename(root).replace(
        "_merged1000",
        "",
    )

    config_path = os.path.join(
        root,
        "front3_eval_config.json",
    )
    stflow_path = os.path.join(
        root,
        "stflow_traj_result_gate16_front3.json",
    )

    fvd_candidates = sorted(
        glob.glob(
            os.path.join(
                root,
                "paired_fvd_result_front3_all*.json",
            )
        ),
        key=os.path.getmtime,
    )


    status = "OK"

    config = {}
    temporal = None
    cross_raw = None
    cross_inlier = None
    traj = None
    traj_inlier2 = None
    stflow_d = None
    stflow_c = None
    fvd = None


    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        with open(stflow_path, "r", encoding="utf-8") as file:
            stflow_data = json.load(file)

        if not fvd_candidates:
            raise FileNotFoundError(
                "Front3 FVD result is missing."
            )

        with open(
            fvd_candidates[-1],
            "r",
            encoding="utf-8",
        ) as file:
            fvd_data = json.load(file)


        mean = stflow_data.get("mean", {})

        temporal = mean.get("temporal_l1")
        cross_raw = mean.get("cross_raw_epi_px")
        cross_inlier = mean.get("cross_inlier_ratio")
        traj = mean.get("traj_epi_px")
        traj_inlier2 = mean.get("traj_inlier2")
        stflow_d = mean.get("stflow_d_score")
        stflow_c = mean.get("stflow_c_score")
        fvd = fvd_data.get("fvd")


        required = [
            temporal,
            cross_raw,
            cross_inlier,
            traj,
            traj_inlier2,
            stflow_d,
            stflow_c,
            fvd,
        ]

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in required
        ):
            status = "INVALID_VALUE"
            failed = True


    except Exception as error:
        status = f"ERROR: {error}"
        failed = True


    def format_value(value, digits):
        if not isinstance(value, (int, float)):
            return "-"

        if not math.isfinite(float(value)):
            return "-"

        return f"{float(value):.{digits}f}"


    indices = ",".join(
        str(value)
        for value in config.get(
            "selected_view_indices",
            [],
        )
    )


    print(
        f"| {method} "
        f"| {indices} "
        f"| {config.get('num_videos', 0)} "
        f"| {format_value(temporal, 6)} "
        f"| {format_value(cross_raw, 3)} "
        f"| {format_value(cross_inlier, 4)} "
        f"| {format_value(traj, 3)} "
        f"| {format_value(traj_inlier2, 4)} "
        f"| {format_value(stflow_d, 3)} "
        f"| {format_value(stflow_c, 3)} "
        f"| {format_value(fvd, 3)} "
        f"| {status} |"
    )


    summary[method] = {
        "root": root,
        "selected_view_indices": config.get(
            "selected_view_indices",
            [],
        ),
        "canonical_camera_names": config.get(
            "canonical_camera_names",
            [],
        ),
        "num_videos": config.get(
            "num_videos",
            0,
        ),
        "temporal_l1": temporal,
        "cross_raw_epi_px": cross_raw,
        "cross_inlier_ratio": cross_inlier,
        "traj_epi_px": traj,
        "traj_inlier2": traj_inlier2,
        "stflow_d_score": stflow_d,
        "stflow_c_score": stflow_c,
        "fvd": fvd,
        "status": status,
    }


summary_path = os.path.join(
    base,
    "front3_evaluation_summary.json",
)

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("Summary:")
print(summary_path)


if failed:
    sys.exit(1)
PY

SUMMARY_EXIT=$?


if [ "$SUMMARY_EXIT" -ne 0 ]
then
    echo "Front3 result validation failed."
    exit "$SUMMARY_EXIT"
fi


echo
echo "============================================================"
echo "All nuScenes Front3 evaluations completed"
echo "============================================================"

echo "Only evaluated:"
echo "  $BASE/nuscenesablationori"
echo "  $BASE/nuscenesablationori3"

echo
echo "Summary:"
echo "$BASE/front3_evaluation_summary.json"
