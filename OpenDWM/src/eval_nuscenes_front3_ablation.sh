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
        echo "Checkpoint not found: $CHECKPOINT"
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
# 2. 数据配置
# ============================================================

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation

MAX_VIDEOS=1000
GATE=16

if [ ! -d "$BASE" ]
then
    echo "Base directory not found:"
    echo "$BASE"
    exit 1
fi


# ============================================================
# 3. 自动发现包含 rank_* 的原始实验目录
#
# 会忽略：
#   *_merged1000
#   eval_statistics
#   其他不包含 rank_* 的目录
# ============================================================

SOURCES=()

while IFS= read -r DIRECTORY
do
    if compgen -G "$DIRECTORY/rank_*" > /dev/null
    then
        SOURCES+=("$DIRECTORY")
    fi
done < <(
    find "$BASE" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        | sort
)

if [ "${#SOURCES[@]}" -eq 0 ]
then
    echo "No source directories containing rank_* were found."
    echo "BASE=$BASE"
    exit 1
fi

echo
echo "============================================================"
echo "Discovered source experiments"
echo "============================================================"

for INDEX in "${!SOURCES[@]}"
do
    GPU=$((INDEX % 2))
    echo "[$INDEX] GPU $GPU -> ${SOURCES[$INDEX]}"
done

if [ "${#SOURCES[@]}" -ne 2 ]
then
    echo
    echo "Warning: expected 2 experiments, found ${#SOURCES[@]}."
    echo "The script will evaluate every discovered source directory."
fi


# ============================================================
# 4. 合并 rank 输出，并生成仅含 Front3 的 manifest
#
# Front3 选择规则：
#
# A. 存在语义名称时：
#    CAM_FRONT_LEFT
#    CAM_FRONT
#    CAM_FRONT_RIGHT
#
# B. 填充到 8 视角时：
#    CAM_00 = Front Left
#    CAM_02 = Front
#    CAM_04 = Front Right
#
# C. 真正只有 3 视角时：
#    直接使用 manifest 中全部三个视角
#
# Cross-view 只建立：
#    Front Left -> Front
#    Front -> Front Right
#
# 不建立 Front Right -> Front Left。
# ============================================================

ROOTS=()

for SRC in "${SOURCES[@]}"
do
    NAME=$(basename "$SRC")
    ROOT=$BASE/${NAME}_merged1000

    MANIFEST=$ROOT/stflow_manifest.jsonl
    FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl
    FRONT_CONFIG=$ROOT/front3_eval_config.json

    ROOTS+=("$ROOT")

    echo
    echo "============================================================"
    echo "Merge experiment"
    echo "NAME: $NAME"
    echo "SRC:  $SRC"
    echo "ROOT: $ROOT"
    echo "============================================================"

    if ! python -m dwm.tools.merge_rank_preview_manifests_interleave \
        --input-root "$SRC" \
        --output-root "$ROOT" \
        --dataset-name nuscenes \
        --max-videos "$MAX_VIDEOS" \
        --overwrite
    then
        echo "Default merge failed. Retry with --copy."

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
            echo "Merge failed: $NAME"
            exit "$MERGE_EXIT"
        fi
    fi

    if [ ! -f "$MANIFEST" ]
    then
        echo "Merged manifest not found:"
        echo "$MANIFEST"
        exit 1
    fi

    python - \
        "$MANIFEST" \
        "$FRONT_MANIFEST" \
        "$FRONT_CONFIG" <<'PY'
import json
import os
import sys
from collections import Counter


manifest_path = sys.argv[1]
front_manifest_path = sys.argv[2]
config_path = sys.argv[3]

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

first_frames = items[0].get("frames", [])

if not first_frames:
    print("First video contains no frames.", file=sys.stderr)
    sys.exit(1)

first_views = first_frames[0].get("views", [])
camera_names = [
    str(view.get("camera"))
    for view in first_views
]

semantic_front3 = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
]

generic_padded_front3 = [
    "CAM_00",
    "CAM_02",
    "CAM_04",
]

generic_three_front3 = [
    "CAM_00",
    "CAM_01",
    "CAM_02",
]

if all(name in camera_names for name in semantic_front3):
    selected_cameras = semantic_front3
    selection_mode = "semantic_front3"

elif all(name in camera_names for name in generic_padded_front3):
    selected_cameras = generic_padded_front3
    selection_mode = "generic_padded_00_02_04"

elif (
    len(camera_names) == 3
    and all(name in camera_names for name in generic_three_front3)
):
    selected_cameras = generic_three_front3
    selection_mode = "generic_three_00_01_02"

elif len(camera_names) == 3:
    selected_cameras = camera_names
    selection_mode = "exactly_three_manifest_views"

else:
    print(
        "Cannot determine Front3 cameras.",
        file=sys.stderr,
    )
    print(
        f"Available cameras: {camera_names}",
        file=sys.stderr,
    )
    sys.exit(1)

camera_pairs = [
    f"{selected_cameras[0]}__{selected_cameras[1]}",
    f"{selected_cameras[1]}__{selected_cameras[2]}",
]

frame_count_distribution = Counter()
view_count_distribution = Counter()

for item_index, item in enumerate(items):
    frames = item.get("frames", [])
    frame_count_distribution[len(frames)] += 1

    for frame_index, frame in enumerate(frames):
        views = frame.get("views", [])
        view_map = {
            str(view.get("camera")): view
            for view in views
        }

        missing = [
            name
            for name in selected_cameras
            if name not in view_map
        ]

        if missing:
            print(
                f"Missing cameras at item={item_index}, "
                f"frame={frame_index}: {missing}",
                file=sys.stderr,
            )
            print(
                f"Available: {list(view_map.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        frame["views"] = [
            view_map[name]
            for name in selected_cameras
        ]

        view_count_distribution[len(frame["views"])] += 1

with open(front_manifest_path, "w", encoding="utf-8") as file:
    for item in items:
        file.write(
            json.dumps(item, ensure_ascii=False)
            + "\n"
        )

config = {
    "source_manifest": manifest_path,
    "filtered_manifest": front_manifest_path,
    "num_videos": len(items),
    "selected_cameras": selected_cameras,
    "camera_names_csv": ",".join(selected_cameras),
    "camera_pairs": camera_pairs,
    "camera_pairs_csv": ",".join(camera_pairs),
    "selection_mode": selection_mode,
    "original_first_frame_cameras": camera_names,
    "frame_count_distribution": {
        str(key): value
        for key, value in frame_count_distribution.items()
    },
    "filtered_view_count_distribution": {
        str(key): value
        for key, value in view_count_distribution.items()
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
print("selection mode:", selection_mode)
print("original cameras:", camera_names)
print("selected cameras:", selected_cameras)
print("camera pairs:", camera_pairs)
print("videos:", len(items))
print("output:", front_manifest_path)
print("=" * 80)
PY

    FILTER_EXIT=$?

    if [ "$FILTER_EXIT" -ne 0 ]
    then
        echo "Front3 manifest generation failed: $NAME"
        exit "$FILTER_EXIT"
    fi

    cat "$FRONT_CONFIG"
done


# ============================================================
# 5. 两张卡并行评测
#
# 方法 0 -> GPU 0
# 方法 1 -> GPU 1
#
# 若发现多于两个实验，则 GPU 0/1 轮流分配。
# ============================================================

echo
echo "============================================================"
echo "Run Front3 ST-Flow and FVD"
echo "============================================================"

PIDS=()

for INDEX in "${!ROOTS[@]}"
do
(
    set -euo pipefail

    GPU=$((INDEX % 2))
    export CUDA_VISIBLE_DEVICES="$GPU"

    ROOT=${ROOTS[$INDEX]}
    NAME=$(basename "$ROOT")

    FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl
    FRONT_CONFIG=$ROOT/front3_eval_config.json
    LOG_DIR=$ROOT/eval_logs_front3

    mkdir -p "$LOG_DIR"

    CAMERA_CSV=$(python - "$FRONT_CONFIG" <<'PY'
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
    raise RuntimeError(
        f"Inconsistent frame counts: {distribution}"
    )

print(next(iter(distribution.keys())))
PY
    )

    echo
    echo "============================================================"
    echo "[GPU $GPU] Front3 evaluation"
    echo "METHOD: $NAME"
    echo "VIDEOS: $VIDEO_COUNT"
    echo "CAMERAS: $CAMERA_CSV"
    echo "PAIRS: $CAMERA_PAIRS"
    echo "SEQUENCE_COUNT: $SEQ_COUNT"
    echo "============================================================"


    # --------------------------------------------------------
    # ST-Flow / Traj
    #
    # filtered manifest 中只有三个前视相机。
    # 因此 Temporal 和 Traj 也只计算这三个视角。
    #
    # camera-pairs 只包含：
    #   左前 -> 前
    #   前 -> 右前
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
    #
    # 显式指定三个前视相机。
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
        --camera-names "$CAMERA_CSV" \
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
            --camera-names "$CAMERA_CSV" \
            --batch-size 1 \
            2>&1 | tee -a "$FVD_LOG"
    fi

    echo "[GPU $GPU] Completed: $NAME"
) &

    PIDS+=("$!")
done


# ============================================================
# 6. 等待评测结束
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
# 7. 汇总检查
# ============================================================

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
print("=" * 120)
print("nuScenes Front3 evaluation summary")
print("=" * 120)
print()
print(
    "| Method | Cameras | Videos | Temporal-L1 ↓ | "
    "Cross-Raw-Epi ↓ | Cross-Inlier ↑ | "
    "Traj-Epi ↓ | Traj-Inlier@2 ↑ | "
    "ST-Flow-D ↑ | ST-Flow-C ↑ | FVD ↓ | Status |"
)
print(
    "|---|---|---:|---:|---:|---:|---:|---:|"
    "---:|---:|---:|---|"
)

for root in roots:
    method = os.path.basename(root)

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

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        with open(stflow_path, "r", encoding="utf-8") as file:
            stflow_data = json.load(file)

        if not fvd_candidates:
            raise FileNotFoundError(
                "Front3 FVD result is missing."
            )

        fvd_path = fvd_candidates[-1]

        with open(fvd_path, "r", encoding="utf-8") as file:
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
        config = {
            "selected_cameras": [],
            "num_videos": 0,
        }
        temporal = None
        cross_raw = None
        cross_inlier = None
        traj = None
        traj_inlier2 = None
        stflow_d = None
        stflow_c = None
        fvd = None
        status = f"ERROR: {error}"
        failed = True

    def fmt(value, digits):
        if not isinstance(value, (int, float)):
            return "-"
        if not math.isfinite(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"

    cameras = ",".join(
        config.get("selected_cameras", [])
    )

    print(
        f"| {method} "
        f"| {cameras} "
        f"| {config.get('num_videos', 0)} "
        f"| {fmt(temporal, 6)} "
        f"| {fmt(cross_raw, 3)} "
        f"| {fmt(cross_inlier, 4)} "
        f"| {fmt(traj, 3)} "
        f"| {fmt(traj_inlier2, 4)} "
        f"| {fmt(stflow_d, 3)} "
        f"| {fmt(stflow_c, 3)} "
        f"| {fmt(fvd, 3)} "
        f"| {status} |"
    )

    summary[method] = {
        "root": root,
        "selected_cameras": config.get(
            "selected_cameras",
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
print("Summary:", summary_path)

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
echo "Summary:"
echo "$BASE/front3_evaluation_summary.json"
