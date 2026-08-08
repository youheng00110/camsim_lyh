#!/bin/bash

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

set -uo pipefail

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src || exit 1


# ============================================================
# 0. 环境
# ============================================================

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
# 1. 本地权重
# ============================================================

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
        echo "Checkpoint missing:"
        echo "$CHECKPOINT"
        exit 1
    fi
done


mkdir -p "$CACHE_DIR"

cp -f "$RAFT_CHECKPOINT" "$RAFT_CACHE"
cp -f "$LOFTR_CHECKPOINT" "$LOFTR_CACHE"


echo "============================================================"
echo "Local checkpoints ready"
echo "============================================================"

ls -lh "$RAFT_CACHE"
ls -lh "$LOFTR_CACHE"
ls -lh "$I3D_CHECKPOINT"


# ============================================================
# 2. 数据配置
# ============================================================

BASE_NUPLAN=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplan2fps

BASE_ABLATION=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuscenesablation


MAX_VIDEOS=500
GATE=16


# ============================================================
# 5 个任务
# ============================================================

SOURCES=(
    "$BASE_NUPLAN/implicit"
    "$BASE_NUPLAN/plucker"
    "$BASE_ABLATION/nuplan"
    "$BASE_ABLATION/nuscenesablationori3"
    "$BASE_ABLATION/nuscenesablationori6"
)


ROOTS=(
    "$BASE_NUPLAN/implicit_merged500"
    "$BASE_NUPLAN/plucker_merged500"
    "$BASE_ABLATION/nuplan_merged500"
    "$BASE_ABLATION/nuscenesablationori3_merged500"
    "$BASE_ABLATION/nuscenesablationori6_merged500"
)


DATASETS=(
    "nuplan"
    "nuplan"
    "nuplan"
    "nuscenes"
    "nuscenes"
)


# ============================================================
# 是否只测 Front3
#
# 前三个 nuPlan：全部视角
# 后两个 nuScenes：只测三个前视
# ============================================================

FRONT3=(
    "0"
    "0"
    "0"
    "1"
    "1"
)


# ============================================================
# nuScenes 前视 camera index
#
# ori3：
#   0 = front-left
#   1 = front
#   2 = front-right
#
# ori6：
#   1 = front-left
#   2 = front
#   3 = front-right
# ============================================================

VIEW_INDICES=(
    ""
    ""
    ""
    "0,1,2"
    "1,2,3"
)


# ============================================================
# GPU 分配
#
# GPU 0:
#   implicit
#
# GPU 1:
#   plucker
#
# GPU 2:
#   nuplan
#
# GPU 3:
#   nuscenesablationori3
#   nuscenesablationori6
#
# 两个 Front3 较轻，因此放同一张卡顺序跑。
# ============================================================

GPUS=(
    "0"
    "1"
    "2"
    "3"
    "3"
)


echo
echo "============================================================"
echo "Task assignment"
echo "============================================================"

for INDEX in "${!SOURCES[@]}"
do
    echo "[$INDEX] GPU ${GPUS[$INDEX]}"
    echo "    ${SOURCES[$INDEX]}"
done


# ============================================================
# 3. 合并所有实验 -> 前 500
# ============================================================

for INDEX in "${!SOURCES[@]}"
do
    SRC=${SOURCES[$INDEX]}
    ROOT=${ROOTS[$INDEX]}
    DATASET=${DATASETS[$INDEX]}

    MANIFEST=$ROOT/stflow_manifest.jsonl


    echo
    echo "============================================================"
    echo "MERGE [$INDEX]"
    echo "SRC:     $SRC"
    echo "ROOT:    $ROOT"
    echo "DATASET: $DATASET"
    echo "============================================================"


    if [ ! -d "$SRC" ]
    then
        echo "Source directory missing:"
        echo "$SRC"
        exit 1
    fi


    # --------------------------------------------------------
    # 已有 merged500 且数量为 500，则跳过
    # --------------------------------------------------------

    NEED_MERGE=1


    if [ -f "$MANIFEST" ]
    then
        EXISTING_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")

        echo "Existing merged videos: $EXISTING_COUNT"

        if [ "$EXISTING_COUNT" -eq "$MAX_VIDEOS" ]
        then
            NEED_MERGE=0

            echo "Already merged500."
            echo "Skip merge."
        fi
    fi


    # --------------------------------------------------------
    # 开始合并
    # --------------------------------------------------------

    if [ "$NEED_MERGE" -eq 1 ]
    then

        if ! python -m dwm.tools.merge_rank_preview_manifests_interleave \
            --input-root "$SRC" \
            --output-root "$ROOT" \
            --dataset-name "$DATASET" \
            --max-videos "$MAX_VIDEOS" \
            --overwrite
        then

            echo "Hard-link merge failed."
            echo "Retry with --copy."


            python -m dwm.tools.merge_rank_preview_manifests_interleave \
                --input-root "$SRC" \
                --output-root "$ROOT" \
                --dataset-name "$DATASET" \
                --max-videos "$MAX_VIDEOS" \
                --overwrite \
                --copy


            MERGE_EXIT=$?


            if [ "$MERGE_EXIT" -ne 0 ]
            then
                echo "Merge failed:"
                echo "$SRC"

                exit "$MERGE_EXIT"
            fi
        fi
    fi


    # --------------------------------------------------------
    # 检查 manifest
    # --------------------------------------------------------

    if [ ! -f "$MANIFEST" ]
    then
        echo "Merged manifest missing:"
        echo "$MANIFEST"

        exit 1
    fi


    COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")


    echo "Merged videos: $COUNT"


    if [ "$COUNT" -ne "$MAX_VIDEOS" ]
    then
        echo "ERROR:"
        echo "expected=$MAX_VIDEOS"
        echo "actual=$COUNT"

        exit 1
    fi
done


# ============================================================
# 4. 为两个 nuScenes 生成 Front3 manifest
#
# ori3 -> 0,1,2
# ori6 -> 1,2,3
#
# 重命名为：
#
# CAM_FRONT_LEFT
# CAM_FRONT
# CAM_FRONT_RIGHT
#
# Cross-view pair：
#
# CAM_FRONT_LEFT -> CAM_FRONT
# CAM_FRONT      -> CAM_FRONT_RIGHT
#
# 不闭环。
# ============================================================

for INDEX in 3 4
do

    ROOT=${ROOTS[$INDEX]}
    INDEX_SET=${VIEW_INDICES[$INDEX]}

    MANIFEST=$ROOT/stflow_manifest.jsonl

    FRONT_MANIFEST=$ROOT/stflow_manifest_front3.jsonl

    FRONT_CONFIG=$ROOT/front3_eval_config.json


    echo
    echo "============================================================"
    echo "Generate Front3 manifest [$INDEX]"
    echo "ROOT: $ROOT"
    echo "INDICES: $INDEX_SET"
    echo "============================================================"


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
output_path = sys.argv[2]
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


items = []


with open(
    manifest_path,
    "r",
    encoding="utf-8",
) as file:

    for line_number, line in enumerate(
        file,
        start=1,
    ):

        if not line.strip():
            continue


        try:
            item = json.loads(line)

        except json.JSONDecodeError as error:

            print(
                f"Invalid JSON line "
                f"{line_number}: {error}",
                file=sys.stderr,
            )

            sys.exit(1)


        items.append(item)


if not items:

    print(
        "Manifest empty.",
        file=sys.stderr,
    )

    sys.exit(1)


filtered_items = []

frame_counts = Counter()

original_view_counts = Counter()

filtered_view_counts = Counter()

first_original_cameras = None


for item_index, item in enumerate(items):

    new_item = copy.deepcopy(item)


    frames = new_item.get(
        "frames",
        [],
    )


    if not frames:

        print(
            f"Video {item_index} "
            f"has no frames.",
            file=sys.stderr,
        )

        sys.exit(1)


    frame_counts[
        len(frames)
    ] += 1


    for frame_index, frame in enumerate(frames):

        views = frame.get(
            "views",
            [],
        )


        original_view_counts[
            len(views)
        ] += 1


        if first_original_cameras is None:

            first_original_cameras = [
                view.get("camera")
                for view in views
            ]


        if len(views) <= max(selected_indices):

            print(
                f"Not enough views: "
                f"video={item_index}, "
                f"frame={frame_index}, "
                f"views={len(views)}, "
                f"indices={selected_indices}",
                file=sys.stderr,
            )


            print(
                "available cameras:",
                [
                    view.get("camera")
                    for view in views
                ],
                file=sys.stderr,
            )


            sys.exit(1)


        selected_views = []


        for output_index, source_index in enumerate(
            selected_indices
        ):

            view = copy.deepcopy(
                views[source_index]
            )


            view[
                "source_camera_name"
            ] = view.get(
                "camera"
            )


            view[
                "source_view_index"
            ] = source_index


            view[
                "camera"
            ] = canonical_names[
                output_index
            ]


            selected_views.append(
                view
            )


        frame[
            "views"
        ] = selected_views


        filtered_view_counts[
            len(selected_views)
        ] += 1


    filtered_items.append(
        new_item
    )


with open(
    output_path,
    "w",
    encoding="utf-8",
) as file:

    for item in filtered_items:

        file.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


config = {

    "num_videos":
        len(filtered_items),

    "selected_view_indices":
        selected_indices,

    "canonical_camera_names":
        canonical_names,

    "camera_pairs": [
        "CAM_FRONT_LEFT__CAM_FRONT",
        "CAM_FRONT__CAM_FRONT_RIGHT",
    ],

    "camera_pairs_csv": (
        "CAM_FRONT_LEFT__CAM_FRONT,"
        "CAM_FRONT__CAM_FRONT_RIGHT"
    ),

    "first_original_cameras":
        first_original_cameras,

    "frame_count_distribution":
        dict(frame_counts),

    "original_view_count_distribution":
        dict(original_view_counts),

    "filtered_view_count_distribution":
        dict(filtered_view_counts),
}


with open(
    config_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=2,
        ensure_ascii=False,
    )


print("=" * 80)

print(
    "Front3 manifest generated"
)

print(
    "videos:",
    len(filtered_items),
)

print(
    "selected indices:",
    selected_indices,
)

print(
    "original cameras:",
    first_original_cameras,
)

print(
    "canonical cameras:",
    canonical_names,
)

print(
    "pairs:",
    config["camera_pairs"],
)

print(
    "output:",
    output_path,
)

print("=" * 80)

PY


    FILTER_EXIT=$?


    if [ "$FILTER_EXIT" -ne 0 ]
    then
        echo "Front3 filtering failed:"
        echo "$ROOT"

        exit "$FILTER_EXIT"
    fi


    echo
    echo "Front3 config:"

    cat "$FRONT_CONFIG"

done


# ============================================================
# 5. 四卡并行评测
#
# GPU 0:
#   implicit
#
# GPU 1:
#   plucker
#
# GPU 2:
#   nuplan
#
# GPU 3:
#   ori3
#   -> ori6
#
# 单个任务内部仍然单卡。
# ============================================================

echo
echo "============================================================"
echo "Start 4 GPU workers"
echo "============================================================"


PIDS=()


for GPU in 0 1 2 3
do

(
    set -euo pipefail


    export CUDA_VISIBLE_DEVICES="$GPU"


    echo
    echo "============================================================"
    echo "[GPU $GPU] worker start"
    echo "============================================================"


    for INDEX in "${!SOURCES[@]}"
    do

        TASK_GPU=${GPUS[$INDEX]}


        if [ "$TASK_GPU" -ne "$GPU" ]
        then
            continue
        fi


        ROOT=${ROOTS[$INDEX]}

        IS_FRONT3=${FRONT3[$INDEX]}

        NAME=$(basename "$ROOT")


        LOG_DIR=$ROOT/eval_logs

        mkdir -p "$LOG_DIR"


        # ====================================================
        # 选择 manifest
        # ====================================================

        if [ "$IS_FRONT3" -eq 1 ]
        then

            MANIFEST=$ROOT/stflow_manifest_front3.jsonl

            STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16_front3.json

            STFLOW_LOG=$LOG_DIR/stflow_gate16_front3.log

        else

            MANIFEST=$ROOT/stflow_manifest.jsonl

            STFLOW_OUTPUT=$ROOT/stflow_traj_result_gate16.json

            STFLOW_LOG=$LOG_DIR/stflow_gate16.log

        fi


        if [ ! -f "$MANIFEST" ]
        then
            echo "[GPU $GPU] Manifest missing:"
            echo "$MANIFEST"

            exit 1
        fi


        VIDEO_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")


        # ====================================================
        # 获取 sequence count
        # ====================================================

        SEQ_COUNT=$(python - "$MANIFEST" <<'PY'

import json
import sys

from collections import Counter


manifest_path = sys.argv[1]


frame_counts = []


with open(
    manifest_path,
    "r",
    encoding="utf-8",
) as file:

    for line in file:

        if not line.strip():
            continue


        item = json.loads(line)


        frame_counts.append(
            len(
                item["frames"]
            )
        )


distribution = Counter(
    frame_counts
)


if not frame_counts:

    print(
        "Manifest empty.",
        file=sys.stderr,
    )

    sys.exit(1)


if len(distribution) != 1:

    print(
        "Inconsistent frame counts: "
        f"{dict(distribution)}",
        file=sys.stderr,
    )

    sys.exit(1)


print(
    frame_counts[0]
)

PY
        )


        echo
        echo "============================================================"
        echo "[GPU $GPU]"
        echo "TASK:      $INDEX"
        echo "NAME:      $NAME"
        echo "VIDEOS:    $VIDEO_COUNT"
        echo "SEQ_COUNT: $SEQ_COUNT"
        echo "FRONT3:    $IS_FRONT3"
        echo "============================================================"


        # ====================================================
        # ST-Flow / Traj
        # ====================================================

        if [ "$IS_FRONT3" -eq 1 ]
        then

            CAMERA_PAIRS="CAM_FRONT_LEFT__CAM_FRONT,CAM_FRONT__CAM_FRONT_RIGHT"


            echo
            echo "[GPU $GPU] Run Front3 ST-Flow gate16"


            python -m dwm.tools.evaluate_stflow \
                --manifest "$MANIFEST" \
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


        else

            echo
            echo "[GPU $GPU] Run nuPlan ST-Flow gate16"


            python -m dwm.tools.evaluate_stflow \
                --manifest "$MANIFEST" \
                --output "$STFLOW_OUTPUT" \
                --device cuda \
                --max-videos "$VIDEO_COUNT" \
                --frame-stride 2 \
                --min-matches 16 \
                --max-matches 256 \
                --loftr-confidence 0.1 \
                --pair-policy dataset \
                --cross-gate-px "$GATE" \
                2>&1 | tee "$STFLOW_LOG"

        fi


        # ====================================================
        # FVD
        # ====================================================

        if [ "$IS_FRONT3" -eq 1 ]
        then

            FVD_OUTPUT=$ROOT/paired_fvd_result_front3_all${SEQ_COUNT}.json

            FVD_LOG=$LOG_DIR/fvd_front3_all${SEQ_COUNT}.log

        else

            FVD_OUTPUT=$ROOT/paired_fvd_result_all${SEQ_COUNT}.json

            FVD_LOG=$LOG_DIR/fvd_all${SEQ_COUNT}.log

        fi


        echo
        echo "[GPU $GPU] Run FVD"


        if ! python -m dwm.tools.evaluate_fvd_from_paired_manifest \
            --manifest "$MANIFEST" \
            --output "$FVD_OUTPUT" \
            --i3d-checkpoint "$I3D_CHECKPOINT" \
            --device cuda \
            --max-videos "$VIDEO_COUNT" \
            --sequence-count "$SEQ_COUNT" \
            --batch-size 2 \
            2>&1 | tee "$FVD_LOG"
        then

            echo
            echo "[GPU $GPU] FVD batch-size=2 failed."
            echo "[GPU $GPU] Retry batch-size=1."


            python -m dwm.tools.evaluate_fvd_from_paired_manifest \
                --manifest "$MANIFEST" \
                --output "$FVD_OUTPUT" \
                --i3d-checkpoint "$I3D_CHECKPOINT" \
                --device cuda \
                --max-videos "$VIDEO_COUNT" \
                --sequence-count "$SEQ_COUNT" \
                --batch-size 1 \
                2>&1 | tee -a "$FVD_LOG"

        fi


        echo
        echo "[GPU $GPU] DONE:"
        echo "$NAME"

    done


    echo
    echo "============================================================"
    echo "[GPU $GPU] worker finished"
    echo "============================================================"

) &


    PIDS+=("$!")

done


# ============================================================
# 6. 等待四张卡
# ============================================================

STATUS=0


for PID in "${PIDS[@]}"
do

    if ! wait "$PID"
    then
        echo "Worker failed:"
        echo "PID=$PID"

        STATUS=1
    fi

done


if [ "$STATUS" -ne 0 ]
then
    echo "At least one evaluation failed."

    exit 1
fi


# ============================================================
# 7. 最终统计
# ============================================================

echo
echo "============================================================"
echo "Evaluation summary"
echo "============================================================"


python - "${ROOTS[@]}" <<'PY'

import glob
import json
import math
import os
import sys


roots = sys.argv[1:]


print()

print(
    "| Method | Videos | Temporal-L1 ↓ | "
    "Cross-Raw-Epi ↓ | Cross-Inlier ↑ | "
    "Traj-Epi ↓ | Traj-Inlier@2 ↑ | "
    "ST-Flow-D ↑ | FVD ↓ |"
)

print(
    "|---|---:|---:|---:|---:|"
    "---:|---:|---:|---:|"
)


for root in roots:

    method = os.path.basename(root)


    front3_path = os.path.join(
        root,
        "stflow_traj_result_gate16_front3.json",
    )


    normal_path = os.path.join(
        root,
        "stflow_traj_result_gate16.json",
    )


    if os.path.isfile(front3_path):

        stflow_path = front3_path


        fvd_candidates = glob.glob(
            os.path.join(
                root,
                "paired_fvd_result_front3_all*.json",
            )
        )


        manifest = os.path.join(
            root,
            "stflow_manifest_front3.jsonl",
        )


    else:

        stflow_path = normal_path


        fvd_candidates = glob.glob(
            os.path.join(
                root,
                "paired_fvd_result_all*.json",
            )
        )


        manifest = os.path.join(
            root,
            "stflow_manifest.jsonl",
        )


    videos = 0


    if os.path.isfile(manifest):

        with open(
            manifest,
            "r",
            encoding="utf-8",
        ) as file:

            videos = sum(
                1
                for line in file
                if line.strip()
            )


    if not os.path.isfile(stflow_path):

        print(
            f"| {method} "
            f"| {videos} "
            "| MISSING | | | | | | |"
        )

        continue


    with open(
        stflow_path,
        "r",
        encoding="utf-8",
    ) as file:

        stflow = json.load(file)


    mean = stflow.get(
        "mean",
        {},
    )


    fvd = None


    if fvd_candidates:

        fvd_path = max(
            fvd_candidates,
            key=os.path.getmtime,
        )


        with open(
            fvd_path,
            "r",
            encoding="utf-8",
        ) as file:

            fvd_data = json.load(file)


        fvd = fvd_data.get(
            "fvd"
        )


    def fmt(value, digits):

        if not isinstance(
            value,
            (int, float),
        ):
            return "-"


        if not math.isfinite(
            float(value)
        ):
            return "-"


        return (
            f"{float(value):.{digits}f}"
        )


    print(
        f"| {method} "
        f"| {videos} "
        f"| {fmt(mean.get('temporal_l1'), 6)} "
        f"| {fmt(mean.get('cross_raw_epi_px'), 3)} "
        f"| {fmt(mean.get('cross_inlier_ratio'), 4)} "
        f"| {fmt(mean.get('traj_epi_px'), 3)} "
        f"| {fmt(mean.get('traj_inlier2'), 4)} "
        f"| {fmt(mean.get('stflow_d_score'), 3)} "
        f"| {fmt(fvd, 3)} |"
    )

PY


echo
echo "============================================================"
echo "ALL 5 EVALUATIONS DONE"
echo "============================================================"

echo
echo "GPU assignment:"
echo "GPU0: implicit"
echo "GPU1: plucker"
echo "GPU2: nuplan"
echo "GPU3: nuscenesablationori3 -> nuscenesablationori6"
