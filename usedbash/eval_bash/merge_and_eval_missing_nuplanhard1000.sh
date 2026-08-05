#!/usr/bin/env bash
source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
set -euo pipefail

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH"
export PYTHONPATH="$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH"
export PYTHONPATH="$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH"

BASE=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/eval/nuplanhard1000
MAX_VIDEOS=1000
GATE=16

# 只处理这三个当前没有 merged1000 的方法。
# 明确忽略：
#   shared_box_preview_paired_200
#   nuplandwm_preview_paired_200
NAMES=(
    0plucker_special_noid_preview_paired_200
    nuplanfull_preview_paired_200
    tokenearly24000_preview_paired_200
)

MODE=${1:-merge}

if [[ "$MODE" != "merge" && "$MODE" != "eval" && "$MODE" != "all" ]]
then
    echo "Usage: bash $0 [merge|eval|all]"
    exit 2
fi

if [[ "$MODE" == "merge" || "$MODE" == "all" ]]
then
    echo "============================================================"
    echo "Merge missing nuPlan hard previews"
    echo "============================================================"

    for NAME in "${NAMES[@]}"
    do
        SRC="$BASE/$NAME"
        DST="$BASE/${NAME}_merged1000"
        MANIFEST="$DST/stflow_manifest.jsonl"

        echo
        echo "METHOD: $NAME"
        echo "SOURCE: $SRC"
        echo "TARGET: $DST"

        if [[ ! -d "$SRC" ]]
        then
            echo "Source directory not found: $SRC"
            exit 1
        fi

        MERGED_VALID=0

        if [[ -f "$MANIFEST" ]]
        then
            if python - "$MANIFEST" "$DST" "$MAX_VIDEOS" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
merged_root = sys.argv[2]
expected_count = int(sys.argv[3])

items = []
with open(manifest_path, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON at line {line_number}: {error}"
            ) from error

if len(items) != expected_count:
    raise RuntimeError(
        f"Manifest count mismatch: {len(items)} != {expected_count}"
    )

for item_index in (0, len(items) - 1):
    item = items[item_index]
    frames = item.get("frames", [])
    if not frames:
        raise RuntimeError(f"Video {item_index} has no frames")

    for frame_index in (0, len(frames) - 1):
        views = frames[frame_index].get("views", [])
        if not views:
            raise RuntimeError(
                f"Video {item_index}, frame {frame_index} has no views"
            )

        image_path = views[0].get("image_path")
        if not image_path:
            raise RuntimeError("Missing image_path")

        if not os.path.isabs(image_path):
            image_path = os.path.join(merged_root, image_path)

        if not os.path.isfile(image_path):
            raise RuntimeError(
                f"Image does not exist: {image_path}"
            )

print(f"Valid merged manifest: {len(items)} videos")
PY
            then
                MERGED_VALID=1
            fi
        fi

        if [[ "$MERGED_VALID" -eq 1 ]]
        then
            echo "Already valid. Skip merge."
            continue
        fi

        rm -rf "$DST"

        if ! python -m dwm.tools.merge_rank_preview_manifests_interleave             --input-root "$SRC"             --output-root "$DST"             --dataset-name nuplan             --max-videos "$MAX_VIDEOS"             --overwrite
        then
            echo "Hard-link merge failed. Retry with --copy."
            rm -rf "$DST"

            python -m dwm.tools.merge_rank_preview_manifests_interleave                 --input-root "$SRC"                 --output-root "$DST"                 --dataset-name nuplan                 --max-videos "$MAX_VIDEOS"                 --overwrite                 --copy
        fi

        if [[ ! -f "$MANIFEST" ]]
        then
            echo "Manifest missing after merge: $MANIFEST"
            exit 1
        fi

        VIDEO_COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")
        echo "Merged video count: $VIDEO_COUNT"

        if [[ "$VIDEO_COUNT" -ne "$MAX_VIDEOS" ]]
        then
            echo "Expected $MAX_VIDEOS videos, found $VIDEO_COUNT"
            exit 1
        fi
    done
fi

if [[ "$MODE" == "eval" || "$MODE" == "all" ]]
then
    export TORCH_HOME=/root/.cache/torch

    CKPT_ROOT="$CAMSIM_ROOT/ckpt"
    CACHE_DIR="$TORCH_HOME/hub/checkpoints"

    RAFT_SOURCE="$CKPT_ROOT/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    LOFTR_SOURCE="$CKPT_ROOT/loftr_outdoor.ckpt"
    I3D_CHECKPOINT="$CKPT_ROOT/i3d_pretrained_400.pt"

    for CHECKPOINT in         "$RAFT_SOURCE"         "$LOFTR_SOURCE"         "$I3D_CHECKPOINT"
    do
        if [[ ! -f "$CHECKPOINT" ]]
        then
            echo "Checkpoint not found: $CHECKPOINT"
            exit 1
        fi
    done

    mkdir -p "$CACHE_DIR"
    cp -f "$RAFT_SOURCE"         "$CACHE_DIR/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
    cp -f "$LOFTR_SOURCE"         "$CACHE_DIR/loftr_outdoor.ckpt"

    for NAME in "${NAMES[@]}"
    do
        MANIFEST="$BASE/${NAME}_merged1000/stflow_manifest.jsonl"

        if [[ ! -f "$MANIFEST" ]]
        then
            echo "Merged manifest missing: $MANIFEST"
            exit 1
        fi

        COUNT=$(grep -cve '^[[:space:]]*$' "$MANIFEST")
        if [[ "$COUNT" -ne "$MAX_VIDEOS" ]]
        then
            echo "Invalid merged count for $NAME: $COUNT"
            exit 1
        fi
    done

    echo "============================================================"
    echo "Run ST-Flow gate16 on GPUs 0,1,2"
    echo "============================================================"

    STFLOW_PIDS=()

    for INDEX in "${!NAMES[@]}"
    do
        GPU=$INDEX
        NAME=${NAMES[$INDEX]}

        (
            set -euo pipefail
            export CUDA_VISIBLE_DEVICES="$GPU"

            ROOT="$BASE/${NAME}_merged1000"
            MANIFEST="$ROOT/stflow_manifest.jsonl"
            OUTPUT="$ROOT/stflow_traj_result_gate16.json"
            LOG_DIR="$ROOT/eval_logs"
            LOG_FILE="$LOG_DIR/stflow_gate16.log"

            mkdir -p "$LOG_DIR"

            echo "[GPU $GPU] ST-Flow start: $NAME"

            python -m dwm.tools.evaluate_stflow                 --manifest "$MANIFEST"                 --output "$OUTPUT"                 --device cuda                 --max-videos "$MAX_VIDEOS"                 --frame-stride 2                 --min-matches 16                 --max-matches 256                 --loftr-confidence 0.1                 --pair-policy dataset                 --cross-gate-px "$GATE"                 2>&1 | tee "$LOG_FILE"

            echo "[GPU $GPU] ST-Flow completed: $NAME"
        ) &

        STFLOW_PIDS+=("$!")
    done

    STFLOW_STATUS=0

    for PID in "${STFLOW_PIDS[@]}"
    do
        if ! wait "$PID"
        then
            echo "ST-Flow worker failed: PID=$PID"
            STFLOW_STATUS=1
        fi
    done

    if [[ "$STFLOW_STATUS" -ne 0 ]]
    then
        exit 1
    fi

    echo "============================================================"
    echo "Run FVD on GPUs 0,1,2"
    echo "============================================================"

    FVD_PIDS=()

    for INDEX in "${!NAMES[@]}"
    do
        GPU=$INDEX
        NAME=${NAMES[$INDEX]}

        (
            set -euo pipefail
            export CUDA_VISIBLE_DEVICES="$GPU"

            ROOT="$BASE/${NAME}_merged1000"
            MANIFEST="$ROOT/stflow_manifest.jsonl"
            LOG_DIR="$ROOT/eval_logs"

            mkdir -p "$LOG_DIR"

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

if not frame_counts:
    raise RuntimeError("Manifest is empty")

if len(distribution) != 1:
    raise RuntimeError(
        f"Inconsistent frame counts: {dict(distribution)}"
    )

print(frame_counts[0])
PY
            )

            OUTPUT="$ROOT/paired_fvd_result_all${SEQ_COUNT}.json"
            LOG_FILE="$LOG_DIR/fvd_all${SEQ_COUNT}.log"

            echo "[GPU $GPU] FVD start: $NAME, seq=$SEQ_COUNT"

            if ! python -m dwm.tools.evaluate_fvd_from_paired_manifest                 --manifest "$MANIFEST"                 --output "$OUTPUT"                 --i3d-checkpoint "$I3D_CHECKPOINT"                 --device cuda                 --max-videos "$MAX_VIDEOS"                 --sequence-count "$SEQ_COUNT"                 --batch-size 2                 2>&1 | tee "$LOG_FILE"
            then
                echo "[GPU $GPU] FVD batch-size=2 failed. Retry with batch-size=1."

                python -m dwm.tools.evaluate_fvd_from_paired_manifest                     --manifest "$MANIFEST"                     --output "$OUTPUT"                     --i3d-checkpoint "$I3D_CHECKPOINT"                     --device cuda                     --max-videos "$MAX_VIDEOS"                     --sequence-count "$SEQ_COUNT"                     --batch-size 1                     2>&1 | tee -a "$LOG_FILE"
            fi

            echo "[GPU $GPU] FVD completed: $NAME"
        ) &

        FVD_PIDS+=("$!")
    done

    FVD_STATUS=0

    for PID in "${FVD_PIDS[@]}"
    do
        if ! wait "$PID"
        then
            echo "FVD worker failed: PID=$PID"
            FVD_STATUS=1
        fi
    done

    if [[ "$FVD_STATUS" -ne 0 ]]
    then
        exit 1
    fi

    echo "============================================================"
    echo "Evaluation outputs"
    echo "============================================================"

    for NAME in "${NAMES[@]}"
    do
        ROOT="$BASE/${NAME}_merged1000"
        echo "$NAME"
        ls -lh             "$ROOT/stflow_traj_result_gate16.json"             "$ROOT"/paired_fvd_result_all*.json
    done
fi

echo "Done. MODE=$MODE"
