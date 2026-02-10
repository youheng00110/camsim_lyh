#!/usr/bin/env bash
set -euo pipefail

############################
# 全局配置
############################
DATA_ROOT="/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/Avrgoverse2_sensor_data"
DATASET_NAME="sensor"  # 固定下载 sensor
MAX_RETRIES=5

############################
# 工具函数
############################
log() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

retry() {
  local n=1
  until "$@"; do
    if (( n >= MAX_RETRIES )); then
      echo "Command failed after $n attempts."
      return 1
    fi
    log "Retrying ($n/$MAX_RETRIES)..."
    sleep 3
    ((n++))
  done
}

############################
# 下载数据集（可中断恢复）
############################
TARGET_DIR="$DATA_ROOT"
mkdir -p "$TARGET_DIR"

log "Downloading AV2 dataset: $DATASET_NAME (train parts 1-14 only)"
log "Target dir: $TARGET_DIR"

# 仅下载 Train Part 1-14
PARTS=(
  "train/part_1"
  "train/part_2"
  "train/part_3"
  "train/part_4"
  "train/part_5"
  "train/part_6"
  "train/part_7"
  "train/part_8"
  "train/part_9"
  "train/part_10"
  "train/part_11"
  "train/part_12"
  "train/part_13"
  "train/part_14"
)

for part in "${PARTS[@]}"; do
  log "Downloading $part ..."
  mkdir -p "$TARGET_DIR/$part"
  retry s5cmd --no-sign-request \
    cp "s3://argoverse/datasets/av2/$DATASET_NAME/$part/*" "$TARGET_DIR/$part"
done

log "Download completed successfully."
