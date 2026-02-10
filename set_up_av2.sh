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

BASE_URI="s3://argoverse/datasets/av2/$DATASET_NAME"

log "Downloading AV2 dataset: $DATASET_NAME (train only)"
log "Target dir: $TARGET_DIR"

log "Downloading train ..."
mkdir -p "$TARGET_DIR/train"
retry s5cmd --no-sign-request \
  cp "$BASE_URI/train/*" "$TARGET_DIR/train"

log "Download completed successfully."
