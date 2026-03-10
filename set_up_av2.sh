#!/usr/bin/env bash
set -euo pipefail

############################
# 全局配置
############################
DATA_ROOT="/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/Avrgoverse2_sensor_data"
DATASET_NAME="sensor"
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
    sleep 5
    ((n++))
  done
}

############################
# 下载 val
############################
TARGET_DIR="$DATA_ROOT"
mkdir -p "$TARGET_DIR"

BASE_URI="s3://argoverse/datasets/av2/$DATASET_NAME"

log "Downloading AV2 dataset: $DATASET_NAME (val only)"
log "Target directory: $TARGET_DIR"

log "Downloading val split ..."
mkdir -p "$TARGET_DIR/val"

retry s5cmd --no-sign-request cp \
  "$BASE_URI/val/*" \
  "$TARGET_DIR/val/"

log "Download completed successfully."