#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim"

NUPLAN_DIR="$CONFIG_DIR/nuplan"
WAYMO_DIR="$CONFIG_DIR/waymo"

NUPLAN_4CAM_DIR="$NUPLAN_DIR/preview4cam"
WAYMO_4CAM_DIR="$WAYMO_DIR/preview4cam"

mkdir -p \
    "$NUPLAN_4CAM_DIR" \
    "$WAYMO_4CAM_DIR"

shopt -s nullglob

# 先移动 preview4cam 配置，避免被普通规则提前移动。
for file in "$CONFIG_DIR"/nuplan*preview4cam.json; do
    target="$NUPLAN_4CAM_DIR/$(basename "$file")"

    if [[ -e "$target" ]]; then
        echo "[跳过] 目标已存在：$target"
        continue
    fi

    mv "$file" "$target"
    echo "[移动] $(basename "$file") -> nuplan/preview4cam/"
done

for file in "$CONFIG_DIR"/waymo*preview4cam.json; do
    target="$WAYMO_4CAM_DIR/$(basename "$file")"

    if [[ -e "$target" ]]; then
        echo "[跳过] 目标已存在：$target"
        continue
    fi

    mv "$file" "$target"
    echo "[移动] $(basename "$file") -> waymo/preview4cam/"
done

# 再移动剩余的普通配置。
for file in "$CONFIG_DIR"/nuplan*.json; do
    target="$NUPLAN_DIR/$(basename "$file")"

    if [[ -e "$target" ]]; then
        echo "[跳过] 目标已存在：$target"
        continue
    fi

    mv "$file" "$target"
    echo "[移动] $(basename "$file") -> nuplan/"
done

for file in "$CONFIG_DIR"/waymo*.json; do
    target="$WAYMO_DIR/$(basename "$file")"

    if [[ -e "$target" ]]; then
        echo "[跳过] 目标已存在：$target"
        continue
    fi

    mv "$file" "$target"
    echo "[移动] $(basename "$file") -> waymo/"
done

echo
echo "整理完成。"
echo "nuplan 普通配置：$(find "$NUPLAN_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "nuplan 4cam 配置：$(find "$NUPLAN_4CAM_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "waymo 普通配置：$(find "$WAYMO_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "waymo 4cam 配置：$(find "$WAYMO_4CAM_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)"
