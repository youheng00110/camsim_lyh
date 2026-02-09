#!/usr/bin/env bash
set -euo pipefail

############################
# 全局配置
############################
MINIFORGE_DIR="$HOME/conda"
CONDA_BIN="$MINIFORGE_DIR/bin/conda"
DATA_ROOT="$HOME/data/datasets/av2"
AV2_ENV="av2"
DATASET_NAME="${1:-sensor}"  # 默认 sensor，可通过参数覆盖
MAX_RETRIES=5

############################
# 工具函数
############################
log() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

exists() {
  command -v "$1" >/dev/null 2>&1
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
# 1. 安装 Miniforge（若不存在）
############################
if [[ ! -x "$CONDA_BIN" ]]; then
  log "Installing Miniforge..."
  retry wget -O Miniforge3.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3.sh -b -p "$MINIFORGE_DIR"
else
  log "Miniforge already installed."
fi

############################
# 2. 初始化 conda（一次即可）
############################
eval "$($CONDA_BIN shell.bash hook)" || true

############################
# 3. 创建 / 激活 av2 环境
############################
if ! conda env list | grep -q "$AV2_ENV"; then
  log "Creating conda env: $AV2_ENV"
  conda create -y -n "$AV2_ENV" python=3.9
fi

conda activate "$AV2_ENV"

############################
# 4. 安装 av2（conda 优先）
############################
if ! python -c "import av2" >/dev/null 2>&1; then
  log "Installing av2 via conda..."
  retry conda install -y -c conda-forge av2
else
  log "av2 already installed."
fi

############################
# 5. 安装 s5cmd
############################
if ! exists s5cmd; then
  log "Installing s5cmd..."
  retry conda install -y -c conda-forge s5cmd
else
  log "s5cmd already installed."
fi

############################
# 6. 下载数据集（可中断恢复）
############################
TARGET_DIR="$DATA_ROOT/$DATASET_NAME"
mkdir -p "$TARGET_DIR"

log "Downloading AV2 dataset: $DATASET_NAME"
log "Target dir: $TARGET_DIR"

retry s5cmd --no-sign-request \
  cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" "$TARGET_DIR"

log "Download completed successfully."
