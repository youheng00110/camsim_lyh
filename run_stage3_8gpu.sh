#!/usr/bin/env bash
set -euo pipefail

source /inspire/ssd/project/wuliqifa/public/songbur/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/src

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
unset ENABLE_DEBUGPY
unset DEBUGPY_PORT

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/OpenDWM/externals/TATS/tats/fvd"
export PYTHONPATH="${PYTHONPATH}:/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/externals/waymo-open-dataset/src"

CONFIG_PATH="/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/nusctest.json"
OUTPUT_PATH="/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/output/train_stage3_8gpu"

echo "Using python: $(which python)"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

torchrun \
  --nproc_per_node=8 \
  -m dwm.train \
  -c "${CONFIG_PATH}" \
  -o "${OUTPUT_PATH}" \
  --log-steps 100