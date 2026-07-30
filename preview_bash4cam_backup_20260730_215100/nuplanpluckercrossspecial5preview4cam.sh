#!/usr/bin/env bash
set -euo pipefail

source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src

unset ENABLE_DEBUGPY
unset DEBUGPY_PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OPENDWM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM
export CAMSIM_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$OPENDWM_ROOT/externals/TATS/tats/fvd:$PYTHONPATH
export PYTHONPATH=$CAMSIM_ROOT/nuplan-devkit-master:$PYTHONPATH
export PYTHONPATH=$OPENDWM_ROOT/externals/waymo-open-dataset/src:$PYTHONPATH

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m dwm.preview \
  -c "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim/nuplan/preview4cam/nuplanpluckercrossspecial5preview4cam.json" \
  -o "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_nuplanpluckercrossspecial5_4cam"
