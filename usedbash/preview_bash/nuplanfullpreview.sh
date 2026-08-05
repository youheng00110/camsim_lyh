source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate

cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src

unset ENABLE_DEBUGPY
unset DEBUGPY_PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
  --nproc_per_node=8 \
  -m dwm.preview \
  -c /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim/nuplanpluckerfullpreview.json\
  -o /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/debug_nuplanfull18000