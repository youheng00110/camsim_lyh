source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src


unset ENABLE_DEBUGPY
unset DEBUGPY_PORT
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/externals/TATS/tats/fvd
export PYTHONPATH=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan-devkit-master:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/externals/waymo-open-dataset/src
torchrun \
  --nproc_per_node=4 \
  -m dwm.train \
  -c /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/camsim/waymo/waymotokensingle.json\
  -o /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/train_waymotokensinglenew\
  --log-steps  500 \
  --preview-steps 1000 \
  --checkpointing-steps 6000 \
  --evaluation-steps 200000