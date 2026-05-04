source /inspire/ssd/project/advanced-machine-learning/public/inspire_shared/envs/lyhdwm/bin/activate
cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
unset ENABLE_DEBUGPY
unset DEBUGPY_PORT

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/externals/TATS/tats/fvd
export PYTHONPATH=$PYTHONPATH:/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/OpenDWM/externals/waymo-open-dataset/src
torchrun \
  --nproc_per_node=4 \
  -m dwm.preview \
  -c //inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/OpenDWM/configs/ctsd/unimlvg/singlenusctest_copy.json\
  -o /inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/output/train4_refunisingle \
  --log-steps 300 \
  