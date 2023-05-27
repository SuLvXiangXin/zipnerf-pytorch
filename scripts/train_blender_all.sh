#!/bin/bash

# outdoor
EXPERIMENT_PREFIX=blender
SCENE=("drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
DATA_ROOT=/SSD_DISK/datasets/nerf_synthetic

len=${#SCENE[@]}
for((i=0; i<$len; i++ ))
do
  EXPERIMENT=$EXPERIMENT_PREFIX/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  accelerate launch train.py \
    --gin_configs=configs/blender.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'"

  accelerate launch eval.py \
  --gin_configs=configs/blender.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"

  accelerate launch extract.py \
  --gin_configs=configs/blender.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
done