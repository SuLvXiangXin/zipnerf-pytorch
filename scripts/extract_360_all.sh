#!/bin/bash

# outdoor
EXPERIMENT_PREFIX=360_v2
SCENE=("bicycle" "garden" "stump" )
DATA_ROOT=/SSD_DISK/datasets/360_v2

len=${#SCENE[@]}
for(( i=0; i<$len; i++ ))
do
  EXPERIMENT=$EXPERIMENT_PREFIX/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"
  accelerate launch bake.py --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.factor = 4"
done

# indoor "Config.factor = 2"
SCENE=("room" "counter" "kitchen" "bonsai")
len=${#SCENE[@]}
for((i=0; i<$len; i++ ))
do
  EXPERIMENT=$EXPERIMENT_PREFIX/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  accelerate launch bake.py --gin_configs=configs/360.gin \
      --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
      --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
      --gin_bindings="Config.factor = 2"
done