#!/bin/bash

SCENE=("bicycle" "garden" "stump" "room" "counter" "kitchen" "bonsai")
SCENE=("bonsai")
DATA_ROOT=/SSD_DISK/datasets/360_v2

len=${#SCENE[@]}
for(( i=0; i<$len; i++ ))
do
  EXPERIMENT=360_v2/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  rm exp/"$EXPERIMENT"/*
  accelerate launch train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
      --gin_bindings="Config.factor = 0"

  accelerate launch eval.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 0"
done
