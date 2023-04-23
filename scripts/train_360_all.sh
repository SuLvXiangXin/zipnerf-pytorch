#!/bin/bash

SCENE=("bicycle" "garden" "stump" "room" "counter" "kitchen" "bonsai")
SCENE=("bicycle" "garden" "stump")
DATA_ROOT=/SSD_DISK/datasets/360_v2

len=${#SCENE[@]}
for(( i=0; i<$len; i++ ))
do
  EXPERIMENT=360_v2_0423_02/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  rm exp/"$EXPERIMENT"/*
  accelerate launch --main_process_port 1250 train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
      --gin_bindings="Config.factor = 0"
done
