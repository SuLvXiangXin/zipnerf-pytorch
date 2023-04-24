#!/bin/bash

#OUTDOOR_SCENE=("bicycle" "garden" "stump")
INDOOR_SCENE=("room" "counter" "kitchen" "bonsai")
INDOOR_SCENE=("room" "counter" "kitchen")
DATA_ROOT=/SSD_DISK/datasets/360_v2

#len=${#OUTDOOR_SCENE[@]}
#for(( i=0; i<$len; i++ ))
#do
#  EXPERIMENT=360_v2/"${OUTDOOR_SCENE[i]}"
#  DATA_DIR="$DATA_ROOT"/"${OUTDOOR_SCENE[i]}"
#
#  accelerate launch --main_process_port 1270 eval.py \
#    --gin_configs=configs/360.gin \
#    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
#    --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
#done

len=${#INDOOR_SCENE[@]}
for(( i=0; i<$len; i++ ))
do
  EXPERIMENT=360_v2_0423_01/"${INDOOR_SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${INDOOR_SCENE[i]}"

  accelerate launch --main_process_port 1270 eval.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.factor = 4"
done