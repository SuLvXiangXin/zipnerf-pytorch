#!/bin/bash

SCENE=("bicycle" "garden" "stump" "room" "counter" "kitchen" "bonsai")
DATA_ROOT=/SSD_DISK/datasets/360_v2

len=${#SCENE[@]}
for(( i=0; i<$len; i++ ))
do
  EXPERIMENT=360_v2/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"
  accelerate launch render.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 480" \
    --gin_bindings="Config.render_video_fps = 60" \
    --gin_bindings="Config.factor = 8"
done
