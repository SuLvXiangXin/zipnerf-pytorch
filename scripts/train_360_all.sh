#!/bin/bash

# outdoor
EXPERIMENT_PREFIX=360_v2_10w
SCENE=("bicycle" "garden" "stump" )
DATA_ROOT=data/360_v2

len=${#SCENE[@]}
for((i=0; i<$len; i++ ))
do
  EXPERIMENT=$EXPERIMENT_PREFIX/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  accelerate launch train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
      --gin_bindings="Config.factor = 4"

  accelerate launch eval.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4"

  accelerate launch render.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 120" \
  --gin_bindings="Config.render_video_fps = 30" \
  --gin_bindings="Config.factor = 4"

  accelerate launch extract.py \
  --gin_configs=configs/360.gin \
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

  accelerate launch train.py \
    --gin_configs=configs/360.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.factor = 2" \
    --gin_bindings="Config.max_steps = 100000" \

  accelerate launch eval.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 2"

  accelerate launch render.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 120" \
  --gin_bindings="Config.render_video_fps = 30" \
  --gin_bindings="Config.factor = 2"

  accelerate launch extract.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 2"
done