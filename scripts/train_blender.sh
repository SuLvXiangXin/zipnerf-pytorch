#!/bin/bash

SCENE=chair
EXPERIMENT=blender/"$SCENE"
DATA_ROOT=/SSD_DISK/datasets/nerf_synthetic
DATA_DIR="$DATA_ROOT"/"$SCENE"

rm exp/"$EXPERIMENT"/*
accelerate launch train.py --gin_configs=configs/blender.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
