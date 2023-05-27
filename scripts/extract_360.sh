#!/bin/bash

SCENE=bicycle
EXPERIMENT=blender/"$SCENE"
DATA_ROOT=/SSD_DISK/datasets/360_v2
DATA_DIR="$DATA_ROOT"/"$SCENE"

accelerate launch extract.py --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4"
