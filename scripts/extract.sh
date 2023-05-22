#!/bin/bash

SCENE=bicycle
EXPERIMENT=360_v2_0427_01/"$SCENE"
DATA_ROOT=/SSD_DISK/datasets/360_v2
DATA_DIR="$DATA_ROOT"/"$SCENE"

accelerate launch bake.py --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 8" \
  --gin_bindings="Config.render_chunk_size = 16384"
#  --gin_bindings="Config.llff_use_all_images_for_testing=True"