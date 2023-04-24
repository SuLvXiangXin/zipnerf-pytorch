#!/bin/bash

SCENE=bicycle
EXPERIMENT=360_v2/"$SCENE"
DATA_ROOT=/SSD_DISK/datasets/360_v2
DATA_DIR="$DATA_ROOT"/"$SCENE"

# If running one of the indoor scenes, add
# --gin_bindings="Config.factor = 2"

#rm exp/"$EXPERIMENT"/*
accelerate launch --main_process_port 1824 train.py --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.batch_size = 65536" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.render_chunk_size = 16384"
