#!/bin/bash

SCENE=bicycle
EXPERIMENT=360_v2_glo/"$SCENE"
DATA_ROOT=/SSD_DISK/datasets/360_v2
DATA_DIR="$DATA_ROOT"/"$SCENE"

rm exp/"$EXPERIMENT"/*
python train.py --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.batch_size = 1024"

