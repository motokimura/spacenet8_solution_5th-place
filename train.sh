#!/bin/bash

START_TIME=$SECONDS

TRAIN_DIR=$1

# remove motokimura"s home-built models
rm -rf /work/models

# preprocess
python tools/make_folds_v3.py --train_dir $TRAIN_DIR
python tools/prepare_building_masks.py --train_dir $TRAIN_DIR
python tools/prepare_road_masks.py --train_dir $TRAIN_DIR
python tools/warp_post_images.py --root_dir $TRAIN_DIR

echo "mosaicing.. this will take ~20 mins"
echo "you can check progress from /wdata/mosaics/train_*.txt"
mkdir -p /wdata/mosaics/
nohup python tools/mosaic.py --fold_id 0 > /wdata/mosaics/train_0.txt --train_dir $TRAIN_DIR 2>&1 &
nohup python tools/mosaic.py --fold_id 1 > /wdata/mosaics/train_1.txt --train_dir $TRAIN_DIR 2>&1 &
nohup python tools/mosaic.py --fold_id 2 > /wdata/mosaics/train_2.txt --train_dir $TRAIN_DIR 2>&1 &
nohup python tools/mosaic.py --fold_id 3 > /wdata/mosaics/train_3.txt --train_dir $TRAIN_DIR 2>&1 &
nohup python tools/mosaic.py --fold_id 4 > /wdata/mosaics/train_4.txt --train_dir $TRAIN_DIR 2>&1 &
wait

python tools/measure_image_similarities.py --train_dir $TRAIN_DIR --train_only

# training
# TODO

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Total time for training: " $(($ELAPSED_TIME / 60 + 1)) "[min]"
