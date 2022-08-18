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
nohup python tools/mosaic.py --fold_id 0 --train_dir $TRAIN_DIR > /wdata/mosaics/train_0.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 1 --train_dir $TRAIN_DIR > /wdata/mosaics/train_1.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 2 --train_dir $TRAIN_DIR > /wdata/mosaics/train_2.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 3 --train_dir $TRAIN_DIR > /wdata/mosaics/train_3.txt 2>&1 &
nohup python tools/mosaic.py --fold_id 4 --train_dir $TRAIN_DIR > /wdata/mosaics/train_4.txt 2>&1 &
wait

python tools/measure_image_similarities.py --train_dir $TRAIN_DIR --train_only

# training
LOG_DIR=/wdata/logs/train
mkdir -p $LOG_DIR

ARGS=" --override_model_dir /work/models --disable_wandb"
# comment out the line below for dryrun
#ARGS=$ARGS" General.epochs=5"

echo ""
echo "training... (1/5)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation \
    --exp_id 50000 \
    --fold_id 0 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50000.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation \
    --exp_id 50001 \
    --fold_id 1 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50001.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task foundation \
    --exp_id 50002 \
    --fold_id 2 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50002.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task foundation \
    --exp_id 50003 \
    --fold_id 3 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50003.txt 2>&1 &

wait

echo ""
echo "training... (2/5)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation \
    --exp_id 50004 \
    --fold_id 4 \
    --config configs/foundation/effnet-b5_e210.yaml \
    $ARGS \
    > $LOG_DIR/exp_50004.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation \
    --exp_id 60400 \
    --fold_id 0 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60400.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task foundation \
    --exp_id 60401 \
    --fold_id 1 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60401.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task foundation \
    --exp_id 60402 \
    --fold_id 2 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60402.txt 2>&1 &

wait

echo ""
echo "training... (3/5)"
echo "this will take ~3 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task foundation \
    --exp_id 60403 \
    --fold_id 3 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60403.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task foundation \
    --exp_id 60404 \
    --fold_id 4 \
    --config configs/foundation/effnet-b6_e140.yaml \
    $ARGS \
    > $LOG_DIR/exp_60404.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood \
    --exp_id 50010 \
    --pretrained 50000 \
    --fold_id 0 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50010.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task flood \
    --exp_id 50011 \
    --pretrained 50001 \
    --fold_id 1 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50011.txt 2>&1 &

wait

echo ""
echo "training... (4/5)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task flood \
    --exp_id 50012 \
    --pretrained 50002 \
    --fold_id 2 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50012.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task flood \
    --exp_id 50013 \
    --pretrained 50003 \
    --fold_id 3 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50013.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood \
    --exp_id 50014 \
    --pretrained 50004 \
    --fold_id 4 \
    --config configs/flood/effnet-b5_ks7_ema_e80.yaml \
    $ARGS \
    > $LOG_DIR/exp_50014.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=4 python tools/train_net.py \
    --task flood \
    --exp_id 60420 \
    --pretrained 60400 \
    --fold_id 0 \
    --config configs/flood/effnet-b6_ks7_ema.yaml \
    $ARGS \
    > $LOG_DIR/60420.txt 2>&1 &

wait

echo ""
echo "training... (5/5)"
echo "this will take ~4 hours"
echo "you can check progress from $LOG_DIR/*.txt"

nohup env CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --task flood \
    --exp_id 60421 \
    --pretrained 60401 \
    --fold_id 1 \
    --config configs/flood/effnet-b6_ks7_ema.yaml \
    $ARGS \
    > $LOG_DIR/60421.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
    --task flood \
    --exp_id 60422 \
    --pretrained 60402 \
    --fold_id 2 \
    --config configs/flood/effnet-b6_ks7_ema.yaml \
    $ARGS \
    > $LOG_DIR/60422.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
    --task flood \
    --exp_id 60423 \
    --pretrained 60403 \
    --fold_id 3 \
    --config configs/flood/effnet-b6_ks7_ema.yaml \
    $ARGS \
    > $LOG_DIR/60423.txt 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
    --task flood \
    --exp_id 60424 \
    --pretrained 60404 \
    --fold_id 4 \
    --config configs/flood/effnet-b6_ks7_ema.yaml \
    $ARGS \
    > $LOG_DIR/60424.txt 2>&1 &

wait

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Total time for training: " $(($ELAPSED_TIME / 60 + 1)) "[min]"
