#!/bin/bash

TEST_DIR=$2
OUT_PATH=$3

# preprocess
python tools/make_test_csv.py --test_dir $TEST_DIR
python tools/warp_post_images.py --root_dir $TEST_DIR --test
python tools/measure_image_similarities.py --test_dir $TEST_DIR --test_only

# inference
# TODO

# postprocess
# TODO
