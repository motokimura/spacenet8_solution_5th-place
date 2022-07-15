#!/bin/bash

python tools/make_folds.py
python tools/prepare_building_masks.py
python tools/prepare_road_masks.py
python tools/warp_post_images.py
python tools/warp_post_images.py --root_dir /data/test/ --out_dir /wdata/test_warped/

# optional
python tools/visualize_dataset.py
python tools/visualize_dataset.py --root_dir /data/test/ --test
