# spacenet8_solution

## data preparation

```
python tools/prepare_building_masks.py
python tools/prepare_road_masks.py

python tools/warp_post_images.py
python tools/warp_post_images.py --root_dir /data/test/ --out_dir /wdata/test_warped/
```