import argparse
import os
from distutils.command.build import build
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/data/train')
    parser.add_argument('--preprocessed_dir', default='/wdata')
    parser.add_argument('--out_dir', default='/wdata')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def get_mapping_csv(args, aoi):
    mapping_csv_path = glob(os.path.join(args.root_dir, aoi, '*_mapping.csv'))
    assert len(mapping_csv_path) == 1, mapping_csv_path
    return mapping_csv_path[0]


def load_images(images, args, aoi):
    # load pre image
    pre, post1, post2 = images  # pre, post-1, post-2 image file names
    pre_path = os.path.join(args.root_dir, aoi, 'PRE-event', pre)
    assert os.path.exists(pre_path), pre_path
    pre_image = io.imread(pre_path)
    h, w = pre_image.shape[:2]

    # load post-1 image if exists
    warped_dir = 'test_warped' if args.test else 'train_warped'
    post1_path = os.path.join(args.preprocessed_dir, warped_dir, aoi, post1)
    post1_exists = os.path.exists(post1_path)
    if post1_exists:
        post1_image = io.imread(post1_path)
    else:
        post1_image = np.zeros((h, w, 3), dtype=np.uint8)

    # load post-2 image if exists
    post2_exists = False
    post2_path = None
    if isinstance(post2, str):
        post2_path = os.path.join(args.preprocessed_dir, warped_dir, aoi, post2)
        post2_exists = os.path.exists(post2_path)
    if post2_exists:
        post2_image = io.imread(post2_path)
    else:
        post2_image = np.zeros((h, w, 3), dtype=np.uint8)

    # check at least either post1 or post2 exists
    assert post1_exists or post2_exists, (post1_path, post2_path)

    return pre_image, post1_image, post2_image


def load_masks(images, args, aoi):
    pre, _, _ = images
    image_id, _ = os.path.splitext(pre)

    # load building 3-channel mask
    building_3channel_dir = os.path.join(args.preprocessed_dir, 'building_masks_3channel', aoi)
    building_3channel_path = os.path.join(building_3channel_dir, f'{image_id}.png')
    building_3channel = io.imread(building_3channel_path)

    # load building flood mask
    building_flood_dir = os.path.join(args.preprocessed_dir, 'building_masks_flood', aoi)
    building_flood_path = os.path.join(building_flood_dir, f'{image_id}.png')
    building_flood = io.imread(building_flood_path)

    # load road mask
    road_dir = os.path.join(args.preprocessed_dir, 'road_masks', aoi)
    road_path = os.path.join(road_dir, f'{image_id}.png')
    road = io.imread(road_path)

    return building_3channel, building_flood, road


def visualize_image(images, args, aoi, out_dir):
    pre_image, post1_image, post2_image = load_images(images, args, aoi)

    h, w = pre_image.shape[:2]
    canvas_images = np.zeros((h, 3 * w, 3), dtype=np.uint8)
    canvas_images[:, :w] = pre_image
    canvas_images[:, w:2*w] = post1_image
    canvas_images[:, 2*w:] = post2_image

    pre, _, _ = images
    image_id, _ = os.path.splitext(pre)
    out_path = os.path.join(out_dir, f'{image_id}.jpg')
    if args.test:
        io.imsave(out_path, canvas_images, check_contrast=False)
        return

    building_3channel, building_flood, road = load_masks(images, args, aoi)

    foundation = np.zeros((h, w, 3), dtype=np.uint8)
    foundation[:, :, 1] = ((building_3channel[:, :, 0] + road[:, :, 0] + road[:, :, 1]) > 0).astype(np.uint8) * 255
    foundation[:, :, 2] = ((building_3channel[:, :, 1] + building_3channel[:, :, 2] + road[:, :, 2]) > 0).astype(np.uint8) * 255

    flood = np.zeros((h, w, 3), dtype=np.uint8)
    flood[:, :, 0] = ((building_flood[:, :, 0] + road[:, :, 0]) > 0).astype(np.uint8) * 255
    flood[:, :, 1] = ((building_flood[:, :, 1] + road[:, :, 1]) > 0).astype(np.uint8) * 255

    canvas = np.zeros((2 * h, 3 * w, 3), dtype=np.uint8)
    # 1st row
    canvas[:h, :] = canvas_images
    # 2nd row
    canvas[h:, :w] = foundation
    canvas[h:, w:2*w] = flood

    io.imsave(out_path, canvas, check_contrast=False)


def visualize_aoi(args, aoi):
    vis_dir = 'vis_test' if args.test else 'vis_train'
    out_dir = os.path.join(args.out_dir, vis_dir, aoi)
    os.makedirs(out_dir, exist_ok=True)

    mapping_csv_path = get_mapping_csv(args, aoi)
    df = pd.read_csv(mapping_csv_path)
    images_list = df[['pre-event image', 'post-event image 1', 'post-event image 2']].values
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(images_list)) as pbar:
            for _ in pool.imap_unordered(partial(visualize_image, args=args, aoi=aoi, out_dir=out_dir), images_list):
                pbar.update()


def main():
    args = parse_args()
    aois = [d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))]
    for aoi in aois:
        print(f'visualizing {aoi} AOI')
        visualize_aoi(args, aoi)


if __name__ == '__main__':
    main()
