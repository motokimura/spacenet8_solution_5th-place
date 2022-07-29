import argparse
import json
import os

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from skimage import io
from tqdm import tqdm

# isort: off
from spacenet8_model.datasets import get_test_dataloader
from spacenet8_model.models import get_model
from spacenet8_model.utils.config import load_config
from spacenet8_model.utils.misc import get_flatten_classes, save_array_as_geotiff
from train_net import get_default_cfg_path
# isort: on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_id',
        type=int,
        required=True
    )
    parser.add_argument(
        '--config',
        default=None,
        help='YAML config path. This will overwrite `configs/default.yaml`')
    parser.add_argument(
        '--artifact_dir',
        default='/wdata'
    )
    parser.add_argument(
        '--out_dir',
        default='/wdata'
    )
    parser.add_argument(
        '--device',
        default='cuda')
    parser.add_argument(
        '--val',
        action='store_true'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='overwrite configs (e.g., General.fp16=true, etc.)')
    return parser.parse_args()


def load_test_config(args):
    config_exp_path = os.path.join(args.artifact_dir, f'models/exp_{args.exp_id:05d}/config.yaml')
    config_exp: DictConfig = OmegaConf.load(config_exp_path)
    task: str = config_exp.task

    default_cfg_path: str = get_default_cfg_path(task)
    
    cfg_paths = [config_exp_path]
    if args.config is not None:
        cfg_paths.append(args.config)

    config: DictConfig = load_config(
        default_cfg_path,
        cfg_paths,
        update_dotlist=args.opts
    )
    return config


def crop_center(pred, crop_wh):
    _, h, w = pred.shape
    crop_w, crop_h = crop_wh
    assert w >= crop_w
    assert h >= crop_h

    left = (w - crop_w) // 2
    right = crop_w + left
    top = (h - crop_h) // 2
    bottom = crop_h + top

    return pred[:, top:bottom, left:right]


def dump_pred_to_png(pred, png_path):
    c, h, w = pred.shape
    # assert c <= 3
    if c > 3:
        pred = pred[:3]  # XXX: save all channels as TIFF
    assert pred.min() >= 0
    assert pred.max() <= 1
    array = np.zeros(shape=[h, w, 3], dtype=np.uint8)
    array[:, :, :c] = (pred * 255).astype(np.uint8).transpose((1, 2, 0))
    io.imsave(png_path, array, check_contrast=False)


def main():
    args = parse_args()

    config: DictConfig = load_test_config(args)

    model = get_model(config)
    ckpt_path = os.path.join(args.artifact_dir, f'models/exp_{args.exp_id:05d}/best.ckpt')
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    model.to(args.device)
    model.eval()

    out_root = 'val_preds' if args.val else 'preds'
    out_root = os.path.join(args.out_dir, out_root, f'exp_{args.exp_id:05d}')
    print(f'going to save prediction results under {out_root}')

    os.makedirs(out_root, exist_ok=True)

    # dump meta
    meta = {
        'groups': list(config.Class.groups),
        'classes': {
            g: list(cs) for g, cs in config.Class.classes.items()
        }
    }
    with open(os.path.join(out_root, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    test_dataloader = get_test_dataloader(config, test_to_val=args.val)
    for batch in tqdm(test_dataloader):
        images = batch['image'].to(args.device)
        batch_pre_paths = batch['pre_path']
        batch_orig_heights = batch['original_height']
        batch_orig_widths = batch['original_width']

        n_input_post_images = config.Model.n_input_post_images
        assert n_input_post_images in [0, 1, 2]
        images_post_a = None
        images_post_b = None
        if n_input_post_images == 1:
            images_post_a = batch['image_post_a'].to(args.device)
        elif n_input_post_images == 2:
            images_post_a = batch['image_post_a'].to(args.device)
            images_post_b = batch['image_post_b'].to(args.device)

        with torch.no_grad(): 
            batch_preds = model(images, images_post_a, images_post_b)
        batch_preds = torch.sigmoid(batch_preds)
        batch_preds = batch_preds.cpu().numpy()

        for i in range(images.shape[0]):
            pred = batch_preds[i]
            pre_path = batch_pre_paths[i]
            orig_h = batch_orig_heights[i].item()
            orig_w = batch_orig_widths[i].item()

            # set pred=0 on black pixels in the pre-image
            image = images[i].cpu().numpy()  # 3,H,W
            nodata_mask = np.sum(image, axis=0) == 0  # H,W
            pred[:, nodata_mask] = 0.0

            # set flooded_pred=0 on black pixels in the post-images
            nodata_mask = np.zeros(shape=[orig_h, orig_w], dtype=bool)
            if images_post_a is not None:
                image = images_post_a[i].cpu().numpy()
                nodata_mask = np.sum(image, axis=0) == 0  # H,W
            if images_post_b is not None:
                image = images_post_b[i].cpu().numpy()
                nodata_mask = nodata_mask & (np.sum(image, axis=0) == 0)  # H,W 
            classes = get_flatten_classes(config)
            for class_index, class_name in enumerate(classes):
                if class_name in ['flood_building', 'flood_road', 'flood']:
                    pred[class_index, nodata_mask] = 0.0

            pred = crop_center(pred, crop_wh=(orig_w, orig_h))

            assert pred.min() >= 0
            assert pred.max() <= 1
            pred_8bit = (pred * 255).astype(np.uint8)

            aoi = os.path.basename(os.path.dirname(os.path.dirname(pre_path)))
            filename = os.path.basename(pre_path)
            out_dir = os.path.join(out_root, aoi)
            os.makedirs(out_dir, exist_ok=True)

            save_array_as_geotiff(pred_8bit, pre_path, os.path.join(out_dir, filename))


if __name__ == '__main__':
    main()
