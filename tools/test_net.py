import argparse
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
        '--out_dir'
    )
    parser.add_argument(
        '--device',
        default='cuda')
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='overwrite configs (e.g., General.fp16=true, etc.)')
    return parser.parse_args()


def load_test_config(args):
    config_exp_path = os.path.join(args.artifact_dir, f'models/exp_{args.exp_id:04d}/config.yaml')
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
    assert c <= 3
    assert pred.min() >= 0
    assert pred.max() <= 1
    array = np.zeros(shape=[h, w, 3], dtype=np.uint8)
    array[:, :, :c] = (pred * 255).astype(np.uint8).transpose((1, 2, 0))
    io.imsave(png_path, array, check_contrast=False)


def main():
    args = parse_args()
    args.out_dir = args.artifact_dir if args.out_dir is None else args.out_dir

    config: DictConfig = load_test_config(args)

    model = get_model(config)
    ckpt_path = os.path.join(args.artifact_dir, f'models/exp_{args.exp_id:04d}/best.ckpt')
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict'])
    model.to(args.device)
    model.eval()

    out_root = out_dir = os.path.join(args.out_dir, f'preds_{config.task}', f'exp_{args.exp_id:04d}')
    print(f'will save prediction results under {out_root}')

    test_dataloader = get_test_dataloader(config)
    for batch in tqdm(test_dataloader):
        images = batch['image'].to(args.device)
        batch_pre_paths = batch['pre_path']
        batch_orig_heights = batch['original_height']
        batch_orig_widths = batch['original_width']

        with torch.no_grad(): 
            batch_preds = model(images)
        batch_preds = torch.sigmoid(batch_preds)
        batch_preds = batch_preds.cpu().numpy()

        for pred, pre_path, orig_h, orig_w in zip(
            batch_preds, batch_pre_paths, batch_orig_heights, batch_orig_widths):
            pred = crop_center(pred, crop_wh=(orig_w.item(), orig_h.item()))

            aoi = os.path.basename(os.path.dirname(os.path.dirname(pre_path)))
            filename = os.path.basename(pre_path)
            filename, _ = os.path.splitext(filename)
            filename = f'{filename}.png'
            out_dir = os.path.join(out_root, aoi)
            os.makedirs(out_dir, exist_ok=True)
            dump_pred_to_png(pred, os.path.join(out_dir, filename))


if __name__ == '__main__':
    main()
