import argparse
import os
import shutil

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# isort: off
from spacenet8_model.datasets import get_dataloader
from spacenet8_model.models import get_model
from spacenet8_model.utils.config import load_config
from spacenet8_model.utils.wandb import get_wandb_logger
# isort: on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        choices=['building', 'road', 'flood'],
        required=True
    )
    parser.add_argument(
        '--config',
        default=None,
        help='YAML config path. This will overwrite `configs/default.yaml`')
    parser.add_argument(
        '--debug', action='store_true', help='run in debug mode')
    parser.add_argument(
        '--disable_wandb', action='store_true', help='disable W&B logger')
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='overwrite configs (e.g., General.fp16=true, etc.)')
    return parser.parse_args()


def get_default_cfg_path(model_type: str) -> str:
    if model_type == 'building':
        return 'configs/defaults/foundation_building.yaml'
    elif model_type == 'road':
        raise NotImplementedError()
    elif model_type == 'flood':
        raise NotImplementedError()
    else:
        raise ValueError()


def main() -> None:
    args = parse_args()

    default_cfg_path: str = get_default_cfg_path(args.type)
    config: DictConfig = load_config(default_cfg_path, args.config, update_dotlist=args.opts)

    seed_everything(config.General.seed)

    output_dir = '_debug' if args.debug else f'exp_{config.exp_id:04d}'
    output_dir = os.path.join(config.Data.artifact_dir, 'models', output_dir)
    if args.debug:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=False)
    print(f'will save training results under {output_dir}')

    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best',
        save_weights_only=False,
        save_top_k=1,
        monitor='val/iou',
        mode='max',
        save_last=True)
    callbacks = [checkpoint_callback]

    loggers = [TensorBoardLogger(output_dir, name=None)]
    if (not args.debug) and (not args.disable_wandb):
        loggers.append(get_wandb_logger(config))

    trainer = Trainer(
        max_epochs=2 if args.debug else config.General.epochs,
        callbacks=callbacks,
        logger=loggers,
        precision=16 if config.General.fp16 else 32,
        amp_backend=config.General.amp_backend,
        amp_level=config.General.amp_level,
        deterministic=config.General.deterministic,
        benchmark=config.General.benchmark,
        auto_select_gpus=False,
        default_root_dir=os.getcwd(),
        gpus=config.General.gpus,
        limit_train_batches=0.03 if args.debug else 1.0,
        limit_val_batches=0.1 if args.debug else 1.0,
    )

    model = get_model(config)

    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    main()
