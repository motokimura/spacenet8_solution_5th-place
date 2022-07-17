import os
from typing import Optional

import dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


def get_wandb_logger(config: DictConfig,
                     group: Optional[str] = None) -> WandbLogger:
    dotenv.load_dotenv()  # load WANDB_API_KEY from .env file
    assert 'WANDB_API_KEY' in os.environ, \
        ('"WANDB_API_KEY" is empty. '
         'Create ".env" file with your W&B API key. '
         'See ".env.sample" for the file format')

    if group is None:
        group = f'exp_{config.exp_id:04d}'

    return WandbLogger(
        project=f'sn8-{config.task}',
        group=group,
        config=OmegaConf.to_container(config, resolve=True))
