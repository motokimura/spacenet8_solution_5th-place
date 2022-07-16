from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(default_cfg_path: str,
                cfg_path: Optional[str] = None,
                update_dotlist: Optional[List[str]] = None) -> DictConfig:

    config = OmegaConf.load(default_cfg_path)
    if cfg_path is not None:
        optional_config = OmegaConf.load(cfg_path)
        config = OmegaConf.merge(config, optional_config)
    if update_dotlist is not None:
        update_config = OmegaConf.from_dotlist(update_dotlist)
        config = OmegaConf.merge(config, update_config)

    OmegaConf.set_readonly(config, True)

    return config


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
