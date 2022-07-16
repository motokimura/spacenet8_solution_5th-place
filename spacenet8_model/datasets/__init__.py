from omegaconf import DictConfig
from torch.utils.data import DataLoader

# isort: off
from spacenet8_model.datasets.spacenet8 import SpaceNet8Dataset
from spacenet8_model.datasets.transforms import get_transforms
# isort: on


def get_dataloader(config: DictConfig, is_train: bool) -> DataLoader:
    transforms = get_transforms(config, is_train)

    if is_train:
        batch_size = config.Dataloader.train_batch_size
        num_workers = config.Dataloader.train_num_workers
        shuffle = True
    else:
        batch_size = config.Dataloader.val_batch_size
        num_workers = config.Dataloader.val_num_workers
        shuffle = False

    dataset = SpaceNet8Dataset(config, is_train, transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
