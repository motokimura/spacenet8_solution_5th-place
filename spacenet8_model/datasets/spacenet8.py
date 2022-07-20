import os

import numpy as np
import pandas as pd
import torch
from skimage import io


class SpaceNet8Dataset(torch.utils.data.Dataset):
    def __init__(self, config, is_train, transform=None):
        self.pre_paths, self.post_paths, self.mask_paths = self.get_file_paths(config, is_train)
        self.masks_to_load = self.get_mask_types_to_load(config)

        self.classes = config.Model.classes
        self.transform = transform

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = io.imread(self.pre_paths[idx])  # HWC

        masks = self.load_masks(idx)
        mask = self.prepare_target_mask(masks)  # HWC

        # TODO: load post image

        sample = dict(image=pre, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        # convert format HWC -> CHW
        sample['image'] = np.moveaxis(sample['image'], -1, 0)
        sample['mask'] = np.moveaxis(sample['mask'], -1, 0)

        return sample

    def get_file_paths(self, config, is_train):
        split = 'train' if is_train else 'val'
        path = os.path.join(config.Data.artifact_dir,
                            'folds',
                            f'{split}_{config.fold_id}.csv')
        df = pd.read_csv(path)

        # prepare image and mask paths
        pre_paths = []
        post_paths = []
        building_3channel_paths = []
        building_flood_paths = []
        road_paths = []
        for i, row in df.iterrows():
            aoi = row['aoi']

            pre = os.path.join(config.Data.train_dir, aoi, 'PRE-event', row['pre-event image'])
            os.path.exists(pre), pre
            pre_paths.append(pre)

            # TODO: post-1 image and post-2 image

            mask_filename, _ = os.path.splitext(row['pre-event image'])
            mask_filename = f'{mask_filename}.png'

            building_3channel = os.path.join(config.Data.artifact_dir, 'masks_building_3channel', aoi, mask_filename)
            os.path.exists(building_3channel), building_3channel
            building_3channel_paths.append(building_3channel)

            building_flood = os.path.join(config.Data.artifact_dir, 'masks_building_flood', aoi, mask_filename)
            os.path.exists(building_flood), building_flood
            building_flood_paths.append(building_flood)

            road = os.path.join(config.Data.artifact_dir, 'masks_road', aoi, mask_filename)
            os.path.exists(road), road
            road_paths.append(road)

        mask_paths = {
            'building_3channel': building_3channel_paths,
            'building_flood': building_flood_paths,
            'road': road_paths,
        }
        return pre_paths, post_paths, mask_paths

    def get_mask_types_to_load(self, config):
        mask_types = []
        cs = config.Model.classes

        if ('building' in cs) or ('building_border' in cs) or ('building_contact' in cs):
            mask_types.append('building_3channel')

        if 'flood' in cs:
            mask_types.append('building_flood')

        if ('road' in cs) or ('road_junction' in cs) or ('flood' in cs):
            mask_types.append('road')

        assert len(mask_types) > 0
        return mask_types

    def load_masks(self, idx):
        masks = {}
        for mask_type in self.masks_to_load:
            masks[mask_type] = io.imread(self.mask_paths[mask_type][idx])
        return masks

    def prepare_target_mask(self, masks):
        target_mask = []
        for c in self.classes:
            # building
            if c == 'building':
                target_mask.append((masks['building_3channel'][:, :, 0] > 0).astype(np.float32))
            elif c == 'building_border':
                target_mask.append((masks['building_3channel'][:, :, 1] > 0).astype(np.float32))
            elif c == 'building_contact':
                target_mask.append((masks['building_3channel'][:, :, 2] > 0).astype(np.float32))
            # road
            elif c == 'road':
                target_mask.append(((masks['road'][:, :, 0] + masks['road'][:, :, 1]) > 0).astype(np.float32))
            elif c == 'road_junction':
                target_mask.append((masks['road'][:, :, 2] > 0).astype(np.float32))
            # flood
            elif c == 'flood':
                target_mask.append(((masks['building_flood'][:, :, 0] + masks['road'][:, :, 0]) > 0).astype(np.float32))
            else:
                raise ValueError(c)
        target_mask = np.stack(target_mask)  # CHW
        return target_mask.transpose(1, 2, 0)  # HWC


class SpaceNet8TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):
        self.pre_paths, self.post_paths, = self.get_file_paths(config)

        self.transform = transform

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = io.imread(self.pre_paths[idx])  # HWC
        h, w = pre.shape[:2]

        sample = dict(image=pre)
        if self.transform is not None:
            sample = self.transform(**sample)

        # convert format HWC -> CHW
        sample['image'] = np.moveaxis(sample['image'], -1, 0)

        # add image metadata
        meta = {
            'pre_path':self.pre_paths[idx],
            'original_height': h,
            'original_width': w
        }
        sample.update(meta)

        return sample

    def get_file_paths(self, config):
        path = os.path.join(config.Data.artifact_dir, 'test.csv')
        df = pd.read_csv(path)

        # prepare image paths
        pre_paths = []
        post_paths = []
        for i, row in df.iterrows():
            aoi = row['aoi']

            pre = os.path.join(config.Data.test_dir, aoi, 'PRE-event', row['pre-event image'])
            os.path.exists(pre), pre
            pre_paths.append(pre)

            # TODO: post-1 image and post-2 image

        return pre_paths, post_paths
