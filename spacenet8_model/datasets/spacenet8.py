import os

import numpy as np
import pandas as pd
import torch
from skimage import io


class SpaceNet8Dataset(torch.utils.data.Dataset):
    def __init__(self, config, is_train, transform=None):
        self.pre_paths, self.post1_paths, self.post2_paths, self.mask_paths = self.get_file_paths(config, is_train)
        self.masks_to_load = self.get_mask_types_to_load(config)

        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = io.imread(self.pre_paths[idx])  # HWC
        h, w = pre.shape[:2]
        
        mask = self.load_mask(idx)  # HWC
        h_mask, w_mask = mask.shape[:2]
        assert h_mask == h
        assert w_mask == w

        sample = dict(image=pre, mask=mask)

        post_images = self.load_post_images(idx)
        for k in post_images:
            h_post, w_post = post_images[k].shape[:2]
            assert h == h_post
            assert w == w_post
        sample.update(post_images)

        if self.transform is not None:
            sample = self.transform(**sample)

        # convert format HWC -> CHW
        sample['image'] = np.moveaxis(sample['image'], -1, 0)
        sample['mask'] = np.moveaxis(sample['mask'], -1, 0)
        for k in post_images:
            sample[k] = np.moveaxis(sample[k], -1, 0)

        return sample

    def get_file_paths(self, config, is_train):
        split = 'train' if is_train else 'val'
        path = os.path.join(config.Data.artifact_dir,
                            'folds',
                            f'{split}_{config.fold_id}.csv')
        df = pd.read_csv(path)

        # prepare image and mask paths
        pre_paths = []
        post1_paths = []
        post2_paths = []
        building_3channel_paths = []
        building_flood_paths = []
        road_paths = []
        for i, row in df.iterrows():
            aoi = row['aoi']

            pre = os.path.join(config.Data.train_dir, aoi, 'PRE-event', row['pre-event image'])
            assert os.path.exists(pre), pre
            pre_paths.append(pre)

            post1 = os.path.join(config.Data.artifact_dir, 'warped_posts_train', aoi, row['post-event image 1'])
            assert os.path.exists(post1), post1
            post1_paths.append(post1)

            post2 = row['post-event image 2']
            if isinstance(post2, str):
                post2 = os.path.join(config.Data.artifact_dir, 'warped_posts_train', aoi, post2)
                assert os.path.exists(post2), post2
            else:
                post2 = None
            post2_paths.append(post2)

            mask_filename, _ = os.path.splitext(row['pre-event image'])
            mask_filename = f'{mask_filename}.png'

            building_3channel = os.path.join(config.Data.artifact_dir, 'masks_building_3channel', aoi, mask_filename)
            assert os.path.exists(building_3channel), building_3channel
            building_3channel_paths.append(building_3channel)

            building_flood = os.path.join(config.Data.artifact_dir, 'masks_building_flood', aoi, mask_filename)
            assert os.path.exists(building_flood), building_flood
            building_flood_paths.append(building_flood)

            road = os.path.join(config.Data.artifact_dir, 'masks_road', aoi, mask_filename)
            assert os.path.exists(road), road
            road_paths.append(road)

        mask_paths = {
            'building_3channel': building_3channel_paths,
            'building_flood': building_flood_paths,
            'road': road_paths,
        }
        return pre_paths, post1_paths, post2_paths, mask_paths

    def get_mask_types_to_load(self, config):
        mask_types = []
        cs = config.Model.classes

        if ('building' in cs) or ('building_border' in cs) or ('building_contact' in cs):
            mask_types.append('building_3channel')

        if ('flood' in cs) or ('flood_building' in cs) or ('not_flood_building' in cs):
            mask_types.append('building_flood')

        if ('road' in cs) or ('road_junction' in cs) or ('flood' in cs) or ('flood_road' in cs) or ('not_flood_road' in cs):
            mask_types.append('road')

        assert len(mask_types) > 0
        return mask_types

    def load_mask(self, idx):
        masks = {}
        for mask_type in self.masks_to_load:
            masks[mask_type] = io.imread(self.mask_paths[mask_type][idx])
        target_mask = self.prepare_target_mask(masks)
        return target_mask

    def prepare_target_mask(self, masks):
        target_mask = []
        for c in self.config.Model.classes:
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
            elif c == 'flood_building':
                target_mask.append((masks['building_flood'][:, :, 0] > 0).astype(np.float32))
            elif c == 'flood_road':
                target_mask.append((masks['road'][:, :, 0] > 0).astype(np.float32))
            elif c == 'not_flood_building':
                target_mask.append((masks['building_flood'][:, :, 1] > 0).astype(np.float32))
            elif c == 'not_flood_road':
                target_mask.append((masks['road'][:, :, 1] > 0).astype(np.float32))
            else:
                raise ValueError(c)
        target_mask = np.stack(target_mask)  # CHW
        return target_mask.transpose(1, 2, 0)  # HWC

    def load_post_images(self, idx):
        return load_post_images(
            self.post1_paths[idx],
            self.post2_paths[idx],
            self.config.Model.n_input_post_images
        )


class SpaceNet8TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None, test_to_val=False):
        self.pre_paths, self.post1_paths, self.post2_paths = self.get_file_paths(config, test_to_val)

        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = io.imread(self.pre_paths[idx])  # HWC
        h, w = pre.shape[:2]
        sample = dict(image=pre)

        post_images = self.load_post_images(idx)
        for k in post_images:
            h_post, w_post = post_images[k].shape[:2]
            assert h == h_post
            assert w == w_post
        sample.update(post_images)

        if self.transform is not None:
            sample = self.transform(**sample)

        # convert format HWC -> CHW
        sample['image'] = np.moveaxis(sample['image'], -1, 0)
        for k in post_images:
            sample[k] = np.moveaxis(sample[k], -1, 0)

        # add image metadata
        meta = {
            'pre_path': self.pre_paths[idx],
            'original_height': h,
            'original_width': w
        }
        sample.update(meta)

        return sample

    def get_file_paths(self, config, test_to_val=False):
        if test_to_val:
            csv_path = os.path.join(config.Data.artifact_dir, f'folds/val_{config.fold_id}.csv')
            data_root = config.Data.train_dir
            post_image_dir = os.path.join(config.Data.artifact_dir, 'warped_posts_train')
        else:
            csv_path = os.path.join(config.Data.artifact_dir, 'test.csv')
            data_root = config.Data.test_dir
            post_image_dir = os.path.join(config.Data.artifact_dir, 'warped_posts_test')
        df = pd.read_csv(csv_path)

        # prepare image paths
        pre_paths = []
        post1_paths = []
        post2_paths = []
        for i, row in df.iterrows():
            aoi = row['aoi']

            pre = os.path.join(data_root, aoi, 'PRE-event', row['pre-event image'])
            assert os.path.exists(pre), pre
            pre_paths.append(pre)

            post1 = os.path.join(post_image_dir, aoi, row['post-event image 1'])
            assert os.path.exists(post1), post1
            post1_paths.append(post1)

            post2 = row['post-event image 2']
            if isinstance(post2, str):
                post2 = os.path.join(post_image_dir, aoi, post2)
                assert os.path.exists(post2), post2
            else:
                post2 = None
            post2_paths.append(post2)

        return pre_paths, post1_paths, post2_paths

    def load_post_images(self, idx):
        return load_post_images(
            self.post1_paths[idx],
            self.post2_paths[idx],
            self.config.Model.n_input_post_images
        )


def load_post_images(post1_path, post2_path, n_input_post_images):
    if n_input_post_images == 0:
        return {}

    elif n_input_post_images == 1:
        post1 = io.imread(post1_path)
        return {
            'image_post_a': post1
        }

    elif n_input_post_images == 2:
        post1 = io.imread(post1_path)
        if post2_path is None:
            # if post-2 image does not exist, just copy post-1 as post-2 image
            post2 = post1.copy()
        else:
            post2 = io.imread(post2_path)
        return {
            'image_post_a': post1,
            'image_post_b': post2
        }

    else:
        raise ValueError(n_input_post_images)
