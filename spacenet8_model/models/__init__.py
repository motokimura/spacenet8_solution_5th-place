import os
from collections import defaultdict

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig

# isort: off
from spacenet8_model.models.losses import Loss
from spacenet8_model.models.seg import SegmentationModel
from spacenet8_model.models.siamese import SiameseModel
from spacenet8_model.utils.misc import get_flatten_classes
# isort: on


def get_model(config: DictConfig, pretrained_exp_id: int = -1) -> torch.nn.Module:
    kwargs = {
        # TODO: map config parameters to kwargs based on the architecture
    }
    model = Model(config, **kwargs)

    if pretrained_exp_id >= 0:
        model = load_pretrained_siamese_branch(model, config, pretrained_exp_id)

    return model


class Model(pl.LightningModule):
    def __init__(self, config, **kwargs):
        assert config.Model.n_input_post_images in [0, 1, 2], config.Model.n_input_post_images
        assert config.Model.type in ['seg', 'siamese'], config.Model.type

        super().__init__()

        if config.Model.type == 'seg':
            self.model = SegmentationModel(config, **kwargs)
        elif config.Model.type == 'siamese':
            self.model = SiameseModel(config, **kwargs)

        # model parameters to preprocess input image
        params = smp.encoders.get_preprocessing_params(config.Model.encoder)
        self.register_buffer('std',
                             torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean',
                             torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.loss_fn = Loss(config)
        self.config = config

    def forward(self, image, image_post_a=None, image_post_b=None):
        image = self.preprocess_images(image, image_post_a, image_post_b)
        mask = self.model(**image)
        return mask

    def preprocess_images(self, image, image_post_a=None, image_post_b=None):
        n_input_post_images = self.config.Model.n_input_post_images
        # check
        if n_input_post_images == 0:
            assert image_post_a is None
            assert image_post_b is None
        elif n_input_post_images == 1:
            assert image_post_a is not None
            assert image_post_b is None
        elif n_input_post_images == 2:
            assert image_post_a is not None
            assert image_post_b is not None

        # preprocess
        image = (image - self.mean) / self.std
        if n_input_post_images == 1:
            image_post_a = (image_post_a - self.mean) / self.std
        elif n_input_post_images == 2:
            image_post_a = (image_post_a - self.mean) / self.std
            image_post_b = (image_post_b - self.mean) / self.std

        if self.config.Model.type == 'seg':
            if n_input_post_images == 1:
                image = torch.cat([image, image_post_a], axis=1)
            elif n_input_post_images == 2:
                image = torch.cat([image, image_post_a, image_post_b], axis=1)
            return {'image': image}
        elif self.config.Model.type == 'siamese':
            if n_input_post_images == 1:
                images_post = [image_post_a]
            elif n_input_post_images == 2:
                images_post = [image_post_a, image_post_b]
            return {'image': image, 'images_post': images_post}

    def shared_step(self, batch, split):
        image = batch['image']
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        n_input_post_images = self.config.Model.n_input_post_images
        if n_input_post_images == 0:
            image_post_a = None
            image_post_b = None
        if n_input_post_images == 1:
            image_post_a = batch['image_post_a']
            image_post_b = None
        elif n_input_post_images == 2:
            image_post_a = batch['image_post_a']
            image_post_b = batch['image_post_b']

        mask = batch['mask']
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image, image_post_a, image_post_b)
        loss, losses = self.loss_fn(logits_mask, mask)

        thresh = 0.5
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > thresh).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode='multilabel')

        metrics = {
            'loss': loss,
            'losses': losses,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

        return metrics

    def shared_epoch_end(self, outputs, split):
        # loss
        losses = self.aggregate_loss(outputs, prefix=f'{split}/loss')
        # iou
        ious = self.aggregate_iou(outputs, reduction='macro', prefix=f'{split}/iou')
        ious_iw = self.aggregate_iou(outputs, reduction='macro-imagewise', prefix=f'{split}/iou_imagewise')

        metrics = {}
        metrics.update(losses)
        metrics.update(ious)
        metrics.update(ious_iw)

        self.log_dict(metrics, prog_bar=True)

    def aggregate_loss(self, outputs, prefix):
        # aggregate step losses to compute loss
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        losses = defaultdict(lambda: [])
        for x in outputs:
            for k, v in x['losses'].items():
                losses[f'{prefix}/{k}'].append(v)
        for k in losses:
            losses[k] = torch.stack(losses[k]).mean()

        losses[prefix] = loss

        return losses

    def aggregate_iou(self, outputs, reduction, prefix):
        # aggregate step metics to compute iou score
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        ious = {}
        for i, class_name in enumerate(get_flatten_classes(self.config)):
            ious[f'{prefix}/{class_name}'] = smp.metrics.iou_score(
                tp[:, i], fp[:, i], fn[:, i], tn[:, i], reduction=reduction)
        iou = torch.stack([v for v in ious.values()]).mean()

        ious[prefix] = iou

        return ious

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'val')

    def configure_optimizers(self):
        config = self.config

        # optimizer
        if config.Optimizer.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                lr=config.Optimizer.lr, weight_decay=config.Optimizer.weight_decay)
        else:
            raise ValueError(config.Optimizer.type)

        # lr scheduler
        if config.Scheduler.type == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                config.Scheduler.multistep_milestones, config.Scheduler.multistep_gamma)
        elif config.Scheduler.type == 'annealing':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.Scheduler.annealing_t_max,
                eta_min=config.Scheduler.annealing_eta_min,
            )
        else:
            raise ValueError(config.Scheduler.type)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'epoch', 'name': 'lr'}
        }


def load_pretrained_siamese_branch(model, config, pretrained_exp_id):
    assert config.Model.type == 'siamese', config.Model.type

    ckpt_path = os.path.join('/wdata/models', f'exp_{pretrained_exp_id:05d}/best.ckpt')
    assert os.path.exists(ckpt_path), ckpt_path
    print(f'loading {ckpt_path}')

    state_dict_orig = torch.load(ckpt_path)['state_dict']

    state_dict = state_dict_orig.copy()
    for k in state_dict_orig:
        if not k.startswith('model.'):
            continue
        if k.startswith('model.segmentation_head.'):
            del state_dict[k]
            continue
        state_dict[k.replace('model.', 'model.branch.')] = state_dict_orig[k]
        del state_dict[k]

    imcompatible_keys = model.load_state_dict(state_dict, strict=False)
    assert len(imcompatible_keys.unexpected_keys) == 0, len(imcompatible_keys.unexpected_keys)

    expected_missing_keys = []
    for i in range(config.Model.n_siamese_head_convs):
        expected_missing_keys.append(f'model.head.{i}.weight')
        expected_missing_keys.append(f'model.head.{i}.bias')
    assert imcompatible_keys.missing_keys == expected_missing_keys, (imcompatible_keys.missing_keys, expected_missing_keys)

    return model
