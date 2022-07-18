from collections import defaultdict

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig

# isort: off
from spacenet8_model.models.losses import CombinedLoss
# isort: on


def get_model(config: DictConfig) -> torch.nn.Module:
    kwargs = {
        # TODO: map config parameters to kwargs based on the architecture
    }
    return Model(config, **kwargs)


class Model(pl.LightningModule):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            config.Model.arch,
            encoder_name=config.Model.encoder,
            in_channels=3,
            classes=len(config.Model.classes),
            encoder_weights="imagenet",
            **kwargs)

        # model parameters to preprocess input image
        params = smp.encoders.get_preprocessing_params(config.Model.encoder)
        self.register_buffer('std',
                             torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean',
                             torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.loss_fn = CombinedLoss(config)
        self.config = config

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, split):

        image = batch['image']
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
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
        prefix = f'{split}'
        # loss
        losses = self.aggregate_loss(outputs, prefix=prefix)
        # iou
        ious = self.aggregate_iou(outputs, reduction='macro', prefix=f'{prefix}/iou')
        ious_iw = self.aggregate_iou(outputs, reduction='macro-imagewise', prefix=f'{prefix}/iou_imagewise')

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

        losses[f'{prefix}/loss'] = loss

        return losses

    def aggregate_iou(self, outputs, reduction, prefix):
        # aggregate step metics to compute iou score
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        ious = {}
        for i, class_name in enumerate(self.config.Model.classes):
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
        else:
            raise ValueError(config.Scheduler.type)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'epoch', 'name': 'lr'}
        }
