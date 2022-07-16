import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig


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

        # TODO: configure loss functions from config
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.MULTILABEL_MODE, from_logits=True)

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
        loss = self.loss_fn(logits_mask, mask)

        thresh = 0.5
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > thresh).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode='multilabel')

        return {
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def shared_epoch_end(self, outputs, split):
        # aggregate step metics
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction='macro')

        iou_imagewise = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction='macro-imagewise')

        # TODO: compute classwise iou
        # TODO: add loss to metrics

        metrics = {
            f'{split}/iou': iou,
            f'{split}/iou_imagewise': iou_imagewise,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'val')

    def configure_optimizers(self):
        # TODO: configure optimizer and lr scheduler from config
        return torch.optim.Adam(self.parameters(), lr=0.0001)
