from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss


class CombinedLoss(_Loss):
    def __init__(self, config: DictConfig):
        assert len(config.Loss.types) == len(config.Loss.weights)
        assert len(config.Model.classes) == len(config.Loss.class_weights)

        super(CombinedLoss, self).__init__()

        self.loss_weights = config.Loss.weights
        self.loss_names = config.Loss.types
        self.classes = config.Model.classes
        self.class_weights = np.array(config.Loss.class_weights)
        self.class_weights /= np.sum(self.class_weights)

        self.loss_fns = []
        for loss_name in self.loss_names:
            if loss_name == 'dice':
                self.loss_fns.append(smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True))
            elif loss_name == 'bce':
                self.loss_fns.append(smp.losses.SoftBCEWithLogitsLoss())
            else:
                raise ValueError(loss_name)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss = 0
        losses = defaultdict(lambda: 0)
        for loss_fn, loss_weight, loss_name in zip(self.loss_fns, self.loss_weights, self.loss_names):
            for i, (class_name, class_weight) in enumerate(zip(self.classes, self.class_weights)):
                tmp_loss = loss_weight * class_weight * loss_fn(y_pred[:, i], y_true[:, i])
                losses[f'loss/{loss_name} {class_name}'] = tmp_loss.detach().cpu()  # loss-type wise and class wise
                losses[f'loss/{loss_name}'] += tmp_loss.detach().cpu()  # loss-type wise (class aggregated)
                loss += tmp_loss
        return loss, losses
