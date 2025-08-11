import torch.nn as nn
import torch


class SDFLoss:
    def __init__(self, delta=0.6, *args, **kwargs):
        self.delta = delta
        self.loss = nn.L1Loss(reduction='mean')

    def __call__(self, pred, target):
        pred_clipped = torch.clip(pred, -10, self.delta)
        target_clipped = torch.clip(target, -10, self.delta)

        loss = self.loss(pred_clipped, target_clipped)

        return loss
