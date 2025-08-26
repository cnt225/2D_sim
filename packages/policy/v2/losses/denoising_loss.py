import torch.nn as nn


class ProjectedSE3DenoisingLoss:
    def __init__(self, delta=1, *args, **kwargs):
        self.delta = delta
        self.loss = nn.L1Loss()

    def __call__(self, grad_energy, z_target):
        loss = self.loss(grad_energy, z_target) / 10

        return loss
