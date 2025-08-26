import torch


class KLLoss:
    def __init__(self, weight_kl, *args, **kwargs):
        self.weight_kl = weight_kl

    def __call__(self, mu, log_sigma):
        loss_kl = (-0.5 * torch.sum(1 + log_sigma - mu ** 2 - torch.exp(log_sigma), dim=-1)).mean()

        return loss_kl