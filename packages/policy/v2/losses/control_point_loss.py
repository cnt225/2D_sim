import numpy as np
import torch

from utils.utils import transform_control_points


class ControlPointLoss:
    def __init__(self, weight_confidence, *args, **kwargs):
        control_pts = torch.from_numpy(np.load('assets/control_points/control_points.npy'))[:, :3]
        self.control_pts = torch.cat([torch.zeros(2, 3), control_pts[0:2], control_pts[-2:]])

        self.loss_control_point = torch.nn.L1Loss(reduction='none')

        self.weight_confidence = weight_confidence

    def __call__(self, Ts_pred, Ts_target, confidence):
        control_pts_pred = transform_control_points(Ts_pred, self.control_pts)
        control_pts_target = transform_control_points(Ts_target, self.control_pts)

        error = control_pts_pred - control_pts_target
        error = error.abs().sum(dim=2)

        loss_control_point = (error.mean(dim=1) * confidence).mean()
        loss_confidence = (- torch.log(torch.max(confidence, torch.tensor(1e-10).to(confidence)))).mean()

        return loss_control_point, loss_confidence
