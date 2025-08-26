import numpy as np
import torch

from utils.utils import transform_control_points


class MinDistanceLoss:
    def __init__(self, weight_confidence, *args, **kwargs):
        control_pts = torch.from_numpy(np.load('assets/control_points/control_points.npy'))[:, :3]
        self.control_pts = torch.cat([torch.zeros(2, 3), control_pts[0:2], control_pts[-2:]])

        self.weight_confidence = weight_confidence

    def __call__(self, Ts_pred, Ts_target, confidence):
        control_pts_pred = transform_control_points(Ts_pred, self.control_pts)
        control_pts_target = transform_control_points(Ts_target, self.control_pts)

        error = control_pts_pred[:, None] - control_pts_target[None]
        error = error.abs().sum(dim=3).mean(dim=2)

        error_min_distance, idx_closest = error.min(dim=0)

        confidence_selected = torch.nn.functional.one_hot(idx_closest, num_classes=len(idx_closest)).float()
        confidence_selected *= confidence
        confidence_selected = confidence_selected.sum(dim=1)

        error_min_distance *= confidence_selected

        loss_control_point = error_min_distance.mean()
        loss_confidence = - torch.log(torch.max(confidence, torch.tensor(1e-4))).mean()

        return loss_control_point, loss_confidence
