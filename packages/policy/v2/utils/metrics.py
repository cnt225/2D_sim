# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch

from utils.Lie import inv_SO3, log_SO3, bracket_so3, SE3_geodesic
from scipy.optimize import linear_sum_assignment


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_cls(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)


        return {"Overall Acc : \t": acc,
                "Mean Acc : \t": acc_cls,}


    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MMDCalculator:
    def __init__(self, type, num_episodes=100, kernel_mul=1, kernel_num=1, bandwidth_base=None):
        self.num_episodes = num_episodes
        self.type = type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.bandwidth_base = bandwidth_base

    def calculate_squared_distance(self, x, y):
        if self.type == 'SE3':
            batch_size = len(x)

            T_x = x.reshape(-1, 4, 4)
            T_y = y.reshape(-1, 4, 4)

            dist_R = bracket_so3(log_SO3(torch.einsum('bij,bjk->bik', inv_SO3(T_x[:, :3, :3]), T_y[:, :3, :3])))
            dist_p = T_x[:, :3, 3] - T_y[:, :3, 3]

            return torch.sum(dist_R ** 2 + dist_p ** 2, dim=1).reshape(batch_size, batch_size)
        elif self.type == 'L2':
            return ((x - y)**2).sum(dim=2)
        else:
            raise NotImplementedError(f"Type {self.type} is not implemented. Choose type between 'SE3' and 'L2'.")

    def guassian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        total0 = torch.repeat_interleave(total.unsqueeze(1), len(total), dim=1)
        total1 = torch.repeat_interleave(total.unsqueeze(0), len(total), dim=0)

        distance_squared = self.calculate_squared_distance(total0, total1)

        if self.bandwidth_base == None:
            self.bandwidth_base = torch.sum(distance_squared) / (len(total) ** 2 - len(total))

        self.bandwidth_base /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [self.bandwidth_base * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-distance_squared / bandwidth) for bandwidth in bandwidth_list]

        return sum(kernel_val)

    def __call__(self, source, target):
        assert len(source) <= len(target), f"The number of samples in source {len(source)} must be less than or equal to the number of samples in target {len(target)}."

        batch_size = len(source)

        mmd_list = []

        for _ in range(self.num_episodes):
            target_ = target[np.random.choice(range(len(target)), len(source), replace=False)]

            kernels = self.guassian_kernel(source, target_)

            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]

            mmd = torch.mean(XX + YY - XY - YX).item()

            mmd_list += [mmd]

        mmd_avg = sum(mmd_list) / len(mmd_list)

        return mmd_avg


class EMDCalculator:
    def __init__(self, type, num_episodes=100):
        self.num_episodes = num_episodes
        self.type = type

    def calculate_distance(self, x, y):
        if self.type == 'SE3':
            p_x = x[:, :3, 3]
            p_y = y[:, :3, 3]

            R_x = x[:, :3, :3]
            R_y = y[:, :3, :3]

            dist_p = (((p_x.unsqueeze(1) - p_y.unsqueeze(0)) ** 2).sum(-1)).sqrt()

            R_xy = torch.einsum('bij,cjk->bcik', inv_SO3(R_x), R_y)

            R_xy_ = R_xy.reshape(-1, 3, 3)
            dist_theta_ = torch.linalg.norm(bracket_so3(log_SO3(R_xy_)), dim=1)
            dist_R_ = (1 - torch.cos(dist_theta_))
            dist_R = dist_R_.reshape(R_xy.shape[0:2])

            return  dist_p + dist_R
        elif self.type == 'L2':
            xx = torch.repeat_interleave(x.unsqueeze(1), y.shape[0], dim = 1)
            yy = torch.repeat_interleave(y.unsqueeze(0), x.shape[0], dim = 0)

            return torch.sqrt(((xx - yy)**2).sum(dim=2))
        else:
            raise NotImplementedError(f"Type {self.type} is not implemented. Choose type between 'SE3' and 'L2'.")

    def __call__(self, source, target):
        assert len(source) <= len(target), "The number of samples in source must be less than or equal to the number of samples in target."

        emd_list = []

        for _ in range(self.num_episodes):
            target_ = target[np.random.choice(range(len(target)), len(source), replace=False)]

            distance = self.calculate_distance(source, target_).cpu().numpy()

            idxs_row, idxs_col = linear_sum_assignment(distance)

            emd = distance[idxs_row, idxs_col].mean()

            emd_list += [emd]

        emd_avg = sum(emd_list) / len(emd_list)

        return emd_avg
