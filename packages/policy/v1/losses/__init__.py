from .sdf_loss import SDFLoss
from .denoising_loss import ProjectedSE3DenoisingLoss
from .control_point_loss import ControlPointLoss
from .kl_loss import KLLoss
from .min_distance_loss import MinDistanceLoss
from .mse_loss import MSELoss


def get_loss(cfg_loss):
    name = cfg_loss['arch'] if 'arch' in cfg_loss else cfg_loss['name']

    if name == 'sdf_loss':
        loss = SDFLoss(**cfg_loss)
    elif name == 'denoising_loss':
        loss = ProjectedSE3DenoisingLoss(**cfg_loss)
    elif name == 'control_point_loss':
        loss = ControlPointLoss(**cfg_loss)
    elif name == 'kl_loss':
        loss = KLLoss(**cfg_loss)
    elif name == 'min_distance_loss':
        loss = MinDistanceLoss(**cfg_loss)
    elif name == 'mse':
        loss = MSELoss(**cfg_loss)
    else:
        raise NotImplementedError(f"Loss {name} is not implemented")

    return loss
