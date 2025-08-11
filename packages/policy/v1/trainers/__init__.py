from .logger import BaseLogger
from .base import BaseTrainer
from .grasp_trainer import GraspTrainer
from .motion_trainer import MotionTrainer


def get_trainer(cfg):
    trainer_type = cfg.get('trainer', None)
    device = cfg['device']
    if trainer_type == 'base':
        trainer = BaseTrainer(cfg['training'], device=device)
    elif trainer_type == 'grasp':
        trainer = GraspTrainer(cfg['training'], device=device)
    elif trainer_type == 'motion':
        trainer = MotionTrainer(cfg['training'], device=device)
    else:
        raise NotImplementedError(f"Trainer {trainer_type} is not implemented")

    return trainer


def get_logger(cfg, writer, ddp=False):
    endwith = cfg['logger'].get('endwith', [])
    logger = BaseLogger(writer, endwith=endwith, ddp=ddp)

    return logger