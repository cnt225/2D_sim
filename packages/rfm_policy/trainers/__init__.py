"""
Trainers for SE(3) Riemannian Flow Matching

This module provides training infrastructure for SE(3) RFM models.
"""

from .optimizers import get_optimizer
from .schedulers import get_scheduler


def get_trainer(cfg):
    """Get trainer based on configuration"""
    trainer_type = cfg.get('trainer', 'se3_rfm')
    device = cfg.get('device', 'cuda')
    
    if trainer_type == 'se3_rfm':
        from .se3_rfm_trainer import SE3RFMTrainer
        trainer = SE3RFMTrainer(cfg, device=device)
    else:
        raise NotImplementedError(f"Trainer {trainer_type} is not implemented")
    
    return trainer


def get_logger(cfg, writer, ddp=False):
    """Get logger based on configuration"""
    from .logger import BaseLogger
    endwith = cfg.get('logger', {}).get('endwith', [])
    logger = BaseLogger(writer, endwith=endwith, ddp=ddp)
    return logger


__all__ = [
    'get_trainer',
    'get_logger',
    'get_optimizer',
    'get_scheduler',
] 