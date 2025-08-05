"""
Loss Functions for SE(3) Riemannian Flow Matching

This module provides various loss functions for training SE(3) RFM models.
"""

from .flow_matching_loss import FlowMatchingLoss
from .collision_loss import CollisionLoss
from .regularization_loss import RegularizationLoss
from .multi_task_loss import MultiTaskLoss, compute_velocity_metrics, adaptive_loss_weighting


def get_loss(cfg_loss):
    """Get loss function based on configuration"""
    name = cfg_loss.get('name', 'multi_task')
    
    if name == 'flow_matching':
        loss = FlowMatchingLoss(**cfg_loss)
    elif name == 'collision':
        loss = CollisionLoss(**cfg_loss)
    elif name == 'regularization':
        loss = RegularizationLoss(**cfg_loss)
    elif name == 'multi_task':
        loss = MultiTaskLoss(**cfg_loss)
    else:
        raise NotImplementedError(f"Loss {name} is not implemented")
    
    return loss


__all__ = [
    'FlowMatchingLoss',
    'CollisionLoss', 
    'RegularizationLoss',
    'MultiTaskLoss',
    'compute_velocity_metrics',
    'adaptive_loss_weighting',
    'get_loss',
] 