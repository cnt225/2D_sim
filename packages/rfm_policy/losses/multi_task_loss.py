"""
Multi-Task Loss for SE(3) Riemannian Flow Matching

This module implements the combined multi-task loss function for SE(3) RFM models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from .flow_matching_loss import FlowMatchingLoss
from .collision_loss import CollisionLoss
from .regularization_loss import RegularizationLoss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining flow matching, collision avoidance, and regularization
    
    Combines multiple loss components with configurable weights:
    - Flow matching loss: main learning objective
    - Collision loss: obstacle avoidance
    - Regularization loss: trajectory smoothness
    """
    
    def __init__(
        self,
        flow_matching_weight: float = 1.0,
        collision_weight: float = 0.1,
        regularization_weight: float = 0.01,
        **loss_kwargs
    ):
        super().__init__()
        
        self.flow_matching_weight = flow_matching_weight
        self.collision_weight = collision_weight
        self.regularization_weight = regularization_weight
        
        # Initialize individual loss components
        self.flow_matching_loss = FlowMatchingLoss(**loss_kwargs.get('flow_matching', {}))
        self.collision_loss = CollisionLoss(**loss_kwargs.get('collision', {}))
        self.regularization_loss = RegularizationLoss(**loss_kwargs.get('regularization', {}))
    
    def forward(
        self,
        predicted_velocity: torch.Tensor,
        start_poses: torch.Tensor,
        goal_poses: torch.Tensor,
        time_steps: torch.Tensor,
        point_clouds: torch.Tensor,
        geometries: torch.Tensor,
        model: nn.Module,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Compute combined multi-task loss
        
        Args:
            predicted_velocity: Predicted velocity from model
            start_poses: Start poses for interpolation
            goal_poses: Goal poses for interpolation
            time_steps: Time steps for interpolation
            point_clouds: Point cloud environments
            geometries: Robot geometries
            model: RFM model
            return_details: Whether to return detailed loss components
            
        Returns:
            loss: Combined multi-task loss
        """
        # Flow matching loss (main objective)
        fm_loss = self.flow_matching_loss(
            predicted_velocity, start_poses, goal_poses, time_steps,
            point_clouds, geometries, model
        )
        
        # Collision loss (obstacle avoidance)
        collision_loss = self.collision_loss(
            start_poses, point_clouds, geometries
        )
        
        # Regularization loss (smoothness)
        reg_loss = self.regularization_loss(
            predicted_velocity, time_steps
        )
        
        # Combine losses with weights
        total_loss = (
            self.flow_matching_weight * fm_loss +
            self.collision_weight * collision_loss +
            self.regularization_weight * reg_loss
        )
        
        if return_details:
            details = {
                'total_loss': total_loss,
                'flow_matching_loss': fm_loss,
                'collision_loss': collision_loss,
                'regularization_loss': reg_loss,
                'weights': {
                    'flow_matching': self.flow_matching_weight,
                    'collision': self.collision_weight,
                    'regularization': self.regularization_weight
                }
            }
            return total_loss, details
        
        return total_loss


def compute_velocity_metrics(
    predicted_velocities: torch.Tensor,
    target_velocities: torch.Tensor
) -> Dict[str, float]:
    """
    Compute velocity prediction metrics
    
    Args:
        predicted_velocities: Model predictions [B, 6]
        target_velocities: Ground truth velocities [B, 6]
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    # Split into linear and angular components
    pred_linear = predicted_velocities[:, :3]
    pred_angular = predicted_velocities[:, 3:]
    target_linear = target_velocities[:, :3]
    target_angular = target_velocities[:, 3:]
    
    # MSE losses
    linear_mse = F.mse_loss(pred_linear, target_linear).item()
    angular_mse = F.mse_loss(pred_angular, target_angular).item()
    total_mse = F.mse_loss(predicted_velocities, target_velocities).item()
    
    # L1 losses
    linear_l1 = F.l1_loss(pred_linear, target_linear).item()
    angular_l1 = F.l1_loss(pred_angular, target_angular).item()
    total_l1 = F.l1_loss(predicted_velocities, target_velocities).item()
    
    # Velocity magnitudes
    pred_linear_mag = torch.norm(pred_linear, dim=1).mean().item()
    pred_angular_mag = torch.norm(pred_angular, dim=1).mean().item()
    target_linear_mag = torch.norm(target_linear, dim=1).mean().item()
    target_angular_mag = torch.norm(target_angular, dim=1).mean().item()
    
    return {
        'linear_mse': linear_mse,
        'angular_mse': angular_mse,
        'total_mse': total_mse,
        'linear_l1': linear_l1,
        'angular_l1': angular_l1,
        'total_l1': total_l1,
        'pred_linear_mag': pred_linear_mag,
        'pred_angular_mag': pred_angular_mag,
        'target_linear_mag': target_linear_mag,
        'target_angular_mag': target_angular_mag
    }


def adaptive_loss_weighting(
    losses: Dict[str, torch.Tensor],
    loss_history: Dict[str, list],
    window_size: int = 100,
    adaptation_rate: float = 0.01
) -> Dict[str, float]:
    """
    Adaptive loss weighting based on loss history
    
    Args:
        losses: Current loss values
        loss_history: Historical loss values
        window_size: Window size for averaging
        adaptation_rate: Rate of weight adaptation
        
    Returns:
        weights: Adapted loss weights
    """
    weights = {}
    
    for loss_name, current_loss in losses.items():
        if loss_name in loss_history and len(loss_history[loss_name]) >= window_size:
            # Compute average loss over window
            recent_losses = loss_history[loss_name][-window_size:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            
            # Adapt weight based on loss ratio
            if avg_loss > 0:
                loss_ratio = current_loss / avg_loss
                # Increase weight if loss is high relative to history
                weight_adjustment = 1.0 + adaptation_rate * (loss_ratio - 1.0)
                weights[loss_name] = max(0.1, weight_adjustment)  # Minimum weight
            else:
                weights[loss_name] = 1.0
        else:
            weights[loss_name] = 1.0
    
    return weights 