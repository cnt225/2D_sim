"""
Flow Matching Loss for SE(3) Riemannian Flow Matching

This module implements the core flow matching loss for SE(3) RFM models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import numpy as np

from ..utils.se3_utils import SE3Utils


class FlowMatchingLoss(nn.Module):
    """
    Core flow matching loss for SE(3) Riemannian Flow Matching
    
    Implements the conditional flow matching objective:
    L = E_{t,x_0,x_1}[||v_θ(x_t, t, c) - u_t(x_t|x_0,x_1)||^2]
    
    where:
    - v_θ is the learned velocity field
    - u_t is the target velocity field (geodesic velocity)
    - x_t is interpolated pose at time t
    - c is the conditioning (target pose, environment, geometry)
    """
    
    def __init__(
        self,
        se3_utils: Optional[SE3Utils] = None,
        prob_path: str = 'OT',
        noise_scale: float = 0.01,
        weight_linear: float = 1.0,
        weight_angular: float = 1.0,
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.se3_utils = se3_utils or SE3Utils()
        self.prob_path = prob_path
        self.noise_scale = noise_scale
        self.weight_linear = weight_linear
        self.weight_angular = weight_angular
        self.eps = eps
        
        if prob_path not in ['OT', 'OT_CFM']:
            raise ValueError(f"Unsupported prob_path: {prob_path}")
    
    def forward(
        self,
        predicted_velocity: torch.Tensor,  # [B, 6] predicted twist
        start_poses: torch.Tensor,         # [B, 4, 4] start SE(3) poses
        goal_poses: torch.Tensor,          # [B, 4, 4] goal SE(3) poses
        time_steps: torch.Tensor,          # [B, 1] time steps
        point_clouds: torch.Tensor,        # [B, N, 3] point clouds
        geometries: torch.Tensor,          # [B, 3] robot geometries
        model: nn.Module,                  # RFM model for computing predicted velocity
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Compute flow matching loss
        
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
            loss: Flow matching loss
        """
        batch_size = start_poses.shape[0]
        device = start_poses.device
        
        # Sample interpolated poses and compute target velocities
        if self.prob_path == 'OT':
            # Optimal transport path (geodesic)
            interpolated_poses, target_velocities = self._compute_ot_path(
                start_poses, goal_poses, time_steps
            )
        elif self.prob_path == 'OT_CFM':
            # Conditional flow matching with noise
            interpolated_poses, target_velocities = self._compute_ot_cfm_path(
                start_poses, goal_poses, time_steps
            )
        
        # Compute predicted velocities at interpolated poses
        predicted_velocities = model(
            interpolated_poses, goal_poses, point_clouds, geometries, time_steps
        )
        
        # Compute MSE loss with separate weights for linear/angular components
        loss = self._compute_weighted_mse(predicted_velocities, target_velocities)
        
        if return_details:
            details = self._compute_loss_details(
                predicted_velocities, target_velocities, time_steps
            )
            return loss, details
        
        return loss
    
    def _compute_ot_path(
        self,
        start_poses: torch.Tensor,
        goal_poses: torch.Tensor,
        time_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal transport path (geodesic interpolation)"""
        # Geodesic interpolation on SE(3)
        interpolated_poses = self.se3_utils.geodesic_interpolation(
            start_poses, goal_poses, time_steps
        )
        
        # Compute geodesic velocity (constant velocity along geodesic)
        target_velocities = self.se3_utils.compute_geodesic_velocity(
            start_poses, goal_poses
        )
        
        return interpolated_poses, target_velocities
    
    def _compute_ot_cfm_path(
        self,
        start_poses: torch.Tensor,
        goal_poses: torch.Tensor,
        time_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute conditional flow matching path with noise"""
        batch_size = start_poses.shape[0]
        device = start_poses.device
        
        # Add noise to start poses
        noise = torch.randn_like(start_poses) * self.noise_scale
        noisy_start_poses = start_poses + noise
        
        # Linear interpolation with noise
        interpolated_poses = (1 - time_steps) * noisy_start_poses + time_steps * goal_poses
        
        # Target velocity is constant (linear path)
        target_velocities = goal_poses - noisy_start_poses
        
        return interpolated_poses, target_velocities
    
    def _compute_weighted_mse(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted MSE loss separating linear and angular components"""
        # Split into linear (first 3) and angular (last 3) components
        pred_linear = predicted[:, :3]
        pred_angular = predicted[:, 3:]
        target_linear = target[:, :3]
        target_angular = target[:, 3:]
        
        # Compute separate losses
        linear_loss = F.mse_loss(pred_linear, target_linear)
        angular_loss = F.mse_loss(pred_angular, target_angular)
        
        # Weighted combination
        total_loss = (
            self.weight_linear * linear_loss + 
            self.weight_angular * angular_loss
        )
        
        return total_loss
    
    def _compute_loss_details(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        time_steps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute detailed loss components for monitoring"""
        # Split components
        pred_linear = predicted[:, :3]
        pred_angular = predicted[:, 3:]
        target_linear = target[:, :3]
        target_angular = target[:, 3:]
        
        # Individual losses
        linear_loss = F.mse_loss(pred_linear, target_linear)
        angular_loss = F.mse_loss(pred_angular, target_angular)
        
        # Velocity magnitudes
        pred_linear_mag = torch.norm(pred_linear, dim=1).mean()
        pred_angular_mag = torch.norm(pred_angular, dim=1).mean()
        target_linear_mag = torch.norm(target_linear, dim=1).mean()
        target_angular_mag = torch.norm(target_angular, dim=1).mean()
        
        return {
            'linear_loss': linear_loss,
            'angular_loss': angular_loss,
            'pred_linear_mag': pred_linear_mag,
            'pred_angular_mag': pred_angular_mag,
            'target_linear_mag': target_linear_mag,
            'target_angular_mag': target_angular_mag,
            'time_mean': time_steps.mean()
        } 