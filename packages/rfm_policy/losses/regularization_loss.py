"""
Regularization Loss for SE(3) Riemannian Flow Matching

This module implements regularization losses for SE(3) RFM models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class RegularizationLoss(nn.Module):
    """
    Regularization losses for trajectory smoothness and velocity control
    
    Includes:
    - Smoothness loss: penalizes high curvature trajectories
    - Velocity loss: penalizes excessive velocities
    - Acceleration loss: penalizes high accelerations
    """
    
    def __init__(
        self,
        smoothness_weight: float = 0.1,
        velocity_weight: float = 0.01,
        acceleration_weight: float = 0.01
    ):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
    
    def forward(
        self,
        predicted_velocities: torch.Tensor,  # [B, 6] predicted velocities
        time_steps: torch.Tensor = None       # [B, 1] time steps (optional)
    ) -> torch.Tensor:
        """
        Compute regularization loss
        
        Args:
            predicted_velocities: Predicted velocity vectors
            time_steps: Time steps for acceleration computation (optional)
            
        Returns:
            loss: Combined regularization loss
        """
        total_loss = 0.0
        
        # Velocity magnitude penalty
        if self.velocity_weight > 0:
            velocity_magnitudes = torch.norm(predicted_velocities, dim=1)  # [B]
            velocity_loss = torch.mean(velocity_magnitudes ** 2)
            total_loss += self.velocity_weight * velocity_loss
        
        # Smoothness penalty (curvature)
        if self.smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss(predicted_velocities)
            total_loss += self.smoothness_weight * smoothness_loss
        
        # Acceleration penalty (if time steps provided)
        if self.acceleration_weight > 0 and time_steps is not None:
            acceleration_loss = self._compute_acceleration_loss(predicted_velocities, time_steps)
            total_loss += self.acceleration_weight * acceleration_loss
        
        return total_loss
    
    def _compute_smoothness_loss(self, velocities: torch.Tensor) -> torch.Tensor:
        """Compute smoothness loss based on velocity variations"""
        # Split into linear and angular components
        linear_vel = velocities[:, :3]   # [B, 3]
        angular_vel = velocities[:, 3:]  # [B, 3]
        
        # Compute velocity gradients (approximate)
        # This penalizes rapid changes in velocity direction
        linear_smoothness = torch.var(torch.norm(linear_vel, dim=1))
        angular_smoothness = torch.var(torch.norm(angular_vel, dim=1))
        
        return linear_smoothness + angular_smoothness
    
    def _compute_acceleration_loss(self, velocities: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """Compute acceleration penalty (simplified)"""
        # Simplified acceleration computation
        # In practice, this would require velocity history
        # For now, penalize high velocities which correlate with high accelerations
        
        velocity_magnitudes = torch.norm(velocities, dim=1)  # [B]
        
        # Penalize velocities that are too high (indicating potential high acceleration)
        max_velocity_threshold = 2.0  # reasonable velocity limit
        excess_velocities = torch.clamp(velocity_magnitudes - max_velocity_threshold, min=0)
        
        return torch.mean(excess_velocities ** 2) 