"""
Collision Loss for SE(3) Riemannian Flow Matching

This module implements collision avoidance loss for SE(3) RFM models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class CollisionLoss(nn.Module):
    """
    Collision avoidance loss for ellipsoid robots
    
    Penalizes poses that result in collisions with obstacles in the environment.
    """
    
    def __init__(
        self,
        collision_threshold: float = 0.05,
        penalty_scale: float = 1.0,
        use_soft_penalty: bool = True
    ):
        super().__init__()
        
        self.collision_threshold = collision_threshold
        self.penalty_scale = penalty_scale
        self.use_soft_penalty = use_soft_penalty
    
    def forward(
        self,
        poses: torch.Tensor,           # [B, 4, 4] SE(3) poses
        point_clouds: torch.Tensor,    # [B, N, 3] obstacle points
        geometries: torch.Tensor       # [B, 3] ellipsoid parameters [a, b, c]
    ) -> torch.Tensor:
        """
        Compute collision loss
        
        Args:
            poses: SE(3) poses to check for collisions
            point_clouds: Obstacle point clouds
            geometries: Ellipsoid parameters for each robot
            
        Returns:
            loss: Collision penalty loss
        """
        batch_size = poses.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            pose = poses[i]  # [4, 4]
            points = point_clouds[i]  # [N, 3]
            geometry = geometries[i]  # [3]
            
            # Compute collision penalty for this pose
            penalty = self._compute_ellipsoid_collision_penalty(pose, points, geometry)
            total_loss += penalty
        
        return total_loss / batch_size
    
    def _compute_ellipsoid_collision_penalty(
        self,
        pose: torch.Tensor,        # [4, 4] SE(3) pose
        points: torch.Tensor,      # [N, 3] obstacle points
        geometry: torch.Tensor     # [3] ellipsoid parameters
    ) -> torch.Tensor:
        """
        Compute collision penalty for ellipsoid at given pose
        
        Args:
            pose: SE(3) transformation matrix
            points: Obstacle points in world coordinates
            geometry: Ellipsoid semi-axes [a, b, c]
            
        Returns:
            penalty: Collision penalty (higher = more collision)
        """
        # Transform points to ellipsoid local coordinates
        R = pose[:3, :3]  # Rotation matrix
        t = pose[:3, 3]   # Translation vector
        
        # Transform points: p_local = R^T * (p_world - t)
        points_local = torch.matmul(points - t, R)  # [N, 3]
        
        # Normalize by ellipsoid semi-axes
        points_normalized = points_local / geometry  # [N, 3]
        
        # Compute distance to ellipsoid surface
        # For ellipsoid: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
        distances = torch.norm(points_normalized, dim=1)  # [N]
        
        # Find points inside or close to ellipsoid
        if self.use_soft_penalty:
            # Soft penalty based on distance
            inside_mask = distances < 1.0
            if inside_mask.any():
                # Penalty increases as points get closer to center
                penalties = torch.exp(-distances[inside_mask] * 10)  # Exponential penalty
                penalty = penalties.mean() * self.penalty_scale
            else:
                penalty = torch.tensor(0.0, device=pose.device)
        else:
            # Hard penalty for any collision
            collision_mask = distances < (1.0 + self.collision_threshold)
            penalty = collision_mask.float().mean() * self.penalty_scale
        
        return penalty 