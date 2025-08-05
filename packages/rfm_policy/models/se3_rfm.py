"""
SE(3) Riemannian Flow Matching Model for Robot Obstacle Avoidance

Based on TODO.md ideas and fm-main reference implementation.
Implements Riemannian Flow Matching in SE(3) space for single rigid body (ellipsoid) 
obstacle avoidance in point cloud environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import wandb
from copy import deepcopy

try:
    from .modules import PointCloudEncoder, GeometryEncoder, SE3Encoder, VelocityFieldNetwork
    from ..utils.se3_utils import SE3Utils
    from ..utils.ode_solver import get_ode_solver
except ImportError:
    # For direct execution
    from modules import PointCloudEncoder, GeometryEncoder, SE3Encoder, VelocityFieldNetwork
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from se3_utils import SE3Utils
    from ode_solver import get_ode_solver


class SE3RFM(nn.Module):
    """
    SE(3) Riemannian Flow Matching Model
    
    Implements Riemannian Flow Matching for SE(3) rigid body motion planning
    with point cloud obstacle avoidance.
    
    Architecture:
    1. PointCloudEncoder: Point cloud → latent features (DGCNN-based)
    2. GeometryEncoder: Ellipsoid parameters → geometry features
    3. SE3Encoder: SE(3) poses → pose features  
    4. VelocityFieldNetwork: All features + time → SE(3) twist
    """
    
    def __init__(
        self,
        point_cloud_config: Dict[str, Any],
        geometry_config: Dict[str, Any], 
        velocity_field_config: Dict[str, Any],
        prob_path: str = 'OT',
        init_dist: Dict[str, Any] = {'type': 'uniform'},
        ode_solver: Dict[str, Any] = {'type': 'rk4', 'n_steps': 20},
        **kwargs
    ):
        super().__init__()
        
        # Core components
        self.point_cloud_encoder = PointCloudEncoder(**point_cloud_config)
        self.geometry_encoder = GeometryEncoder(**geometry_config)
        self.se3_encoder = SE3Encoder()  # No config needed - just flatten
        self.velocity_field = VelocityFieldNetwork(**velocity_field_config)
        
        # Flow matching configuration
        self.prob_path = prob_path
        self.init_dist = init_dist
        self.ode_steps = ode_solver['n_steps']
        self.ode_solver = get_ode_solver(ode_solver['type'])
        
        # SE(3) utilities
        self.se3_utils = SE3Utils()
        
        # NFE counter for evaluation
        self.nfe_count = 0
        
        if self.prob_path not in ['OT', 'OT_CFM']:
            raise NotImplementedError(f"Prob Path: {self.prob_path} not implemented")
    
    def reset_nfe(self):
        """Reset NFE (Number of Function Evaluations) counter"""
        self.nfe_count = 0
    
    def get_nfe(self):
        """Get current NFE count"""
        return self.nfe_count
    
    def forward(
        self, 
        current_pose: torch.Tensor,  # [B, 4, 4] SE(3) matrices
        target_pose: torch.Tensor,   # [B, 4, 4] SE(3) matrices  
        point_cloud: torch.Tensor,   # [B, N, 3] point cloud
        geometry: torch.Tensor,      # [B, 3] ellipsoid parameters (a, b, c)
        time: torch.Tensor           # [B, 1] time parameter
    ) -> torch.Tensor:
        """
        Forward pass of SE3RFM
        
        Args:
            current_pose: Current SE(3) pose [B, 4, 4]
            target_pose: Target SE(3) pose [B, 4, 4] 
            point_cloud: Point cloud environment [B, N, 3]
            geometry: Ellipsoid parameters [B, 3]
            time: Time parameter [B, 1]
            
        Returns:
            twist: SE(3) twist vector [B, 6] (linear + angular velocity)
        """
        self.nfe_count += 1
        
        batch_size = current_pose.shape[0]
        
        # Encode point cloud to latent features
        # Transpose from [B, N, 3] to [B, 3, N] for DGCNN
        pc_transposed = point_cloud.transpose(1, 2)  # [B, 3, N]
        pc_features = self.point_cloud_encoder(pc_transposed)  # [B, D_pc]
        
        # Encode geometry (ellipsoid parameters)
        geom_features = self.geometry_encoder(geometry)  # [B, D_geom]
        
        # Encode SE(3) poses
        current_features = self.se3_encoder(current_pose)  # [B, D_se3]
        target_features = self.se3_encoder(target_pose)    # [B, D_se3]
        
        # Concatenate all features
        combined_features = torch.cat([
            current_features,    # Current pose features
            target_features,     # Target pose features
            pc_features,         # Point cloud features
            geom_features,       # Geometry features
            time                 # Time parameter
        ], dim=1)  # [B, D_total]
        
        # Predict twist vector
        twist = self.velocity_field(combined_features)  # [B, 6]
        
        return twist
    
    def sample_initial_distribution(
        self, 
        num_samples: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Sample from initial distribution for flow matching
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            initial_poses: Initial SE(3) poses [num_samples, 4, 4]
        """
        if device is None:
            device = next(self.parameters()).device
        
        if self.init_dist['type'] == 'uniform':
            # Sample uniformly in SE(3)
            return self.se3_utils.sample_uniform_se3(num_samples, device=device)
            
        elif self.init_dist['type'] == 'gaussian':
            # Sample from Gaussian distribution around identity
            std = self.init_dist.get('std', 0.1)
            return self.se3_utils.sample_gaussian_se3(num_samples, std=std, device=device)
            
        else:
            raise NotImplementedError(f"Initial distribution {self.init_dist['type']} not implemented")
    
    def compute_flow_matching_loss(
        self,
        current_poses: torch.Tensor,
        target_poses: torch.Tensor, 
        point_clouds: torch.Tensor,
        geometries: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss for training
        
        Args:
            current_poses: Current poses [B, 4, 4]
            target_poses: Target poses [B, 4, 4]
            point_clouds: Point clouds [B, N, 3]
            geometries: Ellipsoid parameters [B, 3]
            
        Returns:
            loss: Flow matching loss
        """
        batch_size = current_poses.shape[0]
        device = current_poses.device
        
        # Sample random times
        t = torch.rand(batch_size, 1, device=device)  # [B, 1]
        
        if self.prob_path == 'OT':
            # Optimal transport path: x_t = (1-t) * x_0 + t * x_1
            # For SE(3), we use geodesic interpolation
            interpolated_poses = self.se3_utils.geodesic_interpolation(
                current_poses, target_poses, t.squeeze(1)
            )  # [B, 4, 4]
            
            # True velocity field (derivative of geodesic)
            true_velocity = self.se3_utils.geodesic_velocity(
                current_poses, target_poses, t.squeeze(1)
            )  # [B, 6]
            
        elif self.prob_path == 'OT_CFM':
            # Conditional Flow Matching variant
            # Add noise to the interpolation
            noise_scale = 0.01
            interpolated_poses = self.se3_utils.geodesic_interpolation(
                current_poses, target_poses, t.squeeze(1)
            )
            
            # Add SE(3) noise
            noise = self.se3_utils.sample_se3_noise(batch_size, noise_scale, device=device)
            interpolated_poses = self.se3_utils.compose_se3(interpolated_poses, noise)
            
            true_velocity = self.se3_utils.geodesic_velocity(
                current_poses, target_poses, t.squeeze(1)
            )
        
        # Predict velocity
        predicted_velocity = self.forward(
            interpolated_poses, target_poses, point_clouds, geometries, t
        )  # [B, 6]
        
        # Compute MSE loss
        loss = F.mse_loss(predicted_velocity, true_velocity)
        
        return loss
    
    def generate_trajectory(
        self,
        start_pose: torch.Tensor,
        target_pose: torch.Tensor,
        point_cloud: torch.Tensor,
        geometry: torch.Tensor,
        n_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trajectory from start to target using ODE solver
        
        Args:
            start_pose: Starting SE(3) pose [4, 4]
            target_pose: Target SE(3) pose [4, 4]
            point_cloud: Point cloud environment [N, 3]
            geometry: Ellipsoid parameters [3]
            n_steps: Number of ODE steps (uses self.ode_steps if None)
            
        Returns:
            trajectory: SE(3) trajectory [n_steps+1, 4, 4]
            times: Time steps [n_steps+1]
        """
        if n_steps is None:
            n_steps = self.ode_steps
        
        device = start_pose.device
        times = torch.linspace(0, 1, n_steps + 1, device=device)
        
        # Add batch dimension for processing
        current_pose = start_pose.unsqueeze(0)  # [1, 4, 4]
        target_pose = target_pose.unsqueeze(0)  # [1, 4, 4]
        point_cloud = point_cloud.unsqueeze(0)  # [1, N, 3]  
        geometry = geometry.unsqueeze(0)        # [1, 3]
        
        trajectory = [current_pose.clone()]
        
        self.reset_nfe()
        
        for i in range(n_steps):
            t = times[i:i+1].unsqueeze(0)  # [1, 1]
            
            # Predict twist
            twist = self.forward(
                current_pose, target_pose, point_cloud, geometry, t
            )  # [1, 6]
            
            # Integrate using ODE solver
            dt = times[i+1] - times[i]
            next_pose = self.ode_solver.step(
                current_pose, twist, dt, self.se3_utils
            )  # [1, 4, 4]
            
            trajectory.append(next_pose.clone())
            current_pose = next_pose
        
        # Remove batch dimension and stack
        trajectory = torch.stack([pose.squeeze(0) for pose in trajectory])  # [n_steps+1, 4, 4]
        
        return trajectory, times
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'prob_path': self.prob_path,
            'ode_steps': self.ode_steps,
            'init_dist': self.init_dist
        }