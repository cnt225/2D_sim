"""
SE(3) Utilities for Riemannian Flow Matching

This module provides essential SE(3) (Special Euclidean Group) operations
for 3D rigid body motion planning and Riemannian Flow Matching.

Key functionalities:
- SE(3) matrix operations (composition, inverse, etc.)
- Lie algebra se(3) operations (logarithm, exponential maps)
- Geodesic interpolation and velocity computation
- Sampling from SE(3) distributions
- Twist vector operations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation


class SE3Utils:
    """
    Utilities for SE(3) operations in PyTorch
    
    SE(3) is the group of 3D rigid body transformations consisting of
    rotation and translation. Elements are represented as 4x4 matrices:
    
    T = [[R, t],
         [0, 1]]
         
    where R is 3x3 rotation matrix and t is 3x1 translation vector.
    """
    
    def __init__(self):
        self.eps = 1e-8  # Small epsilon for numerical stability
    
    # =================================================================
    # Basic SE(3) Operations
    # =================================================================
    
    def compose_se3(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """
        Compose two SE(3) transformations: T1 * T2
        
        Args:
            T1: [batch_size, 4, 4] or [4, 4]
            T2: [batch_size, 4, 4] or [4, 4]
            
        Returns:
            T_composed: [batch_size, 4, 4] or [4, 4]
        """
        return torch.matmul(T1, T2)
    
    def inverse_se3(self, T: torch.Tensor) -> torch.Tensor:
        """
        Compute SE(3) inverse
        
        For SE(3) matrix T = [[R, t], [0, 1]], 
        inverse is T^(-1) = [[R^T, -R^T * t], [0, 1]]
        
        Args:
            T: [batch_size, 4, 4] or [4, 4] SE(3) matrices
            
        Returns:
            T_inv: [batch_size, 4, 4] or [4, 4] inverse matrices
        """
        if T.dim() == 2:
            # Single matrix
            R = T[:3, :3]
            t = T[:3, 3]
            R_T = R.transpose(0, 1)
            
            T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
            T_inv[:3, :3] = R_T
            T_inv[:3, 3] = -torch.matmul(R_T, t)
            
        else:
            # Batch of matrices
            batch_size = T.shape[0]
            R = T[:, :3, :3]  # [B, 3, 3]
            t = T[:, :3, 3]   # [B, 3]
            R_T = R.transpose(-2, -1)  # [B, 3, 3]
            
            T_inv = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            T_inv[:, :3, :3] = R_T
            T_inv[:, :3, 3] = -torch.bmm(R_T, t.unsqueeze(-1)).squeeze(-1)
        
        return T_inv
    
    def extract_rotation_translation(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract rotation and translation from SE(3) matrix
        
        Args:
            T: [batch_size, 4, 4] or [4, 4] SE(3) matrices
            
        Returns:
            R: [batch_size, 3, 3] or [3, 3] rotation matrices
            t: [batch_size, 3] or [3] translation vectors
        """
        if T.dim() == 2:
            R = T[:3, :3]
            t = T[:3, 3]
        else:
            R = T[:, :3, :3]
            t = T[:, :3, 3]
        
        return R, t
    
    def construct_se3(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Construct SE(3) matrix from rotation and translation
        
        Args:
            R: [batch_size, 3, 3] or [3, 3] rotation matrices
            t: [batch_size, 3] or [3] translation vectors
            
        Returns:
            T: [batch_size, 4, 4] or [4, 4] SE(3) matrices
        """
        if R.dim() == 2:
            T = torch.eye(4, device=R.device, dtype=R.dtype)
            T[:3, :3] = R
            T[:3, 3] = t
        else:
            batch_size = R.shape[0]
            T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            T[:, :3, :3] = R
            T[:, :3, 3] = t
        
        return T
    
    # =================================================================
    # Lie Algebra Operations
    # =================================================================
    
    def skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """
        Convert vector to skew-symmetric matrix
        
        Args:
            v: [batch_size, 3] or [3] vectors
            
        Returns:
            skew: [batch_size, 3, 3] or [3, 3] skew-symmetric matrices
        """
        if v.dim() == 1:
            return torch.tensor([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], device=v.device, dtype=v.dtype)
        else:
            batch_size = v.shape[0]
            zeros = torch.zeros(batch_size, device=v.device, dtype=v.dtype)
            
            skew = torch.stack([
                torch.stack([zeros, -v[:, 2], v[:, 1]], dim=1),
                torch.stack([v[:, 2], zeros, -v[:, 0]], dim=1),
                torch.stack([-v[:, 1], v[:, 0], zeros], dim=1)
            ], dim=1)
            
            return skew
    
    def unskew_symmetric(self, skew: torch.Tensor) -> torch.Tensor:
        """
        Convert skew-symmetric matrix to vector
        
        Args:
            skew: [batch_size, 3, 3] or [3, 3] skew-symmetric matrices
            
        Returns:
            v: [batch_size, 3] or [3] vectors
        """
        if skew.dim() == 2:
            return torch.tensor([skew[2, 1], skew[0, 2], skew[1, 0]], device=skew.device, dtype=skew.dtype)
        else:
            return torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=1)
    
    def twist_to_se3_matrix(self, twist: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D twist vector to 4x4 se(3) matrix
        
        Args:
            twist: [batch_size, 6] or [6] twist vectors [v, w] (linear, angular)
            
        Returns:
            se3_matrix: [batch_size, 4, 4] or [4, 4] se(3) matrices
        """
        if twist.dim() == 1:
            v = twist[:3]  # linear velocity
            w = twist[3:]  # angular velocity
            
            se3_matrix = torch.zeros(4, 4, device=twist.device, dtype=twist.dtype)
            se3_matrix[:3, :3] = self.skew_symmetric(w)
            se3_matrix[:3, 3] = v
            
        else:
            batch_size = twist.shape[0]
            v = twist[:, :3]  # [B, 3] linear velocity
            w = twist[:, 3:]  # [B, 3] angular velocity
            
            se3_matrix = torch.zeros(batch_size, 4, 4, device=twist.device, dtype=twist.dtype)
            se3_matrix[:, :3, :3] = self.skew_symmetric(w)
            se3_matrix[:, :3, 3] = v
        
        return se3_matrix
    
    def se3_matrix_to_twist(self, se3_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert 4x4 se(3) matrix to 6D twist vector
        
        Args:
            se3_matrix: [batch_size, 4, 4] or [4, 4] se(3) matrices
            
        Returns:
            twist: [batch_size, 6] or [6] twist vectors [v, w]
        """
        if se3_matrix.dim() == 2:
            v = se3_matrix[:3, 3]
            w = self.unskew_symmetric(se3_matrix[:3, :3])
            return torch.cat([v, w])
        else:
            v = se3_matrix[:, :3, 3]
            w = self.unskew_symmetric(se3_matrix[:, :3, :3])
            return torch.cat([v, w], dim=1)
    
    def exp_se3(self, twist: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from se(3) to SE(3)
        
        Uses Rodriguez formula for efficient computation.
        
        Args:
            twist: [batch_size, 6] or [6] twist vectors [v, w]
            
        Returns:
            T: [batch_size, 4, 4] or [4, 4] SE(3) matrices
        """
        if twist.dim() == 1:
            v = twist[:3]
            w = twist[3:]
            
            # Angular part
            theta = torch.norm(w)
            if theta < self.eps:
                R = torch.eye(3, device=twist.device, dtype=twist.dtype)
                V = torch.eye(3, device=twist.device, dtype=twist.dtype)
            else:
                w_hat = self.skew_symmetric(w)
                R = torch.eye(3, device=twist.device, dtype=twist.dtype) + \
                    torch.sin(theta) / theta * w_hat + \
                    (1 - torch.cos(theta)) / (theta ** 2) * torch.matmul(w_hat, w_hat)
                
                V = torch.eye(3, device=twist.device, dtype=twist.dtype) + \
                    (1 - torch.cos(theta)) / (theta ** 2) * w_hat + \
                    (theta - torch.sin(theta)) / (theta ** 3) * torch.matmul(w_hat, w_hat)
            
            # Translation part
            t = torch.matmul(V, v)
            
            return self.construct_se3(R, t)
        
        else:
            batch_size = twist.shape[0]
            v = twist[:, :3]  # [B, 3]
            w = twist[:, 3:]  # [B, 3]
            
            # Angular part
            theta = torch.norm(w, dim=1, keepdim=True)  # [B, 1]
            small_angle_mask = (theta < self.eps).squeeze(-1)  # [B]
            
            # Initialize R and V
            I3 = torch.eye(3, device=twist.device, dtype=twist.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            R = I3.clone()
            V = I3.clone()
            
            if (~small_angle_mask).any():
                # Normal case
                valid_indices = ~small_angle_mask
                w_valid = w[valid_indices]  # [B_valid, 3]
                theta_valid = theta[valid_indices]  # [B_valid, 1]
                
                w_hat = self.skew_symmetric(w_valid)  # [B_valid, 3, 3]
                w_hat_squared = torch.bmm(w_hat, w_hat)  # [B_valid, 3, 3]
                
                sin_theta = torch.sin(theta_valid).unsqueeze(-1)  # [B_valid, 1, 1]
                cos_theta = torch.cos(theta_valid).unsqueeze(-1)  # [B_valid, 1, 1]
                theta_sq = (theta_valid ** 2).unsqueeze(-1)  # [B_valid, 1, 1]
                theta_cube = (theta_valid ** 3).unsqueeze(-1)  # [B_valid, 1, 1]
                
                R[valid_indices] = I3[valid_indices] + \
                    sin_theta / theta_valid.unsqueeze(-1) * w_hat + \
                    (1 - cos_theta) / theta_sq * w_hat_squared
                
                V[valid_indices] = I3[valid_indices] + \
                    (1 - cos_theta) / theta_sq * w_hat + \
                    (theta_valid.unsqueeze(-1) - sin_theta) / theta_cube * w_hat_squared
            
            # Translation part
            t = torch.bmm(V, v.unsqueeze(-1)).squeeze(-1)  # [B, 3]
            
            return self.construct_se3(R, t)
    
    def log_se3(self, T: torch.Tensor) -> torch.Tensor:
        """
        Logarithm map from SE(3) to se(3)
        
        Args:
            T: [batch_size, 4, 4] or [4, 4] SE(3) matrices
            
        Returns:
            twist: [batch_size, 6] or [6] twist vectors [v, w]
        """
        # This is a simplified implementation
        # For full implementation, would need careful handling of edge cases
        R, t = self.extract_rotation_translation(T)
        
        if T.dim() == 2:
            # Single matrix case
            # Use scipy for rotation matrix to axis-angle conversion
            if torch.allclose(R, torch.eye(3, device=R.device), atol=self.eps):
                w = torch.zeros(3, device=T.device, dtype=T.dtype)
            else:
                R_np = R.detach().cpu().numpy()
                rot = Rotation.from_matrix(R_np)
                w = torch.from_numpy(rot.as_rotvec()).to(device=T.device, dtype=T.dtype)
            
            # For translation, use simple approximation
            v = t
            
            return torch.cat([v, w])
        
        else:
            # Batch case - simplified implementation
            batch_size = T.shape[0]
            twists = []
            
            for i in range(batch_size):
                twist_i = self.log_se3(T[i])
                twists.append(twist_i)
            
            return torch.stack(twists)
    
    # =================================================================
    # Geodesic Operations
    # =================================================================
    
    def geodesic_interpolation(
        self, 
        T1: torch.Tensor, 
        T2: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Geodesic interpolation between two SE(3) poses
        
        Computes T(t) = T1 * exp(t * log(T1^(-1) * T2))
        
        Args:
            T1: [batch_size, 4, 4] start poses
            T2: [batch_size, 4, 4] end poses  
            t: [batch_size] or scalar interpolation parameter [0, 1]
            
        Returns:
            T_interp: [batch_size, 4, 4] interpolated poses
        """
        # Compute relative transformation
        T1_inv = self.inverse_se3(T1)
        T_rel = self.compose_se3(T1_inv, T2)  # T1^(-1) * T2
        
        # Get twist for relative transformation
        twist_rel = self.log_se3(T_rel)  # [B, 6]
        
        # Scale by interpolation parameter
        if t.dim() == 0:  # scalar
            twist_scaled = t * twist_rel
        else:  # batch
            twist_scaled = t.unsqueeze(-1) * twist_rel  # [B, 1] * [B, 6] = [B, 6]
        
        # Exponentiate and compose with T1
        T_scaled = self.exp_se3(twist_scaled)  # [B, 4, 4]
        T_interp = self.compose_se3(T1, T_scaled)  # T1 * exp(t * log(T1^(-1) * T2))
        
        return T_interp
    
    def geodesic_velocity(
        self,
        T1: torch.Tensor,
        T2: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geodesic velocity (time derivative of geodesic)
        
        Args:
            T1: [batch_size, 4, 4] start poses
            T2: [batch_size, 4, 4] end poses
            t: [batch_size] or scalar interpolation parameter [0, 1]
            
        Returns:
            velocity: [batch_size, 6] twist velocity
        """
        # For geodesic interpolation T(t) = T1 * exp(t * xi),
        # where xi = log(T1^(-1) * T2), the velocity is xi
        T1_inv = self.inverse_se3(T1)
        T_rel = self.compose_se3(T1_inv, T2)
        velocity = self.log_se3(T_rel)  # [B, 6]
        
        return velocity
    
    # =================================================================
    # Sampling Functions
    # =================================================================
    
    def sample_uniform_se3(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample uniformly from SE(3)
        
        Args:
            num_samples: Number of samples
            device: Device to generate samples on
            
        Returns:
            samples: [num_samples, 4, 4] SE(3) matrices
        """
        if device is None:
            device = torch.device('cpu')
        
        # Sample random rotations
        R = torch.from_numpy(
            Rotation.random(num_samples).as_matrix()
        ).float().to(device)
        
        # Sample random translations (in reasonable range)
        t = torch.randn(num_samples, 3, device=device) * 2.0  # Scale as needed
        
        return self.construct_se3(R, t)
    
    def sample_gaussian_se3(
        self, 
        num_samples: int, 
        mean: torch.Tensor = None,
        std: float = 0.1,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Sample from Gaussian distribution in SE(3)
        
        Args:
            num_samples: Number of samples
            mean: [4, 4] mean SE(3) matrix (identity if None)
            std: Standard deviation for perturbations
            device: Device to generate samples on
            
        Returns:
            samples: [num_samples, 4, 4] SE(3) matrices
        """
        if device is None:
            device = torch.device('cpu')
        
        if mean is None:
            mean = torch.eye(4, device=device)
        
        # Sample twist vectors from Gaussian
        twists = torch.randn(num_samples, 6, device=device) * std
        
        # Convert to SE(3) and compose with mean
        perturbations = self.exp_se3(twists)
        samples = self.compose_se3(mean.unsqueeze(0), perturbations)
        
        return samples
    
    def sample_se3_noise(
        self, 
        batch_size: int, 
        noise_scale: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Sample SE(3) noise for augmentation
        
        Args:
            batch_size: Batch size
            noise_scale: Scale of noise
            device: Device
            
        Returns:
            noise: [batch_size, 4, 4] SE(3) noise matrices
        """
        twists = torch.randn(batch_size, 6, device=device) * noise_scale
        return self.exp_se3(twists)