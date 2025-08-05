"""
SE(3) RFM Policy Neural Network Modules

This module contains core neural network components for SE(3) rigid body control:
- 2D PointCloud encoder (DGCNN-based)
- SE(3) velocity field network
- Robot geometry encoder
- Utility functions and activations
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


def get_activation(s_act):
    """Get activation function by name"""
    activations = {
        "relu": nn.ReLU(inplace=True),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),
        "linear": None,
        "tanh": nn.Tanh(),
        "leakyrelu": nn.LeakyReLU(0.2, inplace=True),
        "softmax": nn.Softmax(dim=1),
        "selu": nn.SELU(),
        "elu": nn.ELU(),
    }
    
    if s_act in activations:
        return activations[s_act]
    else:
        raise ValueError(f"Unexpected activation: {s_act}")


class FC_vec(nn.Module):
    """
    Fully connected network for vector data
    """
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)


class vf_FC_vec_se3(nn.Module):
    """
    Velocity Field Network for SE(3) Rigid Body Control
    
    Inputs:
        - Current SE(3) state (3D: x, y, yaw)
        - Goal direction relative to current (3D: dx, dy, dyaw) 
        - Robot geometry features (4D: major, minor, ratio, area_ratio)
        - Time t (1D)
        - Environment features from DGCNN (256D)
    
    Output:
        - SE(3) velocity (3D: vx, vy, omega_z)
    """
    def __init__(
        self,
        in_dim=11,  # state(3) + goal(3) + geometry(4) + time(1) = 11
        lat_dim=256,  # environment features from DGCNN_2D
        out_dim=3,   # SE(3) velocity: [vx, vy, omega_z]
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec_se3, self).__init__()
        
        if l_hidden is None:
            l_hidden = [512, 256, 128, 64]
        if activation is None:
            activation = ['relu'] * len(l_hidden)
        if out_activation is None:
            out_activation = 'linear'
        
        self.in_dim = in_dim
        self.lat_dim = lat_dim  
        self.out_dim = out_dim
        
        l_neurons = l_hidden + [out_dim]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_dim + lat_dim  # combined input dimension
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, state_features, t, env_features):
        """
        Forward pass of velocity field
        
        Args:
            state_features: [batch, in_dim] - concatenated state + goal + geometry
            t: [batch, 1] - time 
            env_features: [batch, lat_dim] - environment features
            
        Returns:
            velocity: [batch, out_dim] - SE(3) velocity [vx, vy, omega_z]
        """
        # Combine all inputs
        combined_input = torch.cat([state_features, t, env_features], dim=1)
        
        # Apply velocity field network
        velocity = self.net(combined_input)
        
        return velocity


class RobotGeometryEncoder(nn.Module):
    """
    Encoder for robot geometry (ellipse parameters)
    
    Takes raw geometry parameters and encodes them with normalized features
    """
    def __init__(
        self,
        workspace_size=12.0,
        typical_obstacle_size=0.5,
        output_dim=4,
    ):
        super(RobotGeometryEncoder, self).__init__()
        
        self.workspace_size = workspace_size
        self.typical_obstacle_size = typical_obstacle_size
        self.output_dim = output_dim
        
    def forward(self, semi_major, semi_minor):
        """
        Encode robot geometry
        
        Args:
            semi_major: [batch] - semi-major axis length
            semi_minor: [batch] - semi-minor axis length
            
        Returns:
            geometry_features: [batch, output_dim] - encoded geometry
        """
        batch_size = semi_major.shape[0]
        
        # Calculate derived features
        aspect_ratio = semi_major / semi_minor
        robot_area = math.pi * semi_major * semi_minor
        env_area = self.workspace_size * self.workspace_size
        area_ratio = robot_area / env_area
        
        # Encode features
        geometry_features = torch.stack([
            semi_major,      # absolute size (major axis)
            semi_minor,      # absolute size (minor axis)  
            aspect_ratio,    # shape characteristic
            area_ratio       # relative size to environment
        ], dim=1)
        
        return geometry_features


# 2D PointCloud processing modules (adapted from DGCNN)

def knn_2d(x, k):
    """KNN for 2D points"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_2d(x, k=20, idx=None):
    """Get graph features for 2D points (adapted from DGCNN)"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn_2d(x, k=k)   # (batch_size, num_points, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN_2D(nn.Module):
    """
    2D PointCloud Encoder based on DGCNN
    
    Processes 2D environment pointclouds and outputs latent environment features
    Adapted from original DGCNN for 2D points (x, y coordinates only)
    """
    def __init__(
        self,
        input_dim=2,     # 2D points (x, y)
        k=20,            # number of neighbors in graph
        emb_dims=256,    # output embedding dimension
        dropout=0.5,
        **kwargs
    ):
        super(DGCNN_2D, self).__init__()
        
        self.k = k
        self.emb_dims = emb_dims
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        # Convolutional layers (adapted for 2D input)
        # First layer: input is 2D points, so edge features are 4D (2*2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=1, bias=False),  # 2D * 2 = 4 input channels
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch, 2, num_points] - 2D pointcloud
            
        Returns:
            features: [batch, emb_dims*2] - environment features (max + avg pooled)
        """
        batch_size = x.size(0)
        
        # Graph convolution layers
        x = get_graph_feature_2d(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_2d(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_2d(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_2d(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # Concatenate multi-scale features
        x = torch.cat((x1, x2, x3, x4), dim=1)  # 64+64+128+256 = 512

        # Final embedding layer
        x = self.conv5(x)

        # Global pooling (both max and average)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)  # [batch, emb_dims*2]
        
        if self.dropout is not None:
            x = self.dropout(x)

        return x


# Utility functions for SE(3) processing

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def se3_relative_transform(current_pose, target_pose):
    """
    Compute relative transformation from current to target
    
    Args:
        current_pose: [batch, 3] - [x, y, yaw] 
        target_pose: [batch, 3] - [x_target, y_target, yaw_target]
        
    Returns:
        relative: [batch, 3] - [dx, dy, dyaw] in current frame
    """
    # Position difference in world frame
    dx_world = target_pose[:, 0] - current_pose[:, 0]
    dy_world = target_pose[:, 1] - current_pose[:, 1]
    
    # Rotate to current robot frame
    cos_yaw = torch.cos(current_pose[:, 2])
    sin_yaw = torch.sin(current_pose[:, 2])
    
    dx_local = cos_yaw * dx_world + sin_yaw * dy_world
    dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
    
    # Angle difference (normalized)
    dyaw = normalize_angle(target_pose[:, 2] - current_pose[:, 2])
    
    return torch.stack([dx_local, dy_local, dyaw], dim=1)


# =============================================================================
# SE(3) RFM Modules for 3D Rigid Body Motion Planning
# =============================================================================

def knn_3d(x, k):
    """KNN for 3D points"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_3d(x, k=20, idx=None):
    """Get graph features for 3D points (DGCNN-style)"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    # Ensure k doesn't exceed number of points
    k = min(k, num_points)
    
    if idx is None:
        idx = knn_3d(x, k=k)   # (batch_size, num_points, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature


class PointCloudEncoder(nn.Module):
    """
    3D Point Cloud Encoder based on DGCNN
    
    Encodes 3D point cloud obstacles into latent features
    for SE(3) motion planning.
    """
    def __init__(
        self,
        k=20,
        emb_dims=1024,
        dropout=0.5
    ):
        super(PointCloudEncoder, self).__init__()
        
        self.k = k
        self.emb_dims = emb_dims
        
        # DGCNN layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # No output projection - keep fm-main style
        # Just use max + avg pooling output directly

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch_size, 3, num_points] point cloud
            
        Returns:
            features: [batch_size, emb_dims*2] encoded features (like fm-main)
        """
        batch_size = x.size(0)
        
        # DGCNN feature extraction
        x = get_graph_feature_3d(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_3d(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_3d(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_3d(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        # Global pooling (exactly like fm-main)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B, emb_dims]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [B, emb_dims]
        features = torch.cat((x1, x2), 1)  # [B, emb_dims*2]
        
        return features


class GeometryEncoder(nn.Module):
    """
    Encoder for ellipsoid geometry parameters
    
    Takes ellipsoid parameters (a, b, c) and encodes them into features
    that capture shape characteristics relevant for motion planning.
    """
    def __init__(
        self,
        input_dim=3,       # (a, b, c) ellipsoid semi-axes
        hidden_dim=64,
        output_dim=32
    ):
        super(GeometryEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for derived features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, geometry):
        """
        Encode ellipsoid geometry
        
        Args:
            geometry: [batch_size, 3] ellipsoid parameters (a, b, c)
            
        Returns:
            features: [batch_size, output_dim] encoded geometry features
        """
        a, b, c = geometry[:, 0], geometry[:, 1], geometry[:, 2]
        
        # Compute derived geometric features
        volume = (4/3) * torch.pi * a * b * c
        aspect_ratio_1 = a / b  # primary aspect ratio
        aspect_ratio_2 = a / c  # secondary aspect ratio
        
        # Combine original and derived features
        enhanced_features = torch.stack([
            a, b, c,           # original parameters
            volume,            # volume
            aspect_ratio_1,    # aspect ratios
            aspect_ratio_2
        ], dim=1)
        
        features = self.encoder(enhanced_features)
        return features


class SE3Encoder(nn.Module):
    """
    Simple SE(3) matrix flattener (like fm-main)
    
    Takes SE(3) transformation matrices and flattens them directly
    without compression, preserving all information.
    """
    def __init__(self):
        super(SE3Encoder, self).__init__()
        # No parameters needed - just flatten
        
    def forward(self, se3_matrix):
        """
        Flatten SE(3) matrix
        
        Args:
            se3_matrix: [batch_size, 4, 4] SE(3) transformation matrices
            
        Returns:
            features: [batch_size, 12] flattened SE(3) features
        """
        batch_size = se3_matrix.shape[0]
        
        # Extract the meaningful 3x4 part and flatten (like fm-main)
        # Concatenate rotation matrix columns + translation vector
        x_flatten = torch.cat([
            se3_matrix[:, 0:3, 0],  # First column of rotation
            se3_matrix[:, 0:3, 1],  # Second column of rotation  
            se3_matrix[:, 0:3, 2],  # Third column of rotation
            se3_matrix[:, 0:3, 3]   # Translation vector
        ], dim=1)  # [B, 12]
        
        return x_flatten


class VelocityFieldNetwork(nn.Module):
    """
    Velocity Field Network for SE(3) Flow Matching
    
    Takes combined features and predicts SE(3) twist vector (6D velocity).
    """
    def __init__(
        self,
        input_dim,         # Total dimension of combined features
        hidden_dims=[512, 512, 256, 256],
        output_dim=6,      # SE(3) twist: [v_x, v_y, v_z, w_x, w_y, w_z]
        dropout=0.1
    ):
        super(VelocityFieldNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for velocity prediction)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, combined_features):
        """
        Predict SE(3) twist vector
        
        Args:
            combined_features: [batch_size, input_dim] combined features
            
        Returns:
            twist: [batch_size, 6] SE(3) twist vector
        """
        twist = self.network(combined_features)
        return twist 