"""
SE(3) RFM Policy Models

This module provides Robot Foundation Model (RFM) based policy models
for SE(3) rigid body navigation and control.
"""

import torch
from omegaconf import OmegaConf

from .modules import (
    vf_FC_vec_se3,
    DGCNN_2D,
    RobotGeometryEncoder,
    # New SE3RFM modules
    PointCloudEncoder,
    GeometryEncoder,
    SE3Encoder,
    VelocityFieldNetwork,
    get_activation
)

from .se3_rcfm import SE3RCFM
from .se3_rfm import SE3RFM


def get_model(model_cfg, *args, **kwargs):
    """
    Get model instance based on configuration
    
    Args:
        model_cfg: Model configuration dict/OmegaConf
        *args, **kwargs: Additional arguments
        
    Returns:
        Initialized model instance
    """
    name = model_cfg["arch"]
    model = _get_model_instance(name)
    model = model(**model_cfg, **kwargs)

    # Load checkpoint if specified
    if 'checkpoint' in model_cfg:
        checkpoint = torch.load(model_cfg['checkpoint'], map_location='cpu')
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)

    return model


def _get_model_instance(name):
    """Get model class by name"""
    try:
        return {
            'se3_rcfm': get_se3_rcfm_model,
            'se3_rfm': get_se3_rfm_model,
        }[name]
    except KeyError:
        raise ValueError(f"Model {name} not available. Available models: se3_rcfm, se3_rfm")


def get_net(**kwargs):
    """
    Get network module based on architecture
    
    Args:
        kwargs: Network configuration
        
    Returns:
        Network module instance
    """
    arch = kwargs["arch"]
    
    if arch == "vf_fc_vec_se3":
        return vf_FC_vec_se3(**kwargs)
    elif arch == "dgcnn_2d":
        return DGCNN_2D(**kwargs)
    elif arch == "robot_geometry_encoder":
        return RobotGeometryEncoder(**kwargs)
    else:
        raise ValueError(f"Network architecture {arch} not implemented")


def get_se3_rcfm_model(**model_cfg):
    """
    Create SE(3) Riemannian Conditional Flow Matching model
    
    Args:
        model_cfg: Model configuration dict
        
    Returns:
        SE3RCFM model instance
    """
    # Create velocity field network
    velocity_field = get_net(**model_cfg['velocity_field'])
    
    # Create environment encoder (2D PointCloud -> latent features)
    env_encoder = get_net(**model_cfg['env_encoder'])
    
    # Create robot geometry encoder
    robot_encoder = get_net(**model_cfg['robot_encoder'])
    
    # Get other configurations
    prob_path = model_cfg.get('prob_path', 'OT')
    init_dist = model_cfg.get('init_dist', {'arch': 'uniform'})
    ode_solver = model_cfg.get('ode_solver', {'arch': 'RK4', 'n_steps': 20})
    
    model = SE3RCFM(
        velocity_field=velocity_field,
        env_encoder=env_encoder,
        robot_encoder=robot_encoder,
        prob_path=prob_path,
        init_dist=init_dist,
        ode_solver=ode_solver
    )
    
    return model


def get_se3_rfm_model(**model_cfg):
    """
    Create SE(3) Riemannian Flow Matching model
    
    Args:
        model_cfg: Model configuration dict
        
    Returns:
        SE3RFM model instance
    """
    # Calculate input dimension for velocity field (like fm-main)
    pc_emb_dims = model_cfg['point_cloud_encoder']['emb_dims']
    pc_output_dim = pc_emb_dims * 2  # max + avg pooling
    geom_output_dim = model_cfg['geometry_encoder']['output_dim']
    se3_output_dim = 12  # Direct flatten, no compression
    
    velocity_field_input_dim = (
        se3_output_dim * 2 +  # current + target pose features (12*2 = 24)
        pc_output_dim +       # point cloud features (1024*2 = 2048)
        geom_output_dim +     # geometry features (32)
        1                     # time (1)
    )  # Total: 24 + 2048 + 32 + 1 = 2105
    
    model_cfg['velocity_field_config']['input_dim'] = velocity_field_input_dim
    
    # Create model
    model = SE3RFM(
        point_cloud_config=model_cfg['point_cloud_encoder'],
        geometry_config=model_cfg['geometry_encoder'],
        velocity_field_config=model_cfg['velocity_field_config'],
        prob_path=model_cfg.get('prob_path', 'OT'),
        init_dist=model_cfg.get('init_dist', {'type': 'uniform'}),
        ode_solver=model_cfg.get('ode_solver', {'type': 'rk4', 'n_steps': 20})
    )
    
    return model


def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    Load pre-trained SE(3) RFM model
    
    Args:
        identifier: Model identifier '<model_name>/<run_name>'
        config_file: Configuration file name
        ckpt_file: Checkpoint file name  
        root: Root directory for pretrained models
        **kwargs: Additional arguments
        
    Returns:
        (model, config) tuple
    """
    import os
    
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    
    cfg = OmegaConf.load(config_path)
    model = get_model(cfg.model)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    
    model.load_state_dict(ckpt)
    
    return model, cfg 