"""
Loaders for SE(3) RFM Policy

This module provides data loading and processing capabilities for training
SE(3) RFM policies. It interfaces with the data_generator package to load:
- 3D environment pointclouds
- SE(3) pose pairs (init-target)
- Reference trajectories from RRT-Connect
- Robot geometry configurations
"""

from torch.utils import data
from .se3_trajectory_dataset import SE3TrajectoryDataset, create_se3_dataloader


def get_dataloader(data_dict, ddp=False, **kwargs):
    """Get dataloader for SE(3) RFM training"""
    dataset = get_dataset(data_dict)
    
    if ddp:
        loader = data.DataLoader(
            dataset,
            batch_size=data_dict["batch_size"],
            pin_memory=True,
            shuffle=False,
            sampler=data.distributed.DistributedSampler(dataset)
        )
    else:
        loader = data.DataLoader(
            dataset,
            batch_size=data_dict["batch_size"],
            shuffle=data_dict.get("shuffle", True),
            num_workers=data_dict.get("num_workers", 4),
            collate_fn=getattr(dataset, 'collate_fn', None)
        )
    return loader


def get_dataset(data_dict):
    """Get dataset for SE(3) RFM training"""
    name = data_dict.get("dataset", "se3_trajectory")
    
    if name == 'se3_trajectory':
        dataset = SE3TrajectoryDataset(**data_dict)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented")
    
    return dataset


__all__ = [
    'SE3TrajectoryDataset',
    'create_se3_dataloader',
    'get_dataloader',
    'get_dataset',
] 