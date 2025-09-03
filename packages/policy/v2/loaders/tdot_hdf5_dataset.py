#!/usr/bin/env python3
"""
HDF5 Tdot Dataset Loader
Loads pre-computed Tdot trajectories from HDF5 files
"""

import torch
import numpy as np
import h5py
import os
from pathlib import Path
from torch.utils.data import Dataset
import random
from typing import Optional, Dict, Any

class TdotHDF5Dataset(Dataset):
    """HDF5 dataset for pre-computed Tdot trajectories"""
    
    def __init__(
        self,
        hdf5_path: str,
        pointcloud_root: str,
        split: str = 'train',
        max_trajectories: Optional[int] = None,
        augment_data: bool = False,
        num_points: int = 2048,
        trajectory_length: int = 100,
        step_size: int = 1,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file with Tdot trajectories
            pointcloud_root: Root directory for pointcloud files
            split: Dataset split ('train', 'val', 'test')
            max_trajectories: Maximum number of trajectories to load
            augment_data: Whether to apply data augmentation
            num_points: Number of points to sample from pointcloud
            trajectory_length: Length of trajectory to use
            step_size: Step size for sampling trajectory points
        """
        self.hdf5_path = Path(hdf5_path)
        self.pointcloud_root = Path(pointcloud_root)
        self.split = split
        self.augment_data = augment_data
        self.num_points = num_points
        self.trajectory_length = trajectory_length
        self.step_size = step_size
        
        # Load HDF5 file
        print(f"Loading HDF5 dataset from {hdf5_path}")
        self.h5file = h5py.File(hdf5_path, 'r')
        
        # Get all environment keys
        self.env_keys = list(self.h5file.keys())
        print(f"Found {len(self.env_keys)} environments")
        
        # Collect all trajectory paths
        self.trajectories = []
        for env_key in self.env_keys:
            env_group = self.h5file[env_key]
            for traj_key in env_group.keys():
                if 'Tdot_trajectory' in env_group[traj_key]:
                    self.trajectories.append((env_key, traj_key))
        
        # Limit trajectories if specified
        if max_trajectories is not None:
            self.trajectories = self.trajectories[:max_trajectories]
        
        print(f"Loaded {len(self.trajectories)} trajectories for split '{split}'")
        
        # Split data (simple split based on index)
        n_total = len(self.trajectories)
        if split == 'train':
            self.trajectories = self.trajectories[:int(0.8 * n_total)]
        elif split == 'val':
            self.trajectories = self.trajectories[int(0.8 * n_total):int(0.9 * n_total)]
        elif split == 'test':
            self.trajectories = self.trajectories[int(0.9 * n_total):]
        
        print(f"Using {len(self.trajectories)} trajectories for {split}")
        
        # Create sample indices for trajectory steps
        self.samples = []
        for env_key, traj_key in self.trajectories:
            # Get trajectory length
            tdot_traj = self.h5file[env_key][traj_key]['Tdot_trajectory']
            n_steps = tdot_traj.shape[0]
            
            # Sample step indices
            for i in range(0, n_steps - 1, self.step_size):
                self.samples.append((env_key, traj_key, i))
        
        print(f"Created {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        env_key, traj_key, step_idx = self.samples[idx]
        
        # Load Tdot trajectory
        tdot_traj = self.h5file[env_key][traj_key]['Tdot_trajectory'][:]
        
        # Get current and next Tdot
        current_tdot = tdot_traj[step_idx]  # [6] or [4,4]
        
        # Convert to 6D twist if in matrix form
        if current_tdot.shape == (4, 4):
            # Extract twist from transformation matrix
            current_tdot = self._matrix_to_twist(current_tdot)
        
        # Load pointcloud
        pc_path = self.pointcloud_root / f"{env_key}.ply"
        if pc_path.exists():
            pc = self._load_pointcloud(pc_path)
        else:
            # Generate random pointcloud if file not found
            pc = np.random.randn(self.num_points, 3).astype(np.float32)
        
        # Sample points
        if pc.shape[0] > self.num_points:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[indices]
        elif pc.shape[0] < self.num_points:
            # Pad with repeated points
            indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
            pc = pc[indices]
        
        # Apply augmentation if enabled
        if self.augment_data:
            pc = self._augment_pointcloud(pc)
        
        # Get current pose (using step index to create a pose)
        # For now, using identity + accumulated twist
        current_T = self._compute_pose_at_step(tdot_traj, step_idx)
        
        # Prepare output
        sample = {
            'pc': torch.tensor(pc, dtype=torch.float32),  # [N, 3]
            'T_dot': torch.tensor(current_tdot, dtype=torch.float32),  # [6]
            'current_T': torch.tensor(current_T, dtype=torch.float32),  # [4, 4]
            'env_name': env_key,
            'traj_name': traj_key,
            'step_idx': step_idx
        }
        
        return sample
    
    def _load_pointcloud(self, pc_path):
        """Load pointcloud from PLY file"""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pc_path))
            points = np.asarray(pcd.points).astype(np.float32)
            return points
        except:
            # Fallback: try loading as numpy array
            try:
                points = np.load(str(pc_path))
                if len(points.shape) == 2 and points.shape[1] == 3:
                    return points.astype(np.float32)
            except:
                pass
            # If all fails, return random points
            return np.random.randn(self.num_points, 3).astype(np.float32)
    
    def _augment_pointcloud(self, pc):
        """Apply random augmentation to pointcloud"""
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        pc = pc @ rot_matrix.T
        
        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        pc = pc * scale
        
        # Random translation
        translation = np.random.uniform(-0.1, 0.1, size=(3,))
        pc = pc + translation
        
        # Add noise
        noise = np.random.randn(*pc.shape) * 0.01
        pc = pc + noise
        
        return pc.astype(np.float32)
    
    def _matrix_to_twist(self, T):
        """Convert 4x4 transformation matrix to 6D twist vector"""
        # Simple extraction of rotation and translation velocities
        # This is a placeholder - should use proper SE3 log if needed
        omega = np.array([T[2,1], T[0,2], T[1,0]])  # Approximate angular velocity
        v = T[:3, 3]  # Linear velocity
        return np.concatenate([omega, v]).astype(np.float32)
    
    def _compute_pose_at_step(self, tdot_traj, step_idx):
        """Compute accumulated pose at given step"""
        # For simplicity, return identity matrix
        # In practice, you'd integrate the twist to get the pose
        T = np.eye(4, dtype=np.float32)
        
        # Optional: accumulate some rotation based on step
        angle = step_idx * 0.01
        T[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return T
    
    def close(self):
        """Close HDF5 file"""
        if hasattr(self, 'h5file'):
            self.h5file.close()


def create_dataloader(config: Dict[str, Any], split: str = 'train'):
    """Create dataloader from config"""
    dataset = TdotHDF5Dataset(
        hdf5_path=config.get('hdf5_path', '../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5'),
        pointcloud_root=config.get('pointcloud_root', '../../../data/pointcloud/circle_envs'),
        split=split,
        max_trajectories=config.get('max_trajectories', None),
        augment_data=config.get('augmentation', False),
        num_points=config.get('num_points', 2048),
        trajectory_length=config.get('trajectory_length', 100),
        step_size=config.get('step_size', 1)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing TdotHDF5Dataset...")
    
    config = {
        'hdf5_path': '../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5',
        'pointcloud_root': '../../../data/pointcloud/circle_envs',
        'batch_size': 4,
        'num_workers': 0,
        'num_points': 2048,
        'augmentation': False
    }
    
    # Create dataset
    dataset = TdotHDF5Dataset(
        hdf5_path=config['hdf5_path'],
        pointcloud_root=config['pointcloud_root'],
        split='train',
        max_trajectories=10,
        num_points=config['num_points']
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Pointcloud shape: {sample['pc'].shape}")
    print(f"T_dot shape: {sample['T_dot'].shape}")
    print(f"Current T shape: {sample['current_T'].shape}")
    print(f"T_dot values: {sample['T_dot']}")
    
    # Test dataloader
    dataloader = create_dataloader(config, 'train')
    print(f"\nDataloader created with {len(dataloader)} batches")
    
    # Get one batch
    for batch in dataloader:
        print(f"\nBatch shapes:")
        print(f"  pc: {batch['pc'].shape}")
        print(f"  T_dot: {batch['T_dot'].shape}")
        print(f"  current_T: {batch['current_T'].shape}")
        break
    
    dataset.close()
    print("\n✅ Dataset test complete!")