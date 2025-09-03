#!/usr/bin/env python3
"""
Minimal test of the training pipeline
"""

import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from loaders.tdot_hdf5_dataset import TdotHDF5Dataset
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM

print("Creating dataset...")
dataset = TdotHDF5Dataset(
    hdf5_path='../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5',
    pointcloud_root='../../../data/pointcloud/circle_envs',
    split='train',
    max_trajectories=2,
    num_points=512
)
print(f"Dataset size: {len(dataset)}")

print("Creating dataloader...")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

print("Creating model...")
latent_feature = DGCNN({'k': 20, 'emb_dims': 1024, 'dropout': 0.5}, output_channels=6)
velocity_field = vf_FC_vec_motion(
    in_chan=25,
    lat_chan=2048,
    out_chan=6,
    l_hidden=[2048, 1024, 512],
    activation=['relu', 'relu', 'relu'],
    out_activation='linear'
)

model = MotionRCFM(
    velocity_field=velocity_field,
    latent_feature=latent_feature,
    prob_path='OT',
    init_dist={'arch': 'uniform'},
    ode_solver={'arch': 'RK4', 'n_steps': 20}
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nTesting forward pass...")
device = torch.device('cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Get one batch
batch = next(iter(dataloader))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

print(f"Batch shapes:")
print(f"  pc: {batch['pc'].shape}")
print(f"  T_dot: {batch['T_dot'].shape}")
print(f"  current_T: {batch['current_T'].shape}")

# Create target pose
target_T = batch['current_T'].clone()
target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1

# Prepare data
data = {
    'pc': batch['pc'],
    'current_T': batch['current_T'],
    'target_T': target_T,
    'time_t': torch.ones(batch['pc'].size(0), 1) * 0.5,
    'T_dot': batch['T_dot']
}

print("\nRunning train step...")
losses = {}
loss_dict = model.train_step(data, losses, optimizer)
print(f"Loss: {loss_dict['loss']:.6f}")

print("\n✅ Test successful!")
dataset.close()