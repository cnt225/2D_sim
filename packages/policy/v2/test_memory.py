#!/usr/bin/env python3
"""
Test actual memory usage with different batch sizes
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from loaders.tdot_hdf5_dataset import create_dataloader
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test different batch sizes
batch_sizes = [8, 12, 16, 20]

for bs in batch_sizes:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Create mini dataset
        config = {
            'hdf5_path': '../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5',
            'pointcloud_root': '../../../data/pointcloud/circle_envs',
            'max_trajectories': 2,
            'batch_size': bs,
            'num_workers': 0,
            'num_points': 2048
        }
        
        loader = create_dataloader(config, 'train')
        
        # Create model
        latent = DGCNN({'k': 20, 'emb_dims': 1024, 'dropout': 0.5, 'mode': 'maxavg'}, 6)
        vf = vf_FC_vec_motion(25, 2048, 6, [2048, 1024, 512, 512, 512], 
                             ['relu']*5, 'linear')
        model = MotionRCFM(vf, latent, 'OT', {'arch':'uniform'}, {'arch':'RK4', 'n_steps':20})
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Forward + backward
        batch = next(iter(loader))
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        target_T = batch['current_T'].clone()
        target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
        
        data = {
            'pc': batch['pc'],
            'current_T': batch['current_T'],
            'target_T': target_T,
            'time_t': torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,
            'T_dot': batch['T_dot']
        }
        
        losses = {}
        loss_dict = model.train_step(data, losses, optimizer)
        
        # Get memory usage
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Batch {bs:2d}: {peak_memory:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Batch {bs:2d}: OOM!")
            break
        else:
            raise e

print(f"\nAvailable GPU memory: {torch.cuda.mem_get_info()[0] / (1024**3):.1f} GB")
print("Recommendation: Use the largest batch size that fits")