#!/usr/bin/env python3
"""Minimal training script"""

print("Script starting...")

import os
import sys
print("Basic imports done")

import torch
print("Torch imported")

import yaml
print("YAML imported")

from pathlib import Path
print("Path imported")

# Add to path
sys.path.append(str(Path(__file__).parent))
print("Path added")

# Main function
def main():
    print("In main function")
    
    # Load config
    with open('configs/tdot_rcfm.yml', 'r') as f:
        config = yaml.safe_load(f)
    print("Config loaded")
    
    # Create dataloader
    from loaders.tdot_hdf5_dataset import create_dataloader
    print("Dataloader module imported")
    
    train_loader = create_dataloader(config['data']['train'], 'train')
    print(f"Train loader created: {len(train_loader)} batches")
    
    # Create model components
    from models.dgcnn import DGCNN
    from models.modules import vf_FC_vec_motion
    from models.motion_rcfm import MotionRCFM
    print("Model modules imported")
    
    # Build model
    latent = DGCNN(config['model']['latent_feature'], output_channels=6)
    vf = vf_FC_vec_motion(25, 2048, 6, [2048, 1024, 512], ['relu']*3, 'linear')
    model = MotionRCFM(vf, latent, 'OT', {'arch':'uniform'}, {'arch':'RK4', 'n_steps':20})
    print("Model created")
    
    # Train one step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    batch = next(iter(train_loader))
    print("Got batch")
    
    # Forward
    target_T = batch['current_T'].clone()
    data = {
        'pc': batch['pc'],
        'current_T': batch['current_T'],
        'target_T': target_T,
        'time_t': torch.ones(batch['pc'].size(0), 1) * 0.5,
        'T_dot': batch['T_dot']
    }
    
    losses = {}
    loss_dict = model.train_step(data, losses, optimizer)
    print(f"Loss: {loss_dict['loss']:.6f}")
    
    print("✅ Training successful!")

if __name__ == "__main__":
    print("Calling main...")
    main()