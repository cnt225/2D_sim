#!/usr/bin/env python3
"""
Quick training test - 10 epochs only
"""

import sys
import torch
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from loaders.tdot_hdf5_dataset import create_dataloader
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM

# Quick config
config = {
    'data': {
        'train': {
            'hdf5_path': '../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5',
            'pointcloud_root': '../../../data/pointcloud/circle_envs',
            'max_trajectories': 5,
            'batch_size': 4,
            'num_points': 256,
            'num_workers': 0
        }
    },
    'model': {
        'latent_feature': {'k': 10, 'emb_dims': 512, 'dropout': 0.5, 'mode': 'maxavg'},
        'velocity_field': {
            'out_dim': 6,
            'l_hidden': [1024, 512, 256],
            'activation': ['relu', 'relu', 'relu'],
            'out_activation': 'linear'
        },
        'prob_path': 'OT',
        'init_dist': {'arch': 'uniform'},
        'ode_solver': {'arch': 'RK4', 'n_steps': 10}
    }
}

print("🚀 Quick Training Test (10 epochs)")
print("-" * 40)

# Create dataset and dataloader
print("Creating dataloader...")
train_loader = create_dataloader(config['data']['train'], 'train')
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Batches per epoch: {len(train_loader)}")

# Create model
print("\nCreating model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent = DGCNN(config['model']['latent_feature'], output_channels=6)
vf = vf_FC_vec_motion(25, 1024, 6,  # 512*2 = 1024 for maxavg pooling
                     config['model']['velocity_field']['l_hidden'],
                     config['model']['velocity_field']['activation'],
                     config['model']['velocity_field']['out_activation'])
model = MotionRCFM(vf, latent, 
                  config['model']['prob_path'],
                  config['model']['init_dist'],
                  config['model']['ode_solver'])
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training
print("\n" + "="*40)
print("Starting Training")
print("="*40)

train_losses = []
best_loss = float('inf')

for epoch in range(10):
    model.train()
    epoch_losses = []
    
    for i, batch in enumerate(train_loader):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Create target
        target_T = batch['current_T'].clone()
        target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
        
        data = {
            'pc': batch['pc'],
            'current_T': batch['current_T'],
            'target_T': target_T,
            'time_t': torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,
            'T_dot': batch['T_dot']
        }
        
        # Train step
        losses = {}
        loss_dict = model.train_step(data, losses, optimizer)
        epoch_losses.append(loss_dict['loss'])
        
        if i == 0:  # Print first batch loss
            print(f"Epoch {epoch+1}/10 - First batch loss: {loss_dict['loss']:.6f}")
    
    avg_loss = np.mean(epoch_losses)
    train_losses.append(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model_quick.pth')
    
    print(f"Epoch {epoch+1}/10 - Avg Loss: {avg_loss:.6f} - Best: {best_loss:.6f}")
    scheduler.step()

print("\n" + "="*40)
print("Training Complete!")
print("="*40)

# Analysis
print("\n📊 Results Analysis:")
print(f"Initial Loss: {train_losses[0]:.6f}")
print(f"Final Loss: {train_losses[-1]:.6f}")
print(f"Improvement: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")

# Check convergence
if train_losses[-1] < train_losses[0]:
    print("✅ Model is learning (loss decreased)")
else:
    print("⚠️  Model may need tuning (loss increased)")

# Save training curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.savefig('training_curve.png')
print("\n📈 Training curve saved to 'training_curve.png'")

print("\n🎯 Testing model inference...")
# Test inference
model.eval()
with torch.no_grad():
    test_batch = next(iter(train_loader))
    test_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in test_batch.items()}
    
    # Get prediction
    target_T = test_batch['current_T'].clone()
    target_T[:, :3, 3] += test_batch['T_dot'][:, 3:6] * 0.1
    
    v = model.get_latent_vector(test_batch['pc'].transpose(2, 1))
    T_dot_pred = model.forward(
        test_batch['current_T'],
        target_T,
        torch.ones(test_batch['pc'].size(0), 1).to(device) * 0.5,
        v,
        None
    )
    
    # Compare prediction with ground truth
    mse = torch.nn.functional.mse_loss(T_dot_pred, test_batch['T_dot'])
    print(f"Test MSE: {mse.item():.6f}")
    
    # Print sample predictions
    print("\nSample predictions (first item):")
    print(f"  Ground truth T_dot: {test_batch['T_dot'][0].cpu().numpy()}")
    print(f"  Predicted T_dot:    {T_dot_pred[0].cpu().numpy()}")
    
    # Check prediction magnitudes
    gt_norm = torch.norm(test_batch['T_dot'], dim=1).mean()
    pred_norm = torch.norm(T_dot_pred, dim=1).mean()
    print(f"\nAverage magnitude:")
    print(f"  Ground truth: {gt_norm:.4f}")
    print(f"  Predicted:    {pred_norm:.4f}")

print("\n✨ Quick training test complete!")