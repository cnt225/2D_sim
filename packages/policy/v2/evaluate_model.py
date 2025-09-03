#!/usr/bin/env python3
"""
Evaluate trained model performance
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from loaders.tdot_hdf5_dataset import TdotHDF5Dataset
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM

print("🔍 Model Evaluation")
print("="*50)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model (same architecture as training)
config = {
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

print("Loading model...")
latent = DGCNN(config['latent_feature'], output_channels=6)
vf = vf_FC_vec_motion(25, 1024, 6,
                     config['velocity_field']['l_hidden'],
                     config['velocity_field']['activation'],
                     config['velocity_field']['out_activation'])
model = MotionRCFM(vf, latent, 
                  config['prob_path'],
                  config['init_dist'],
                  config['ode_solver'])

# Load trained weights
model.load_state_dict(torch.load('best_model_quick.pth', map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded")

# Create test dataset
print("\nLoading test data...")
dataset = TdotHDF5Dataset(
    hdf5_path='../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5',
    pointcloud_root='../../../data/pointcloud/circle_envs',
    split='test',
    max_trajectories=2,
    num_points=256
)
print(f"Test samples: {len(dataset)}")

# Evaluation metrics storage
angular_errors = []
linear_errors = []
total_errors = []
predictions = []
ground_truths = []

print("\n📊 Evaluating predictions...")
print("-"*30)

# Test on multiple samples
n_test = min(20, len(dataset))
for i in range(n_test):
    sample = dataset[i]
    
    # Prepare batch (single sample)
    pc = sample['pc'].unsqueeze(0).to(device)
    current_T = sample['current_T'].unsqueeze(0).to(device)
    T_dot_gt = sample['T_dot'].unsqueeze(0).to(device)
    
    # Create dummy target
    target_T = current_T.clone()
    target_T[:, :3, 3] += T_dot_gt[:, 3:6] * 0.1
    
    # Get prediction
    with torch.no_grad():
        v = model.get_latent_vector(pc.transpose(2, 1))
        T_dot_pred = model.forward(
            current_T,
            target_T,
            torch.ones(1, 1).to(device) * 0.5,
            v,
            None
        )
    
    # Calculate errors
    angular_error = torch.norm(T_dot_pred[0, :3] - T_dot_gt[0, :3]).item()
    linear_error = torch.norm(T_dot_pred[0, 3:] - T_dot_gt[0, 3:]).item()
    total_error = torch.norm(T_dot_pred[0] - T_dot_gt[0]).item()
    
    angular_errors.append(angular_error)
    linear_errors.append(linear_error)
    total_errors.append(total_error)
    
    predictions.append(T_dot_pred[0].cpu().numpy())
    ground_truths.append(T_dot_gt[0].cpu().numpy())
    
    if i < 5:  # Print first 5 samples
        print(f"\nSample {i+1}:")
        print(f"  GT:   ω={T_dot_gt[0, :3].cpu().numpy()}, v={T_dot_gt[0, 3:].cpu().numpy()}")
        print(f"  Pred: ω={T_dot_pred[0, :3].cpu().numpy()}, v={T_dot_pred[0, 3:].cpu().numpy()}")
        print(f"  Angular error: {angular_error:.4f}, Linear error: {linear_error:.4f}")

dataset.close()

# Statistics
print("\n" + "="*50)
print("📈 Overall Statistics")
print("="*50)

print(f"\nAngular Velocity (ω) Errors:")
print(f"  Mean: {np.mean(angular_errors):.4f} rad/s")
print(f"  Std:  {np.std(angular_errors):.4f} rad/s")
print(f"  Max:  {np.max(angular_errors):.4f} rad/s")
print(f"  Min:  {np.min(angular_errors):.4f} rad/s")

print(f"\nLinear Velocity (v) Errors:")
print(f"  Mean: {np.mean(linear_errors):.4f} m/s")
print(f"  Std:  {np.std(linear_errors):.4f} m/s")
print(f"  Max:  {np.max(linear_errors):.4f} m/s")
print(f"  Min:  {np.min(linear_errors):.4f} m/s")

print(f"\nTotal Twist Error:")
print(f"  Mean: {np.mean(total_errors):.4f}")
print(f"  Std:  {np.std(total_errors):.4f}")

# Check prediction distribution
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

print(f"\nPrediction Magnitude Analysis:")
gt_norms = np.linalg.norm(ground_truths, axis=1)
pred_norms = np.linalg.norm(predictions, axis=1)
print(f"  GT magnitude:   {np.mean(gt_norms):.4f} ± {np.std(gt_norms):.4f}")
print(f"  Pred magnitude: {np.mean(pred_norms):.4f} ± {np.std(pred_norms):.4f}")

# Component-wise analysis
print(f"\nComponent-wise Mean Absolute Error:")
mae_per_component = np.mean(np.abs(predictions - ground_truths), axis=0)
for i, component in enumerate(['ωx', 'ωy', 'ωz', 'vx', 'vy', 'vz']):
    print(f"  {component}: {mae_per_component[i]:.4f}")

# Success rate (within threshold)
threshold = 0.5  # Adjust based on your requirements
success_rate = np.mean(np.array(total_errors) < threshold) * 100
print(f"\nSuccess Rate (error < {threshold}): {success_rate:.1f}%")

# Final assessment
print("\n" + "="*50)
print("🎯 Assessment")
print("="*50)

mean_error = np.mean(total_errors)
if mean_error < 0.1:
    print("✨ Excellent: Model predictions are very accurate!")
elif mean_error < 0.5:
    print("✅ Good: Model is learning well, predictions are reasonable")
elif mean_error < 1.0:
    print("⚠️  Fair: Model needs more training or tuning")
else:
    print("❌ Poor: Model requires significant improvements")

print(f"\nRecommendations:")
if len(dataset) < 100:
    print("- Train with more data for better generalization")
if mean_error > 0.5:
    print("- Consider training for more epochs")
    print("- Try adjusting learning rate or model architecture")
if np.std(total_errors) > np.mean(total_errors):
    print("- High variance suggests overfitting - add regularization")

print("\n✨ Evaluation complete!")