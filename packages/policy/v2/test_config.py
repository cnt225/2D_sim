#!/usr/bin/env python3
"""
Test the training configuration (dry run)
"""

import sys
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from loaders.tdot_hdf5_dataset import create_dataloader
from models.dgcnn import DGCNN
from models.modules import vf_FC_vec_motion
from models.motion_rcfm import MotionRCFM

print("🔍 Testing Training Configuration")
print("="*50)

# Load config
config_path = 'configs/tdot_rcfm.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("\n📋 Configuration Summary:")
print(f"  Epochs: {config['training']['n_epoch']}")
print(f"  Batch size: {config['data']['train']['batch_size']}")
print(f"  Learning rate: {config['training']['optimizer']['lr']}")
print(f"  Train trajectories: {config['data']['train']['max_trajectories']}")
print(f"  Val trajectories: {config['data']['valid']['max_trajectories']}")
print(f"  Points per cloud: {config['data']['train']['num_points']}")
print(f"  Warmup epochs: {config['training']['scheduler'].get('warmup_epochs', 0)}")
print(f"  Gradient clipping: {config['training'].get('gradient_clip_norm', 'None')}")

# Test dataloader creation
print("\n📦 Testing Dataloader...")
try:
    # Use smaller numbers for testing
    test_config = config['data']['train'].copy()
    test_config['max_trajectories'] = 2
    test_config['batch_size'] = 2
    test_config['num_workers'] = 0
    
    train_loader = create_dataloader(test_config, 'train')
    print(f"  ✓ Train loader created: {len(train_loader.dataset)} samples")
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"  ✓ Batch loaded successfully")
    print(f"    - pc shape: {batch['pc'].shape}")
    print(f"    - T_dot shape: {batch['T_dot'].shape}")
    print(f"    - current_T shape: {batch['current_T'].shape}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test model creation
print("\n🤖 Testing Model Creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    latent = DGCNN(config['model']['latent_feature'], output_channels=6)
    vf = vf_FC_vec_motion(
        in_chan=25,
        lat_chan=config['model']['latent_feature']['emb_dims'] * 2,
        out_chan=6,
        l_hidden=config['model']['velocity_field']['l_hidden'],
        activation=config['model']['velocity_field']['activation'],
        out_activation=config['model']['velocity_field']['out_activation']
    )
    
    model = MotionRCFM(
        vf, latent,
        config['model']['prob_path'],
        config['model']['init_dist'],
        config['model']['ode_solver']
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created: {n_params:,} parameters")
    print(f"  ✓ Device: {device}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test forward pass
print("\n🚀 Testing Forward Pass...")
try:
    batch = {k: v.to(device) if torch.is_tensor(v) else v 
             for k, v in batch.items()}
    
    target_T = batch['current_T'].clone()
    target_T[:, :3, 3] += batch['T_dot'][:, 3:6] * 0.1
    
    data = {
        'pc': batch['pc'],
        'current_T': batch['current_T'],
        'target_T': target_T,
        'time_t': torch.ones(batch['pc'].size(0), 1).to(device) * 0.5,
        'T_dot': batch['T_dot']
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['optimizer']['lr'])
    losses = {}
    loss_dict = model.train_step(data, losses, optimizer)
    
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Loss: {loss_dict['loss']:.6f}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Memory estimation
print("\n💾 Memory Estimation:")
try:
    if torch.cuda.is_available():
        # Actual batch size from config
        actual_batch = config['data']['train']['batch_size']
        memory_per_sample = torch.cuda.max_memory_allocated() / batch['pc'].size(0)
        estimated_memory = memory_per_sample * actual_batch / (1024**3)
        print(f"  Estimated for batch {actual_batch}: ~{estimated_memory:.2f} GB")
        print(f"  Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        if estimated_memory > 20:
            print(f"  ⚠️  Warning: May need gradient accumulation or smaller batch")
        else:
            print(f"  ✓ Memory usage looks good")
except:
    print("  Unable to estimate (CPU mode)")

# Training time estimation
print("\n⏱️  Time Estimation:")
n_samples = config['data']['train']['max_trajectories'] * 200  # Approximate
n_batches = n_samples // config['data']['train']['batch_size']
total_iterations = n_batches * config['training']['n_epoch']
print(f"  Total iterations: {total_iterations:,}")
print(f"  Batches per epoch: {n_batches:,}")
print(f"  Estimated time: 15-20 hours on RTX 4090")

print("\n" + "="*50)
print("✅ Configuration Test Complete!")
print("\nTo start training, run:")
print("  ./launch_training.sh")
print("\nOr directly:")
print("  python train_tdot_wandb.py --config configs/tdot_rcfm.yml")
print("="*50)