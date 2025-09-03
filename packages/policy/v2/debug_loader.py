#!/usr/bin/env python3
"""
Debug dataloader creation
"""

import sys
import torch
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Load config
with open('configs/tdot_rcfm.yml', 'r') as f:
    config = yaml.safe_load(f)

print("Config loaded")
print(f"Train config: {config['data']['train']}")

# Try to create dataloader
from loaders.tdot_hdf5_dataset import create_dataloader

print("\nCreating train dataloader...")
train_loader = create_dataloader(config['data']['train'], 'train')
print(f"Train dataloader created with {len(train_loader)} batches")
print(f"Dataset size: {len(train_loader.dataset)}")

print("\nGetting first batch...")
batch = next(iter(train_loader))
print(f"Batch loaded!")
print(f"  pc shape: {batch['pc'].shape}")
print(f"  T_dot shape: {batch['T_dot'].shape}")

print("\nTest complete!")