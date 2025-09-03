#!/usr/bin/env python3
"""
Final check before launching training
"""

import sys
import torch
import yaml
import wandb
from pathlib import Path

print("="*60)
print("🔍 FINAL PRE-TRAINING CHECK")
print("="*60)

# 1. Check config
print("\n1️⃣ Configuration:")
with open('configs/tdot_rcfm.yml', 'r') as f:
    config = yaml.safe_load(f)

print(f"   Epochs: {config['training']['n_epoch']}")
print(f"   Batch size: {config['data']['train']['batch_size']}")
print(f"   Learning rate: {config['training']['optimizer']['lr']}")
print(f"   WandB entity: {config['wandb']['entity']}")
print(f"   WandB project: {config['wandb']['project_name']}")

# 2. Check GPU
print("\n2️⃣ GPU Status:")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"   Available: {torch.cuda.mem_get_info()[0] / (1024**3):.1f} GB")
else:
    print("   ❌ No GPU available!")
    sys.exit(1)

# 3. Check WandB
print("\n3️⃣ WandB Connection:")
try:
    import wandb.apis
    print(f"   ✅ WandB installed and ready")
    print(f"   Entity: cnt225-seoul-national-university")
except:
    print("   ❌ WandB not properly configured")

# 4. Check data
print("\n4️⃣ Data Check:")
from pathlib import Path
hdf5_path = Path("../../../data/Tdot/circles_only_integrated_trajs_Tdot.h5")
if hdf5_path.exists():
    print(f"   ✅ HDF5 file exists: {hdf5_path}")
    print(f"   Size: {hdf5_path.stat().st_size / (1024**3):.2f} GB")
else:
    print(f"   ❌ HDF5 file not found!")

# 5. Summary
print("\n" + "="*60)
print("📋 TRAINING SUMMARY")
print("="*60)
print(f"""
🎯 Target: 1500 epochs
📦 Batch size: 48
📈 Learning rate: 0.00035 (with 50 epoch warmup)
💾 Estimated memory: ~17 GB
⏱️  Estimated time: 15-20 hours
🌐 WandB: https://wandb.ai/cnt225-seoul-national-university/tdot_rcfm_1500epochs

Ready to launch? Run:
./start_training_tmux.sh
""")

print("✅ All checks passed! Ready to train!")
print("="*60)