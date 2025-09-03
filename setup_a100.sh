#!/bin/bash

# A100 Server Setup Script
# Run this after cloning the repository on the A100 server

echo "================================================"
echo "  A100 Server Environment Setup"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "packages/policy/v2/train_tdot_improved.py" ]; then
    echo "❌ Error: Please run this script from the 2D_sim root directory"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo ""

# 1. Create conda environment
echo "1. Setting up Python environment..."
if command -v conda &> /dev/null; then
    echo "   Found conda. Creating environment..."
    conda create -n tdot python=3.10 -y
    conda activate tdot
else
    echo "   Using system Python"
fi

# 2. Install dependencies
echo ""
echo "2. Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install wandb tqdm pyyaml h5py matplotlib numpy scipy
pip install open3d  # For pointcloud loading

# 3. Check GPU
echo ""
echo "3. Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'✅ GPU found: {device}')
    print(f'   Memory: {memory:.1f} GB')
    if 'A100' in device:
        print('   🎉 A100 detected!')
else:
    print('❌ No GPU found!')
"

# 4. Login to WandB
echo ""
echo "4. WandB Setup..."
echo "   Please login to WandB:"
wandb login

# 5. Check data files
echo ""
echo "5. Checking data files..."
if [ -f "data/Tdot/circles_only_integrated_trajs_Tdot.h5" ]; then
    echo "✅ Tdot data found"
    ls -lh data/Tdot/*.h5
else
    echo "❌ Tdot data not found! Please run transfer_to_a100.sh first"
fi

# 6. Create necessary directories
echo ""
echo "6. Creating directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "To start training:"
echo "  python packages/policy/v2/train_tdot_improved.py \\"
echo "    --config packages/policy/v2/configs/tdot_rcfm_a100.yml \\"
echo "    --device cuda"
echo ""
echo "Or use the launch script:"
echo "  bash launch_a100_training.sh"
echo ""