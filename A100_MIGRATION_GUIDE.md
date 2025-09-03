# 🚀 A100 Server Migration Guide

## Server Information
- **Host**: 147.46.91.57
- **Port**: 30590
- **User**: root
- **Connection**: `ssh -p 30590 root@147.46.91.57`

## 📋 Migration Steps

### Step 1: Push Latest Code (현재 서버)
```bash
cd /home/dhkang225/2D_sim
git add .
git commit -m "Add A100 configuration and scripts"
git push origin master
```

### Step 2: Transfer Data (현재 서버)
```bash
# Make scripts executable
chmod +x transfer_to_a100.sh

# Transfer data files to A100 server
./transfer_to_a100.sh
```
- Transfers HDF5 Tdot data (~40MB)
- Creates remote directory structure
- Optional: Transfers pointcloud data

### Step 3: Connect to A100 Server
```bash
ssh -p 30590 root@147.46.91.57
```

### Step 4: Setup Environment (A100 서버)
```bash
# Clone repository
git clone https://github.com/cnt225/2D_sim.git
cd 2D_sim

# Run setup script
chmod +x setup_a100.sh
./setup_a100.sh
```
This will:
- Create conda environment
- Install PyTorch with CUDA
- Setup WandB
- Verify GPU availability
- Check data files

### Step 5: Start Training (A100 서버)
```bash
# Option 1: Using launch script
chmod +x launch_a100_training.sh
./launch_a100_training.sh

# Option 2: Direct command
python packages/policy/v2/train_tdot_improved.py \
    --config packages/policy/v2/configs/tdot_rcfm_a100.yml \
    --device cuda
```

## 🎯 A100 Optimized Settings

### RTX 4090 vs A100 Comparison
| Setting | RTX 4090 | A100 | Improvement |
|---------|----------|------|-------------|
| Batch Size | 16 | 128 | 8x |
| Learning Rate | 0.0002 | 0.0008 | Scaled |
| Memory | 24GB | 40/80GB | 1.6-3.3x |
| Training Time | ~20 hours | ~3-5 hours | 4-6x faster |
| Epochs | 1500 | 500 | Faster convergence |

### Key Optimizations
1. **Large Batch Training**: 128 batch size for stable gradients
2. **Mixed Precision**: FP16 training for A100 Tensor Cores
3. **Higher Learning Rate**: Scaled with batch size
4. **Reduced Epochs**: Faster convergence with large batches

## 📊 Monitoring

### Tmux Commands
```bash
# Attach to training session
tmux attach -t tdot_a100

# Detach (keep running)
Ctrl+B, then D

# List sessions
tmux ls

# Kill session
tmux kill-session -t tdot_a100
```

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### WandB Dashboard
- URL: https://wandb.ai/cnt225-seoul-national-university/tdot_rcfm_a100
- Real-time loss tracking
- Learning rate schedules
- GPU utilization

## 🔧 Troubleshooting

### If CUDA not found
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Install appropriate PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### If out of memory
Reduce batch size in config:
```yaml
batch_size: 64  # or 32
```

### If data not found
Check data path and re-run transfer:
```bash
ls -la /root/2D_sim/data/Tdot/
```

## 📈 Expected Results
- **Training Time**: 3-5 hours (vs 20+ hours on RTX 4090)
- **Final Loss**: < 0.1
- **Convergence**: Stable without oscillation
- **Checkpoints**: Saved every 50 epochs in `logs/a100_*/`

## 📝 Notes
- A100 Tensor Cores provide significant speedup with mixed precision
- Large batch training leads to better generalization
- Monitor early epochs closely for divergence
- Early stopping implemented (patience=20 epochs)