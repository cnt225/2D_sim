#!/bin/bash

# A100 Training Launch Script
# Optimized for high batch size training

SESSION_NAME="tdot_a100"
CONFIG="packages/policy/v2/configs/tdot_rcfm_a100.yml"

echo "================================================"
echo "  A100 Optimized Training"
echo "================================================"
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

echo ""
echo "Configuration:"
echo "  - Batch size: 128"
echo "  - Learning rate: 0.0008"
echo "  - Epochs: 500"
echo "  - Mixed precision: Enabled"
echo ""

# Kill existing session if exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Setup environment
tmux send-keys -t $SESSION_NAME "conda activate tdot 2>/dev/null || true" C-m

# Run training with A100 optimizations
tmux send-keys -t $SESSION_NAME "python packages/policy/v2/train_tdot_improved.py \
    --config $CONFIG \
    --device cuda \
    --logdir logs/a100_$(date +%Y%m%d) \
    2>&1 | tee logs/a100_training_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training launched!"
echo ""
echo "================================================"
echo "  Monitoring Commands"
echo "================================================"
echo ""
echo "📺 Attach to session:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "📊 Monitor GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "📈 WandB Dashboard:"
echo "   https://wandb.ai/cnt225-seoul-national-university/tdot_rcfm_a100"
echo ""
echo "🔄 Detach: Ctrl+B, D"
echo "❌ Stop: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Expected training time: ~3-5 hours on A100"
echo "================================================"