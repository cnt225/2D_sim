#!/bin/bash

# Tmux session for Tdot RCFM training (1500 epochs, batch 48)
# This will run in background even if SSH disconnects

SESSION_NAME="tdot_train_1500"
GPU_ID=0

echo "================================================"
echo "  Tdot RCFM Training Launch"
echo "================================================"
echo ""
echo "Configuration:"
echo "  - 1500 epochs"
echo "  - Batch size: 48"
echo "  - Learning rate: 0.00035 (with warmup)"
echo "  - WandB logging enabled"
echo "  - Session: $SESSION_NAME"
echo "  - GPU: $GPU_ID"
echo ""

# Check if session exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "❌ Session $SESSION_NAME already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach:  tmux attach -t $SESSION_NAME"
    echo "  2. Kill:    tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new detached session
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Setup environment and run training
tmux send-keys -t $SESSION_NAME "cd /home/dhkang225/2D_sim/packages/policy/v2" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Tdot RCFM Training - 1500 epochs'" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "" C-m

# Activate conda environment
tmux send-keys -t $SESSION_NAME "conda activate fm" C-m
tmux send-keys -t $SESSION_NAME "echo 'Environment activated'" C-m

# Set GPU
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
tmux send-keys -t $SESSION_NAME "echo 'Using GPU $GPU_ID'" C-m

# Create log directory
tmux send-keys -t $SESSION_NAME "mkdir -p logs/tdot_1500ep_$(date +%Y%m%d)" C-m

# Start training with logging
tmux send-keys -t $SESSION_NAME "echo 'Starting training...'" C-m
tmux send-keys -t $SESSION_NAME "python train_tdot_wandb.py \
    --config configs/tdot_rcfm.yml \
    --device cuda \
    --logdir logs/tdot_1500ep_$(date +%Y%m%d) \
    2>&1 | tee logs/tdot_training_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training launched successfully!"
echo ""
echo "================================================"
echo "  Monitoring Commands"
echo "================================================"
echo ""
echo "📺 Attach to session:"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "📊 View logs:"
echo "   tail -f logs/tdot_training_*.log"
echo ""
echo "🔍 Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🌐 WandB Dashboard:"
echo "   https://wandb.ai/cnt225-seoul-national-university/tdot_rcfm_1500epochs"
echo ""
echo "🔄 Detach from tmux:"
echo "   Press: Ctrl+B, then D"
echo ""
echo "❌ Kill session (if needed):"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================"
echo ""
echo "Training will run for ~15-20 hours"
echo "The session will continue even if SSH disconnects"
echo ""