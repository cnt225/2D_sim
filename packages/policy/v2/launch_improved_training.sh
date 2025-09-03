#!/bin/bash

# Improved training launch script with gradient accumulation
SESSION_NAME="tdot_improved"
GPU_ID=0

echo "================================================"
echo "  Improved Tdot Training Launch"
echo "================================================"
echo ""

# Check arguments
LR=${1:-0.00005}  # Default: 5e-5 (much lower)
BATCH=${2:-16}     # Default: 16
ACCUM=${3:-2}      # Default: 2 (effective batch 32)
EPOCHS=${4:-200}   # Default: 200 epochs

echo "Configuration:"
echo "  - Learning Rate: $LR"
echo "  - Batch Size: $BATCH"
echo "  - Accumulation Steps: $ACCUM"
echo "  - Effective Batch: $((BATCH * ACCUM))"
echo "  - Epochs: $EPOCHS"
echo "  - GPU: $GPU_ID"
echo ""

# Kill existing session if exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME

# Setup and run
tmux send-keys -t $SESSION_NAME "cd /home/dhkang225/2D_sim/packages/policy/v2" C-m
tmux send-keys -t $SESSION_NAME "conda activate fm" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m

# Run with improved settings
tmux send-keys -t $SESSION_NAME "python train_tdot_improved.py \
    --config configs/tdot_rcfm.yml \
    --device cuda \
    --lr $LR \
    --batch_size $BATCH \
    --accumulation_steps $ACCUM \
    --epochs $EPOCHS \
    --logdir logs/improved_$(date +%Y%m%d) \
    2>&1 | tee logs/improved_training_$(date +%Y%m%d_%H%M%S).log" C-m

echo "✅ Training launched!"
echo ""
echo "Commands:"
echo "  Watch:  tmux attach -t $SESSION_NAME"
echo "  Detach: Ctrl+B, D"
echo "  Kill:   tmux kill-session -t $SESSION_NAME"
echo "  GPU:    watch -n 1 nvidia-smi"
echo ""
echo "Effective batch size: $((BATCH * ACCUM))"
echo "Training with stable convergence settings"