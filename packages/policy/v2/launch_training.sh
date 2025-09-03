#!/bin/bash

# Launch training in tmux session
# Usage: ./launch_training.sh [session_name] [gpu_id]

SESSION_NAME=${1:-tdot_training}
GPU_ID=${2:-0}

echo "=========================================="
echo "Launching Tdot RCFM Training"
echo "=========================================="
echo "Session: $SESSION_NAME"
echo "GPU: $GPU_ID"
echo "Config: 1500 epochs, batch 64"
echo ""

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION_NAME already exists!"
    echo "Use 'tmux attach -t $SESSION_NAME' to attach"
    echo "Or 'tmux kill-session -t $SESSION_NAME' to remove"
    exit 1
fi

# Create new tmux session
echo "Creating tmux session..."
tmux new-session -d -s $SESSION_NAME

# Send commands to session
tmux send-keys -t $SESSION_NAME "cd /home/dhkang225/2D_sim/packages/policy/v2" C-m
tmux send-keys -t $SESSION_NAME "conda activate fm" C-m  # Activate conda environment
tmux send-keys -t $SESSION_NAME "echo 'Starting training in 5 seconds...'" C-m
tmux send-keys -t $SESSION_NAME "sleep 5" C-m

# Launch training with WandB
tmux send-keys -t $SESSION_NAME "CUDA_VISIBLE_DEVICES=$GPU_ID python train_tdot_wandb.py \
    --config configs/tdot_rcfm.yml \
    --device cuda \
    --logdir results/tdot_1500epochs \
    2>&1 | tee training.log" C-m

echo "✅ Training launched in tmux session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  Attach to session:  tmux attach -t $SESSION_NAME"
echo "  Detach:            Ctrl+B, then D"
echo "  List sessions:     tmux ls"
echo "  Kill session:      tmux kill-session -t $SESSION_NAME"
echo "  Monitor GPU:       watch nvidia-smi"
echo "  View logs:         tail -f training.log"
echo ""
echo "Training will run for approximately 15-20 hours"
echo "WandB dashboard will be available for real-time monitoring"