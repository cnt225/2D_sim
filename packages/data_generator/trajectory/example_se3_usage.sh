#!/bin/bash
# Example usage of SE(3) batch smoothing

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate 2dsim

echo "🚀 SE(3) Trajectory Smoothing Examples"
echo "======================================"
echo ""

# Example 1: Smooth all trajectories in an environment using SE(3)
echo "Example 1: SE(3) smoothing for all pairs"
echo "python batch_smooth_trajectories.py \\"
echo "    --env-name circle_env_000000 \\"
echo "    --all-pairs \\"
echo "    --use-se3 \\"
echo "    --num-samples 200"
echo ""

# Example 2: Smooth specific trajectories with custom parameters
echo "Example 2: SE(3) smoothing with custom parameters"
echo "python batch_smooth_trajectories.py \\"
echo "    --env-name circle_env_000000 \\"
echo "    --pair-ids 0,1,2 \\"
echo "    --use-se3 \\"
echo "    --bspline-degree 3 \\"
echo "    --smoothing-factor 0.01 \\"
echo "    --num-samples 250"
echo ""

# Example 3: SE(2) smoothing (legacy mode)
echo "Example 3: SE(2) smoothing (legacy mode)"
echo "python batch_smooth_trajectories.py \\"
echo "    --env-name circle_env_000000 \\"
echo "    --all-pairs \\"
echo "    --use-se2 \\"
echo "    --density-multiplier 2.0"
echo ""

# Example 4: SE(3) smoothing without collision check
echo "Example 4: SE(3) smoothing without collision validation"
echo "python batch_smooth_trajectories.py \\"
echo "    --env-name circle_env_000000 \\"
echo "    --all-pairs \\"
echo "    --use-se3 \\"
echo "    --no-collision-check"
echo ""

echo "📝 Notes:"
echo "  - SE(3) mode uses B-spline for positions and SLERP for rotations"
echo "  - Arc-length resampling ensures uniform spacing"
echo "  - Collision validation is enabled by default"
echo "  - Use --list-pairs to see available trajectory pairs"