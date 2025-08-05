# SE(3) Riemannian Flow Matching for Robot Obstacle Avoidance

Complete implementation of SE(3) Riemannian Flow Matching for single rigid body (ellipsoid) robot obstacle avoidance in point cloud environments.

## 🎯 Overview

This project implements a novel approach to robot motion planning using Riemannian Flow Matching on SE(3) manifold. The system learns to generate smooth, collision-free trajectories for ellipsoid robots navigating through point cloud obstacle environments.

### Key Features

- **SE(3) Manifold Operations**: Complete SE(3) utilities with geodesic interpolation
- **3D Point Cloud Processing**: DGCNN-based environment encoder 
- **Ellipsoid Robot Model**: Configurable ellipsoid geometry with collision detection
- **Flow Matching Training**: Riemannian Flow Matching with multiple loss components
- **ODE Integration**: Multiple ODE solvers for trajectory generation
- **Comprehensive Evaluation**: Trajectory quality metrics and visualization

## 🏗️ Architecture

```
SE3RFM Model Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Point Cloud     │    │ Current Pose    │    │ Target Pose     │
│ [B, N, 3]      │    │ [B, 4, 4]      │    │ [B, 4, 4]      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PointCloudEncoder│    │   SE3Encoder    │    │   SE3Encoder    │
│ (DGCNN-based)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       └───────┬───────────────┘
┌─────────────────┐                      ▼
│ Geometry [B,3]  │              ┌─────────────────┐
│                 │              │ Pose Features   │
└─────────────────┘              │ [B, 2*D_se3]   │
         │                       └─────────────────┘
         ▼                                │
┌─────────────────┐                      │
│ GeometryEncoder │                      │
│                 │                      │
└─────────────────┘                      │
         │                               │
         └──────────┬────────────────────┘
                    ▼
            ┌─────────────────┐
            │ Combined        │
            │ Features + Time │
            │ [B, D_total]   │
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │ VelocityField   │
            │ Network         │
            └─────────────────┘
                    │
                    ▼
            ┌─────────────────┐
            │ SE(3) Twist     │
            │ [B, 6]         │
            └─────────────────┘
```

## 📦 Components

### Core Models
- **SE3RFM**: Main Riemannian Flow Matching model
- **PointCloudEncoder**: 3D DGCNN for environment encoding
- **GeometryEncoder**: Ellipsoid parameter encoder
- **SE3Encoder**: SE(3) pose encoder
- **VelocityFieldNetwork**: Predicts SE(3) twist vectors

### Training System
- **MultiTaskLoss**: Flow matching + collision + regularization losses
- **SE3Utils**: Complete SE(3) manifold operations
- **ODE Solvers**: Euler, RK4, adaptive solvers
- **Data Loaders**: SE(3) trajectory dataset interface

### Evaluation
- **SE3RFMEvaluator**: Comprehensive trajectory evaluation
- **Collision Detection**: Ellipsoid-point cloud collision checking
- **Trajectory Metrics**: Smoothness, efficiency, success rate
- **Visualization**: 3D trajectory plotting

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision numpy scipy matplotlib wandb tensorboardX omegaconf
```

### 2. Test the System
```bash
cd packages/rfm_policy
python test_se3_rfm.py
```

### 3. Training
```bash
python train_se3_rfm.py --config configs/se3_rfm_config.yaml --num_gpus 1
```

### 4. Evaluation
```bash
python -m evaluation.evaluator --model_path checkpoints/best_model.pth --test_data data/trajectories --save_dir results/
```

## 🔧 Configuration

Key configuration options in `configs/se3_rfm_config.yaml`:

```yaml
model:
  point_cloud_encoder:
    k: 20                    # KNN neighbors
    emb_dims: 1024          # Embedding dimensions
    output_dim: 512         # Output features

  geometry_encoder:
    output_dim: 32          # Geometry features

  se3_encoder:
    output_dim: 64          # SE(3) pose features

  velocity_field_config:
    hidden_dims: [512, 512, 256, 256, 128]
    dropout: 0.1

training:
  num_epochs: 1000
  batch_size: 16
  learning_rate: 0.0005
  
  loss:
    flow_matching_weight: 1.0
    collision_weight: 0.1
    regularization_weight: 0.01
```

## 📊 Results

The system generates smooth, collision-free trajectories for ellipsoid robots:

- **Success Rate**: Percentage of collision-free trajectories
- **Path Efficiency**: Ratio of geodesic distance to actual path length  
- **Smoothness**: Based on velocity and acceleration variations
- **Computation Time**: Real-time trajectory generation capability

## 🛠️ Technical Details

### SE(3) Operations
- Exponential and logarithm maps
- Geodesic interpolation and velocity computation
- Twist vector operations
- Robust numerical implementations

### Flow Matching
- Optimal Transport (OT) and Conditional Flow Matching (CFM)
- Riemannian manifold considerations
- Multi-task loss with collision avoidance

### Data Pipeline
- Compatible with existing B-splined trajectory data
- Automatic point cloud loading and preprocessing
- Efficient batch processing with variable lengths

## 📁 Project Structure

```
packages/rfm_policy/
├── models/
│   ├── se3_rfm.py              # Main SE3RFM model
│   ├── modules.py              # Neural network components
│   └── __init__.py
├── losses/
│   ├── flow_matching_loss.py   # Flow matching loss
│   ├── collision_loss.py       # Collision avoidance loss
│   ├── regularization_loss.py  # Regularization loss
│   └── multi_task_loss.py      # Combined multi-task loss
├── trainers/
│   ├── optimizers.py           # Optimizers
│   └── schedulers.py           # Learning rate schedulers
├── utils/
│   ├── se3_utils.py           # SE(3) manifold operations
│   └── ode_solver.py          # ODE integration
├── loaders/
│   └── se3_trajectory_dataset.py  # Data loading
├── evaluation/
│   └── evaluator.py           # Model evaluation
├── configs/
│   └── se3_rfm_config.yaml    # Configuration
├── train_se3_rfm.py           # Training script
└── test_se3_rfm.py            # Integration tests
```

## 🔬 Research Background

This implementation is based on the TODO.md research plan for SE(3) Riemannian Flow Matching:

1. **Single Rigid Body Focus**: Ellipsoid robot in 3D space
2. **SE(3) Manifold**: Proper handling of rotation and translation
3. **Point Cloud Environments**: Real-world obstacle representation
4. **Flow Matching**: Modern generative modeling for trajectory planning

## 🤝 Contributing

The system is designed to be modular and extensible:

- Add new robot geometries in `GeometryEncoder`
- Implement different flow paths in loss functions
- Extend to robot arms using the foundation
- Add new evaluation metrics

## 📝 Status

✅ **Completed Components**:
- SE(3) Riemannian Flow Matching model
- Complete training pipeline with Wandb
- Comprehensive evaluation system
- Integration tests and documentation

🔄 **Next Steps**:
- Train on large-scale trajectory datasets
- Extend to multi-robot scenarios
- Robot arm integration (as per TODO.md plan)
- Real robot deployment

---

**Note**: This implementation provides a complete foundation for SE(3) robot motion planning using Riemannian Flow Matching. The system is ready for training and can be extended to more complex robotic systems.
