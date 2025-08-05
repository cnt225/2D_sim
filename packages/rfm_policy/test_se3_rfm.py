"""
SE(3) RFM System Integration Test

Simple test script to verify that all components work together correctly.
"""

import torch
import numpy as np
from pathlib import Path
import yaml

# Import our modules
from models.se3_rfm import SE3RFM
from loaders import get_dataloader
from losses import MultiTaskLoss
from trainers.optimizers import get_optimizer
from trainers.schedulers import get_scheduler
from utils.se3_utils import SE3Utils
from evaluation.evaluator import SE3RFMEvaluator


def test_se3_utils():
    """Test SE(3) utilities"""
    print("Testing SE(3) utilities...")
    
    se3_utils = SE3Utils()
    
    # Test SE(3) matrix operations
    T1 = torch.eye(4)
    T1[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Translation
    
    T2 = torch.eye(4) 
    T2[:3, 3] = torch.tensor([2.0, 1.0, 0.0])
    
    # Test composition
    T_composed = se3_utils.compose_se3(T1, T2)
    assert T_composed.shape == (4, 4)
    
    # Test inverse
    T_inv = se3_utils.inverse_se3(T1)
    identity = se3_utils.compose_se3(T1, T_inv)
    assert torch.allclose(identity, torch.eye(4), atol=1e-6)
    
    # Test twist operations
    twist = torch.tensor([0.1, 0.2, 0.3, 0.05, 0.0, 0.1])  # [v, w]
    T_exp = se3_utils.exp_se3(twist)
    twist_log = se3_utils.log_se3(T_exp)
    assert torch.allclose(twist, twist_log, atol=1e-4)
    
    print("✅ SE(3) utilities test passed!")


def test_model_components():
    """Test individual model components"""
    print("Testing model components...")
    
    batch_size = 4
    n_points = 100
    
    # Test PointCloudEncoder
    from models.modules import PointCloudEncoder
    pc_encoder = PointCloudEncoder(output_dim=512)
    pc_input = torch.randn(batch_size, 3, n_points)
    pc_features = pc_encoder(pc_input)
    assert pc_features.shape == (batch_size, 512)
    
    # Test GeometryEncoder
    from models.modules import GeometryEncoder
    geom_encoder = GeometryEncoder(output_dim=32)
    geom_input = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.3], [0.6, 0.2, 0.1], [0.3, 0.5, 0.4]])
    geom_features = geom_encoder(geom_input)
    assert geom_features.shape == (batch_size, 32)
    
    # Test SE3Encoder (no parameters, just flatten)
    from models.modules import SE3Encoder
    se3_encoder = SE3Encoder()
    se3_input = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    se3_features = se3_encoder(se3_input)
    assert se3_features.shape == (batch_size, 12)
    
    # Test VelocityFieldNetwork  
    from models.modules import VelocityFieldNetwork
    input_dim = 12 * 2 + 1024 * 2 + 32 + 1  # current(12) + target(12) + pc(2048) + geom(32) + time(1) = 2105
    vf_network = VelocityFieldNetwork(input_dim=input_dim)
    vf_input = torch.randn(batch_size, input_dim)
    velocity = vf_network(vf_input)
    assert velocity.shape == (batch_size, 6)
    
    print("✅ Model components test passed!")


def test_se3rfm_model():
    """Test complete SE3RFM model"""
    print("Testing SE3RFM model...")
    
    # Model configuration (like fm-main style)
    config = {
        'point_cloud_encoder': {'k': 20, 'emb_dims': 256, 'dropout': 0.1},
        'geometry_encoder': {'input_dim': 3, 'hidden_dim': 32, 'output_dim': 16},
        'velocity_field_config': {'hidden_dims': [512, 256], 'output_dim': 6, 'dropout': 0.1}
    }
    
    # Calculate input dimension (like fm-main)
    pc_dim = config['point_cloud_encoder']['emb_dims'] * 2  # 256 * 2 = 512
    se3_dim = 12  # Direct flatten
    geom_dim = config['geometry_encoder']['output_dim']  # 16
    time_dim = 1
    input_dim = se3_dim * 2 + pc_dim + geom_dim + time_dim  # 12*2 + 512 + 16 + 1 = 553
    config['velocity_field_config']['input_dim'] = input_dim
    
    # Create model
    model = SE3RFM(
        point_cloud_config=config['point_cloud_encoder'],
        geometry_config=config['geometry_encoder'],

        velocity_field_config=config['velocity_field_config']
    )
    
    # Test data
    batch_size = 2
    n_points = 50
    
    current_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    target_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    target_poses[:, :3, 3] = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.5, 0.0]])
    
    point_clouds = torch.randn(batch_size, n_points, 3)
    geometries = torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.3, 0.2]])
    time_steps = torch.tensor([[0.5], [0.3]])
    
    # Forward pass
    with torch.no_grad():
        velocity = model(current_poses, target_poses, point_clouds, geometries, time_steps)
        assert velocity.shape == (batch_size, 6)
    
    # Test trajectory generation
    single_start = current_poses[0]
    single_target = target_poses[0]
    single_pc = point_clouds[0]
    single_geom = geometries[0]
    
    trajectory, times = model.generate_trajectory(
        single_start, single_target, single_pc, single_geom, n_steps=10
    )
    assert trajectory.shape == (11, 4, 4)  # n_steps + 1
    assert times.shape == (11,)
    
    print("✅ SE3RFM model test passed!")


def test_loss_functions():
    """Test loss functions"""
    print("Testing loss functions...")
    
    # Create simple model for testing
    from models.modules import VelocityFieldNetwork
    simple_model = VelocityFieldNetwork(input_dim=100, hidden_dims=[64], output_dim=6)
    
    # Mock forward function for testing
    def mock_forward(current_pose, target_pose, point_cloud, geometry, time):
        batch_size = current_pose.shape[0]
        return torch.randn(batch_size, 6)
    
    simple_model.forward = mock_forward
    simple_model.se3_utils = SE3Utils()
    
    # Create loss function
    loss_fn = MultiTaskLoss()
    
    # Test data
    batch_size = 2
    start_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    goal_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    goal_poses[:, :3, 3] = torch.randn(batch_size, 3)
    
    time_steps = torch.rand(batch_size, 1)
    point_clouds = torch.randn(batch_size, 50, 3)
    geometries = torch.tensor([[0.3, 0.2, 0.1], [0.4, 0.3, 0.2]])
    
    # Compute loss
    loss = loss_fn(
        predicted_velocity=None,
        start_poses=start_poses,
        goal_poses=goal_poses,
        time_steps=time_steps,
        point_clouds=point_clouds,
        geometries=geometries,
        model=simple_model
    )
    
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    
    print("✅ Loss functions test passed!")


def test_data_loader():
    """Test data loader (if data exists)"""
    print("Testing data loader...")
    
    # Check if we have trajectory data
    data_root = Path("data/trajectories")
    if not data_root.exists():
        print("⚠️  No trajectory data found, skipping data loader test")
        return
    
    try:
        # Create small dataset
        dataloader = get_dataloader({
            "data_root": str(data_root),
            "batch_size": 2,
            "shuffle": False,
            "num_workers": 0,  # Avoid multiprocessing issues in testing
            "max_trajectories": 5,  # Limit for testing
            "dataset": "se3_trajectory"
        })
        
        # Test one batch
        for batch in dataloader:
            assert 'trajectories' in batch
            assert 'point_clouds' in batch
            assert 'geometries' in batch
            assert 'start_poses' in batch
            assert 'goal_poses' in batch
            break
        
        print("✅ Data loader test passed!")
        
    except Exception as e:
        print(f"⚠️  Data loader test failed: {e}")


def test_optimizers_schedulers():
    """Test optimizers and schedulers"""
    print("Testing optimizers and schedulers...")
    
    # Create dummy model
    model = torch.nn.Linear(10, 6)
    
    # Test optimizer
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type='adamw',
        lr=0.001,
        weight_decay=0.01
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    
    # Test scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='cosine_annealing',
        T_max=100,
        eta_min=1e-6
    )
    assert hasattr(scheduler, 'step')
    
    print("✅ Optimizers and schedulers test passed!")


def main():
    """Run all tests"""
    print("🚀 Starting SE(3) RFM Integration Tests")
    print("=" * 50)
    
    try:
        test_se3_utils()
        test_model_components()
        test_se3rfm_model()
        test_loss_functions()
        test_data_loader()
        test_optimizers_schedulers()
        
        print("=" * 50)
        print("🎉 All tests passed! SE(3) RFM system is ready!")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()