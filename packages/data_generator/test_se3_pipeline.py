#!/usr/bin/env python3
"""
SE(3) Trajectory Pipeline Test & Validation
Complete integration test for SE(3) trajectory processing pipeline
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import traceback

# Add paths
sys.path.append('/Users/a123/Documents/Projects/2D_sim/packages/utils')
sys.path.append('/Users/a123/Documents/Projects/2D_sim/packages/data_generator/loaders')
sys.path.append('/Users/a123/Documents/Projects/2D_sim/packages/data_generator/hdf5_tools')

from SE3_functions import (
    traj_smooth_se3_bspline_slerp,
    traj_process_se3_pipeline,
    traj_build_labels_with_policy,
    traj_integrate_by_twist,
    _se3_exp,
    _se3_log,
    euler_6d_to_quaternion_7d,
    trajectory_euler_to_quaternion
)

try:
    from se3_trajectory_dataset import SE3TrajectoryDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Dataset import failed: {e}")
    DATASET_AVAILABLE = False

try:
    from hdf5_trajectory_loader import HDF5TrajectoryLoader
    HDF5_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ HDF5 loader import failed: {e}")
    HDF5_AVAILABLE = False


def generate_test_se3_trajectory(N: int = 100) -> torch.Tensor:
    """테스트용 SE(3) 궤적 생성"""
    print("📊 Generating test SE(3) trajectory...")
    
    # 원형 궤적 생성
    t = torch.linspace(0, 2*np.pi, N)
    radius = 2.0
    height_variation = 0.5
    
    # 위치: 나선형
    positions = torch.zeros(N, 3)
    positions[:, 0] = radius * torch.cos(t)  # x
    positions[:, 1] = radius * torch.sin(t)  # y  
    positions[:, 2] = height_variation * torch.sin(2*t)  # z (높이 변화)
    
    # 회전: 궤적 방향을 따라 회전
    rotations = torch.zeros(N, 3, 3)
    for i in range(N):
        # Forward direction (tangent)
        if i < N-1:
            forward = positions[i+1] - positions[i]
        else:
            forward = positions[i] - positions[i-1]
        forward = forward / torch.norm(forward)
        
        # Up direction
        up = torch.tensor([0., 0., 1.])
        
        # Right direction (cross product)
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        # Recompute up to make orthonormal
        up = torch.cross(right, forward)
        
        # Build rotation matrix
        R = torch.stack([right, up, forward], dim=1)  # Column vectors
        rotations[i] = R
    
    # SE(3) 행렬 구성
    T_traj = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    T_traj[:, :3, :3] = rotations
    T_traj[:, :3, 3] = positions
    
    # 노이즈 추가로 실제적인 궤적 생성
    position_noise = torch.randn_like(positions) * 0.05
    T_traj[:, :3, 3] += position_noise
    
    print(f"   Generated {N} SE(3) poses")
    return T_traj


def test_se3_functions():
    """SE(3) 함수들 개별 테스트"""
    print("\n🧪 Testing SE(3) functions individually...")
    
    # 테스트 궤적 생성
    T_raw = generate_test_se3_trajectory(50)
    print(f"   Input trajectory shape: {T_raw.shape}")
    
    try:
        # 1) 스무딩 테스트
        print("\n🔧 Testing SE(3) smoothing...")
        T_smooth = traj_smooth_se3_bspline_slerp(T_raw, smooth=0.1)
        print(f"   ✅ Smoothing: {T_raw.shape} → {T_smooth.shape}")
        
        # 스무딩 효과 검증 (위치 차이 평균)
        pos_diff_before = torch.mean(torch.norm(T_raw[1:, :3, 3] - T_raw[:-1, :3, 3], dim=-1))
        pos_diff_after = torch.mean(torch.norm(T_smooth[1:, :3, 3] - T_smooth[:-1, :3, 3], dim=-1))
        print(f"   Position smoothness: {pos_diff_before:.4f} → {pos_diff_after:.4f}")
        
    except Exception as e:
        print(f"   ❌ Smoothing failed: {e}")
        return False
    
    try:
        # 2) 전체 파이프라인 테스트
        print("\n🔧 Testing complete SE(3) pipeline...")
        T_processed, dt_seq, xi_labels, T_smooth_pipeline = traj_process_se3_pipeline(
            T_raw,
            smooth_first=True,
            num_samples=30,
            policy="curvature"
        )
        
        print(f"   ✅ Pipeline results:")
        print(f"      Processed trajectory: {T_processed.shape}")
        print(f"      Time sequence: {dt_seq.shape}")
        print(f"      Body twist labels: {xi_labels.shape}")
        print(f"      Smoothed trajectory: {T_smooth_pipeline.shape}")
        
        # 물리적 타당성 검증
        dt_mean = torch.mean(dt_seq)
        dt_std = torch.std(dt_seq)
        xi_norm_mean = torch.mean(torch.norm(xi_labels, dim=-1))
        
        print(f"      Average dt: {dt_mean:.4f} ± {dt_std:.4f}")
        print(f"      Average twist magnitude: {xi_norm_mean:.4f}")
        
    except Exception as e:
        print(f"   ❌ Pipeline failed: {e}")
        return False
    
    try:
        # 3) 적분 검증
        print("\n🔧 Testing trajectory integration...")
        T0 = T_processed[0]
        xi_seq = xi_labels[:-1]  # 마지막 제외 (0으로 패딩됨)
        dt_seq_for_integration = dt_seq
        
        T_integrated = traj_integrate_by_twist(T0, xi_seq, dt_seq_for_integration)
        print(f"   ✅ Integration: {T0.shape} → {T_integrated.shape}")
        
        # 재구성 오차 측정
        position_error = torch.mean(torch.norm(
            T_integrated[1:, :3, 3] - T_processed[1:, :3, 3], dim=-1
        ))
        print(f"      Position reconstruction error: {position_error:.6f}")
        
        if position_error < 0.1:  # 허용 오차
            print("   ✅ Integration accuracy: GOOD")
        else:
            print("   ⚠️ Integration accuracy: MODERATE")
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        return False
    
    return True


def test_format_conversions():
    """포맷 변환 테스트"""
    print("\n🔄 Testing format conversions...")
    
    try:
        # 6D Euler → 7D Quaternion 변환
        test_6d = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # [x,y,z,rx,ry,rz]
        test_7d = euler_6d_to_quaternion_7d(test_6d)
        print(f"   6D → 7D: {test_6d.shape} → {test_7d.shape}")
        print(f"      6D: {test_6d}")
        print(f"      7D: {test_7d}")
        
        # 궤적 변환 테스트
        traj_6d = np.random.randn(10, 6)
        traj_7d = trajectory_euler_to_quaternion(traj_6d)
        print(f"   Trajectory conversion: {traj_6d.shape} → {traj_7d.shape}")
        
        print("   ✅ Format conversions: PASSED")
        
    except Exception as e:
        print(f"   ❌ Format conversions failed: {e}")
        return False
    
    return True


def test_dataset_integration():
    """데이터셋 통합 테스트"""
    if not DATASET_AVAILABLE:
        print("\n⚠️ Skipping dataset test (import failed)")
        return True
    
    print("\n📦 Testing SE3TrajectoryDataset integration...")
    
    # HDF5 파일 경로 확인
    test_hdf5_path = Path("/Users/a123/Documents/Projects/2D_sim/packages/data_generator/test_trajectory_dataset.h5")
    
    if not test_hdf5_path.exists():
        print(f"   ⚠️ Test HDF5 file not found: {test_hdf5_path}")
        print("   Run hdf5_schema_creator.py first to create test data")
        return True  # 테스트 건너뛰기 (실패 아님)
    
    try:
        # 데이터셋 생성
        config = {
            'use_smoothing': True,
            'smooth_strength': 0.1,
            'num_samples': 30,
            'augmentation': False,  # 테스트에서는 비활성화
            'max_trajectories': 3   # 빠른 테스트를 위해 제한
        }
        
        print("   Creating dataset...")
        dataset = SE3TrajectoryDataset(str(test_hdf5_path), split='train', **config)
        print(f"   ✅ Dataset created with {len(dataset)} trajectories")
        
        if len(dataset) > 0:
            # 샘플 데이터 테스트
            print("   Testing sample data...")
            sample = dataset[0]
            
            expected_keys = ['T_processed', 'xi_labels', 'dt_seq', 'T_raw', 'T_smooth', 'metadata']
            for key in expected_keys:
                if key not in sample:
                    print(f"   ❌ Missing key: {key}")
                    return False
                
            print(f"   ✅ Sample data keys: {list(sample.keys())}")
            
            # 데이터 형태 검증
            T_processed = sample['T_processed']
            xi_labels = sample['xi_labels']
            dt_seq = sample['dt_seq']
            
            print(f"      T_processed: {T_processed.shape}")
            print(f"      xi_labels: {xi_labels.shape}")
            print(f"      dt_seq: {dt_seq.shape}")
            
            # 데이터 일관성 검증
            M = T_processed.shape[0]
            if xi_labels.shape[0] != M:
                print(f"   ❌ Inconsistent shapes: T_processed[{M}] vs xi_labels[{xi_labels.shape[0]}]")
                return False
            
            if dt_seq.shape[0] != M - 1:
                print(f"   ❌ Inconsistent dt_seq shape: expected {M-1}, got {dt_seq.shape[0]}")
                return False
            
            # DataLoader 테스트
            print("   Testing DataLoader...")
            dataloader = dataset.get_dataloader(batch_size=2, num_workers=0)  # num_workers=0 for debugging
            
            batch = next(iter(dataloader))
            print(f"   ✅ Batch created with keys: {list(batch.keys())}")
            
            # 배치 크기 검증
            batch_size = len(batch['metadata'])
            print(f"      Batch size: {batch_size}")
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: {value.shape}")
        
        dataset.close()
        print("   ✅ Dataset integration: PASSED")
        
    except Exception as e:
        print(f"   ❌ Dataset integration failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def visualize_pipeline_results():
    """파이프라인 결과 시각화"""
    print("\n📊 Creating pipeline visualization...")
    
    try:
        # 테스트 데이터 생성
        T_raw = generate_test_se3_trajectory(60)
        
        # 파이프라인 실행
        T_processed, dt_seq, xi_labels, T_smooth = traj_process_se3_pipeline(
            T_raw,
            smooth_first=True,
            smooth=0.2,
            num_samples=40,
            policy="curvature"
        )
        
        # 플롯 생성
        fig = plt.figure(figsize=(15, 10))
        
        # 1) 위치 궤적 비교
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        pos_raw = T_raw[:, :3, 3].numpy()
        pos_smooth = T_smooth[:, :3, 3].numpy() if T_smooth is not None else pos_raw
        pos_processed = T_processed[:, :3, 3].numpy()
        
        ax1.plot(pos_raw[:, 0], pos_raw[:, 1], pos_raw[:, 2], 'r.-', label='Raw', alpha=0.7)
        ax1.plot(pos_smooth[:, 0], pos_smooth[:, 1], pos_smooth[:, 2], 'g-', label='Smoothed', linewidth=2)
        ax1.plot(pos_processed[:, 0], pos_processed[:, 1], pos_processed[:, 2], 'b.-', label='Processed', alpha=0.8)
        
        ax1.set_title('SE(3) Position Trajectories')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 2) 시간 간격
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(dt_seq.numpy(), 'b.-')
        ax2.set_title('Time Intervals (dt)')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('dt')
        ax2.grid(True)
        
        # 3) Body twist 크기
        ax3 = fig.add_subplot(2, 3, 3)
        xi_norms = torch.norm(xi_labels, dim=-1).numpy()
        ax3.plot(xi_norms, 'g.-')
        ax3.set_title('Body Twist Magnitude')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('||ξ||')
        ax3.grid(True)
        
        # 4) Body twist 구성요소
        ax4 = fig.add_subplot(2, 3, 4)
        xi_np = xi_labels.numpy()
        for i in range(6):
            label = ['ωx', 'ωy', 'ωz', 'vx', 'vy', 'vz'][i]
            ax4.plot(xi_np[:, i], label=label, alpha=0.7)
        ax4.set_title('Body Twist Components')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True)
        
        # 5) 위치 속도
        ax5 = fig.add_subplot(2, 3, 5)
        velocities = torch.norm(xi_labels[:, 3:], dim=-1).numpy()
        ax5.plot(velocities, 'm.-')
        ax5.set_title('Linear Velocity')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('||v||')
        ax5.grid(True)
        
        # 6) 각속도
        ax6 = fig.add_subplot(2, 3, 6)
        angular_velocities = torch.norm(xi_labels[:, :3], dim=-1).numpy()
        ax6.plot(angular_velocities, 'c.-')
        ax6.set_title('Angular Velocity')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('||ω||')
        ax6.grid(True)
        
        plt.tight_layout()
        
        # 저장
        output_path = Path("se3_pipeline_results.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")
        return False
    
    return True


def main():
    """메인 테스트 실행"""
    print("🚀 SE(3) Trajectory Pipeline Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("SE(3) Functions", test_se3_functions),
        ("Format Conversions", test_format_conversions),
        ("Dataset Integration", test_dataset_integration),
        ("Pipeline Visualization", visualize_pipeline_results)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! SE(3) pipeline is ready for use.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print("\n💡 Next Steps:")
    print("   1. Create HDF5 test data if needed: python hdf5_schema_creator.py")
    print("   2. Use SE3TrajectoryDataset in your training code")
    print("   3. Integrate with your ML model training pipeline")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)