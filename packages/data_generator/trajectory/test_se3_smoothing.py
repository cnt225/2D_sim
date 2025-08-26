#!/usr/bin/env python3
"""
Test SE(3) smoothing functionality
궤적 스무딩 기능 테스트 스크립트
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'packages'))

# Import SE(3) functions
# Access the SE3_functions from packages/utils
from utils.SE3_functions import (
    traj_smooth_se3_bspline_slerp,
    traj_resample_by_arclength,
    traj_process_se3_pipeline
)

def create_test_trajectory_se2():
    """테스트용 SE(2) 궤적 생성 (지그재그 패턴)"""
    # Create a zigzag trajectory
    waypoints = []
    for i in range(10):
        x = i * 0.5
        y = 0.3 * np.sin(i * np.pi / 2)
        theta = np.arctan2(y, x) if i > 0 else 0
        waypoints.append([x, y, theta])
    
    return np.array(waypoints)

def se2_to_se3_matrices(traj_se2):
    """SE(2) 궤적을 SE(3) 행렬로 변환"""
    N = len(traj_se2)
    T_se3 = torch.zeros((N, 4, 4), dtype=torch.float32)
    
    for i in range(N):
        x, y, theta = traj_se2[i]
        # SE(3) 변환 행렬 생성
        T_se3[i, 0, 0] = np.cos(theta)
        T_se3[i, 0, 1] = -np.sin(theta)
        T_se3[i, 1, 0] = np.sin(theta)
        T_se3[i, 1, 1] = np.cos(theta)
        T_se3[i, 2, 2] = 1.0  # z축은 단위 행렬
        T_se3[i, 0, 3] = x
        T_se3[i, 1, 3] = y
        T_se3[i, 2, 3] = 0.0  # z = 0 (2D 평면)
        T_se3[i, 3, 3] = 1.0
    
    return T_se3

def se3_to_se2_trajectory(T_se3):
    """SE(3) 행렬을 SE(2) 궤적으로 변환"""
    N = T_se3.shape[0]
    traj_se2 = np.zeros((N, 3))
    
    for i in range(N):
        # 위치 추출
        traj_se2[i, 0] = T_se3[i, 0, 3].item()  # x
        traj_se2[i, 1] = T_se3[i, 1, 3].item()  # y
        # yaw 각도 추출 (z축 회전)
        traj_se2[i, 2] = np.arctan2(T_se3[i, 1, 0].item(), T_se3[i, 0, 0].item())
    
    return traj_se2

def visualize_trajectories(traj_raw, traj_smooth, title="SE(3) Smoothing Test"):
    """궤적 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 위치 궤적
    ax1.plot(traj_raw[:, 0], traj_raw[:, 1], 'o-', label='Raw', alpha=0.5, markersize=8)
    ax1.plot(traj_smooth[:, 0], traj_smooth[:, 1], '-', label='Smoothed', linewidth=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Position Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 방향 각도
    ax2.plot(range(len(traj_raw)), traj_raw[:, 2], 'o-', label='Raw', alpha=0.5, markersize=8)
    ax2.plot(range(len(traj_smooth)), traj_smooth[:, 2], '-', label='Smoothed', linewidth=2)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Theta (rad)')
    ax2.set_title('Orientation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_se3_smoothing():
    """SE(3) 스무딩 테스트"""
    print("🧪 SE(3) Smoothing Test")
    print("=" * 50)
    
    # 1. 테스트 궤적 생성
    print("1. Creating test trajectory...")
    traj_raw_se2 = create_test_trajectory_se2()
    print(f"   Raw trajectory: {len(traj_raw_se2)} points")
    
    # 2. SE(2) → SE(3) 변환
    print("2. Converting SE(2) to SE(3)...")
    T_se3_raw = se2_to_se3_matrices(traj_raw_se2)
    print(f"   SE(3) matrices shape: {T_se3_raw.shape}")
    
    # 3. SE(3) 스무딩
    print("3. Applying SE(3) smoothing (B-spline + SLERP)...")
    T_se3_smooth = traj_smooth_se3_bspline_slerp(
        T_se3_raw,
        pos_method="bspline_scipy",
        degree=3,
        smooth=0.01
    )
    print(f"   Smoothed trajectory: {T_se3_smooth.shape}")
    
    # 4. Arc-length 기반 재샘플링
    print("4. Resampling by arc-length...")
    T_se3_resampled, s_values = traj_resample_by_arclength(
        T_se3_smooth,
        num_samples=50,
        lambda_rot=0.1
    )
    print(f"   Resampled trajectory: {T_se3_resampled.shape}")
    print(f"   Arc-length range: [{s_values[0]:.3f}, {s_values[-1]:.3f}]")
    
    # 5. SE(3) → SE(2) 변환
    print("5. Converting back to SE(2)...")
    traj_smooth_se2 = se3_to_se2_trajectory(T_se3_resampled)
    
    # 6. 결과 비교
    print("\n📊 Results:")
    print(f"   Original: {len(traj_raw_se2)} points")
    print(f"   Smoothed: {len(traj_smooth_se2)} points")
    
    # 경로 길이 계산
    raw_path_length = np.sum(np.linalg.norm(
        np.diff(traj_raw_se2[:, :2], axis=0), axis=1
    ))
    smooth_path_length = np.sum(np.linalg.norm(
        np.diff(traj_smooth_se2[:, :2], axis=0), axis=1
    ))
    
    print(f"   Raw path length: {raw_path_length:.3f}")
    print(f"   Smooth path length: {smooth_path_length:.3f}")
    print(f"   Length ratio: {smooth_path_length/raw_path_length:.3f}")
    
    # 7. 시각화
    print("\n📈 Visualizing trajectories...")
    visualize_trajectories(traj_raw_se2, traj_smooth_se2)
    
    print("\n✅ SE(3) smoothing test completed successfully!")
    return True

def test_complete_pipeline():
    """Complete SE(3) pipeline 테스트"""
    print("\n🚀 Testing Complete SE(3) Pipeline")
    print("=" * 50)
    
    # 테스트 궤적 생성
    traj_raw_se2 = create_test_trajectory_se2()
    T_se3_raw = se2_to_se3_matrices(traj_raw_se2)
    
    # Complete pipeline 실행
    print("Running complete SE(3) processing pipeline...")
    T_processed, dt_seq, xi_labels = traj_process_se3_pipeline(
        T_se3_raw,
        smooth_first=True,
        pos_method="bspline_scipy",
        degree=3,
        smooth=0.01,
        num_samples=50,
        lambda_rot=0.1,
        policy="curvature",
        v_ref=0.4,
        v_cap=0.5,
        a_lat_max=1.0
    )
    
    print(f"✅ Pipeline results:")
    print(f"   Processed trajectory: {T_processed.shape}")
    print(f"   Time deltas: {dt_seq.shape} (sum={dt_seq.sum():.3f}s)")
    print(f"   Body twist labels: {xi_labels.shape}")
    
    # 평균 속도 계산
    avg_linear_vel = torch.linalg.norm(xi_labels[:, 3:], dim=-1).mean()
    avg_angular_vel = torch.linalg.norm(xi_labels[:, :3], dim=-1).mean()
    print(f"   Average linear velocity: {avg_linear_vel:.3f} m/s")
    print(f"   Average angular velocity: {avg_angular_vel:.3f} rad/s")
    
    return True

if __name__ == "__main__":
    try:
        # Basic SE(3) smoothing test
        success = test_se3_smoothing()
        
        # Complete pipeline test
        success = success and test_complete_pipeline()
        
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()