#!/usr/bin/env python3
"""
작은 샘플로 Tdot 생성 테스트
"""

import numpy as np
import torch
import h5py
from pathlib import Path
import sys

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'packages'))

from packages.utils.SE3_functions import (
    _se3_log, _so3_exp, _so3_hat
)

def create_test_trajectory():
    """테스트용 간단한 궤적 생성"""
    # 원 궤적
    t = np.linspace(0, 2*np.pi, 10)
    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)
    
    # 간단한 회전 (yaw만)
    yaw = t
    roll = np.zeros_like(t)
    pitch = np.zeros_like(t)
    
    # SE(3) 6D 형식
    trajectory = np.column_stack([x, y, z, roll, pitch, yaw])
    return trajectory

def compute_tdot(trajectory, dt=0.01):
    """Tdot 계산"""
    N = len(trajectory)
    
    # SE(3) 6D → 4x4 행렬 변환
    T_matrices = []
    for i in range(N):
        x, y, z, rx, ry, rz = trajectory[i]
        
        # 회전 행렬
        w = torch.tensor([rx, ry, rz], dtype=torch.float32)
        R = _so3_exp(w)
        
        # SE(3) 행렬
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        T_matrices.append(T)
    
    T_matrices = torch.stack(T_matrices)
    
    # Tdot 계산
    Tdot_list = []
    
    for i in range(N-1):
        T_curr = T_matrices[i]
        T_next = T_matrices[i+1]
        
        # Relative transformation
        T_rel = torch.linalg.inv(T_curr) @ T_next
        
        # Log mapping
        xi = _se3_log(T_rel.unsqueeze(0)).squeeze(0)
        
        # Velocity
        xi_vel = xi / dt
        
        # Tdot matrix
        xi_skew = torch.zeros(4, 4, dtype=torch.float32)
        xi_skew[:3, :3] = _so3_hat(xi_vel[:3])
        xi_skew[:3, 3] = xi_vel[3:]
        
        Tdot = T_curr @ xi_skew
        Tdot_list.append(Tdot.numpy())
    
    # 마지막은 0
    Tdot_list.append(np.zeros((4, 4), dtype=np.float32))
    
    return np.stack(Tdot_list)

def main():
    print("🧪 Tdot 생성 테스트")
    
    # 테스트 궤적 생성
    trajectory = create_test_trajectory()
    print(f"궤적 생성: {trajectory.shape}")
    
    # Tdot 계산
    Tdot_traj = compute_tdot(trajectory, dt=0.01)
    print(f"Tdot 계산: {Tdot_traj.shape}")
    
    # 검증
    print("\n첫 번째 Tdot:")
    print(Tdot_traj[0])
    
    print("\n마지막 Tdot (should be zero):")
    print(Tdot_traj[-1])
    
    # 속도 통계
    linear_vel = np.linalg.norm(Tdot_traj[:-1, :3, 3], axis=1)
    print(f"\n선속도 통계:")
    print(f"  평균: {np.mean(linear_vel):.4f} m/s")
    print(f"  최대: {np.max(linear_vel):.4f} m/s")
    print(f"  최소: {np.min(linear_vel):.4f} m/s")
    
    # 작은 h5 파일로 저장
    output_file = project_root / 'data' / 'Tdot' / 'test_tdot.h5'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # 메타데이터
        metadata = f.create_group('metadata')
        metadata.attrs['tdot_dt'] = 0.01
        metadata.attrs['tdot_format'] = '4x4'
        
        # 환경
        env = f.create_group('test_env')
        pair = env.create_group('pair_0')
        
        # 데이터 저장
        pair.create_dataset('raw_trajectory', data=trajectory)
        pair.create_dataset('smooth_trajectory', data=trajectory)
        pair.create_dataset('Tdot_trajectory', data=Tdot_traj)
        
        # 속성
        pair.attrs['tdot_success'] = True
        pair.attrs['tdot_points'] = len(Tdot_traj)
    
    print(f"\n✅ 테스트 파일 저장: {output_file}")
    
    # 검증
    with h5py.File(output_file, 'r') as f:
        loaded_tdot = f['test_env/pair_0/Tdot_trajectory'][:]
        print(f"로드된 Tdot shape: {loaded_tdot.shape}")
        assert np.allclose(loaded_tdot, Tdot_traj)
        print("✅ 검증 성공!")

if __name__ == "__main__":
    main()