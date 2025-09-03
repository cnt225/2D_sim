#!/usr/bin/env python3
"""
Tdot 파일 검증 스크립트
생성된 Tdot 궤적의 구조와 내용을 확인
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def verify_tdot_file(filepath):
    """Tdot 파일 검증"""
    print(f"\n🔍 Tdot 파일 검증: {filepath}")
    print("=" * 60)
    
    with h5py.File(filepath, 'r') as f:
        # 메타데이터 확인
        if 'metadata' in f:
            print("\n📋 메타데이터:")
            for key, value in f['metadata'].attrs.items():
                if 'tdot' in key.lower():
                    print(f"   {key}: {value}")
        
        # 환경 목록
        env_names = [name for name in f.keys() if name != 'metadata']
        print(f"\n📊 총 환경 수: {len(env_names)}")
        
        # 첫 번째 환경 상세 확인
        if env_names:
            env_name = env_names[0]
            env_group = f[env_name]
            
            print(f"\n🔍 샘플 환경: {env_name}")
            
            # 첫 번째 페어 확인
            pair_names = list(env_group.keys())
            if pair_names:
                pair_name = pair_names[0]
                pair_group = env_group[pair_name]
                
                print(f"   페어: {pair_name}")
                print(f"   데이터셋:")
                
                for key in pair_group.keys():
                    dataset = pair_group[key]
                    print(f"      - {key}: {dataset.shape} ({dataset.dtype})")
                
                # Tdot 궤적 상세
                if 'Tdot_trajectory' in pair_group:
                    tdot_traj = pair_group['Tdot_trajectory'][:]
                    print(f"\n   Tdot 궤적 상세:")
                    print(f"      Shape: {tdot_traj.shape}")
                    print(f"      Type: {tdot_traj.dtype}")
                    
                    if tdot_traj.ndim == 3 and tdot_traj.shape[-2:] == (4, 4):
                        # 4x4 형식
                        print(f"      형식: 4x4 행렬")
                        print(f"      첫 번째 Tdot:")
                        print(tdot_traj[0])
                        print(f"      마지막 Tdot (should be zero):")
                        print(tdot_traj[-1])
                    elif tdot_traj.ndim == 2 and tdot_traj.shape[-1] == 6:
                        # 6D 형식
                        print(f"      형식: 6D 벡터 [wx,wy,wz,vx,vy,vz]")
                        print(f"      첫 번째 Tdot: {tdot_traj[0]}")
                        print(f"      마지막 Tdot: {tdot_traj[-1]} (should be zero)")
                    
                    # 속도 통계
                    if tdot_traj.ndim == 3:
                        linear_vel = np.linalg.norm(tdot_traj[:-1, :3, 3], axis=1)
                    else:
                        linear_vel = np.linalg.norm(tdot_traj[:-1, 3:], axis=1)
                    
                    print(f"\n      선속도 통계:")
                    print(f"         평균: {np.mean(linear_vel):.4f} m/s")
                    print(f"         표준편차: {np.std(linear_vel):.4f}")
                    print(f"         최대: {np.max(linear_vel):.4f} m/s")
                    print(f"         최소: {np.min(linear_vel):.4f} m/s")
                
                # 속성 확인
                print(f"\n   속성:")
                for key, value in pair_group.attrs.items():
                    if 'tdot' in key.lower():
                        print(f"      {key}: {value}")
        
        # 전체 통계
        total_pairs = 0
        successful_tdot = 0
        failed_tdot = 0
        
        for env_name in env_names[:100]:  # 처음 100개 환경만 확인
            env_group = f[env_name]
            for pair_name in env_group.keys():
                total_pairs += 1
                pair_group = env_group[pair_name]
                if 'Tdot_trajectory' in pair_group:
                    successful_tdot += 1
                else:
                    failed_tdot += 1
        
        print(f"\n📈 전체 통계 (처음 100개 환경):")
        print(f"   총 페어: {total_pairs}")
        print(f"   Tdot 생성 성공: {successful_tdot}")
        print(f"   Tdot 생성 실패: {failed_tdot}")
        if total_pairs > 0:
            print(f"   성공률: {successful_tdot/total_pairs*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/home/dhkang225/2D_sim/data/Tdot/circles_only_integrated_trajs_Tdot.h5"
    
    if Path(filepath).exists():
        verify_tdot_file(filepath)
    else:
        print(f"❌ 파일이 없습니다: {filepath}")