#!/usr/bin/env python3
"""
Simple Trajectory Visualizer for New HDF5 Structure
새로운 HDF5 구조 (circles_only_trajs.h5)에서 궤적 시각화

사용법:
    python simple_trajectory_visualizer.py <env_name> <pair_id>
    
예시:
    python simple_trajectory_visualizer.py circle_env_000006 1
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def load_trajectory_from_new_hdf5(env_name: str, pair_id: str) -> Dict[str, Any]:
    """
    새로운 HDF5 구조에서 궤적 데이터 로드
    
    Args:
        env_name: 환경 이름 (예: circle_env_000006)
        pair_id: 페어 ID (예: "1")
        
    Returns:
        trajectory_data: 궤적 데이터 딕셔너리
    """
    project_root = Path(__file__).parent.parent.parent.parent.parent
    h5_file_path = project_root / "data" / "trajectory" / "circles_only_trajs.h5"
    
    if not h5_file_path.exists():
        raise FileNotFoundError(f"HDF5 파일을 찾을 수 없습니다: {h5_file_path}")
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if env_name not in f:
                raise ValueError(f"환경 '{env_name}'을 찾을 수 없습니다")
            
            env_group = f[env_name]
            if pair_id not in env_group:
                raise ValueError(f"환경 '{env_name}'에서 페어 '{pair_id}'를 찾을 수 없습니다")
            
            pair_group = env_group[pair_id]
            
            # 궤적 데이터 로드
            raw_trajectory = pair_group['raw_trajectory'][:]
            
            # 메타데이터 로드
            metadata = {}
            for attr_name in pair_group.attrs.keys():
                metadata[attr_name] = pair_group.attrs[attr_name]
            
            print(f"✅ 궤적 데이터 로드 완료: {env_name}/{pair_id}")
            print(f"   Raw trajectory: {len(raw_trajectory)}개 점")
            print(f"   경로 길이: {metadata.get('path_length', 'N/A'):.3f}m")
            print(f"   생성 시간: {metadata.get('generation_time', 'N/A'):.3f}초")
            print(f"   시작 pose: {metadata.get('start_pose', 'N/A')}")
            print(f"   끝 pose: {metadata.get('end_pose', 'N/A')}")
            
            return {
                'raw_trajectory': raw_trajectory,
                'metadata': metadata
            }
            
    except Exception as e:
        raise RuntimeError(f"궤적 데이터 로드 실패: {e}")

def load_environment_pointcloud(env_name: str) -> np.ndarray:
    """
    환경 이름으로부터 포인트클라우드 파일 로드
    
    Args:
        env_name: 환경 이름
        
    Returns:
        points: 환경 포인트클라우드 (N x 2)
    """
    project_root = Path(__file__).parent.parent.parent.parent.parent
    ply_path = project_root / "data" / "pointcloud" / "circles_only" / f"{env_name}.ply"
    
    if not ply_path.exists():
        print(f"⚠️ 환경 파일을 찾을 수 없습니다: {ply_path}")
        return np.array([])
    
    print(f"📁 환경 로딩: {ply_path}")
    return load_ply_file(str(ply_path))

def load_ply_file(ply_file: str) -> np.ndarray:
    """PLY 파일에서 포인트 데이터 로드"""
    
    points = []
    try:
        with open(ply_file, 'r') as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == 'end_header':
                        in_header = False
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    points.append([x, y])
    
    except Exception as e:
        print(f"PLY 파일 로드 오류: {e}")
        return np.array([])
    
    return np.array(points)

def visualize_trajectory(env_name: str, pair_id: str, save_image: bool = True) -> str:
    """
    궤적 시각화
    
    Args:
        env_name: 환경 이름
        pair_id: 페어 ID
        save_image: 이미지 저장 여부
        
    Returns:
        output_file: 저장된 파일 경로
    """
    # 데이터 로드
    trajectory_data = load_trajectory_from_new_hdf5(env_name, pair_id)
    environment_points = load_environment_pointcloud(env_name)
    
    # 그래프 설정
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 환경 그리기
    if len(environment_points) > 0:
        ax.scatter(environment_points[:, 0], environment_points[:, 1], 
                  c='red', s=1, alpha=0.6, label='Environment Obstacles')
    
    # 궤적 데이터 추출
    raw_trajectory = trajectory_data['raw_trajectory']
    metadata = trajectory_data['metadata']
    
    # 시작점과 끝점
    start_pose = metadata['start_pose']
    end_pose = metadata['end_pose']
    
    # 원본 궤적 그리기
    ax.plot(raw_trajectory[:, 0], raw_trajectory[:, 1], 
            'b-o', linewidth=2, markersize=2, alpha=0.8, 
            label=f'RRT Trajectory ({len(raw_trajectory)} points)')
    
    # 시작점과 끝점 표시
    ax.plot(start_pose[0], start_pose[1], 
            'go', markersize=12, markeredgecolor='black', linewidth=2, label='Start')
    ax.plot(end_pose[0], end_pose[1], 
            'ro', markersize=12, markeredgecolor='black', linewidth=2, label='Goal')
    
    # 방향 화살표
    arrow_len = 0.4
    ax.arrow(start_pose[0], start_pose[1], 
             arrow_len * np.cos(start_pose[2]), arrow_len * np.sin(start_pose[2]),
             head_width=0.15, head_length=0.15, fc='green', ec='darkgreen', linewidth=2)
    ax.arrow(end_pose[0], end_pose[1], 
             arrow_len * np.cos(end_pose[2]), arrow_len * np.sin(end_pose[2]),
             head_width=0.15, head_length=0.15, fc='red', ec='darkred', linewidth=2)
    
    # 그래프 설정
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    
    # 제목 설정
    planning_time = metadata.get('generation_time', 0.0)
    path_length = metadata.get('path_length', 0.0)
    waypoint_count = metadata.get('waypoint_count', 0)
    
    ax.set_title(f'RRT Trajectory: {env_name} / Pair {pair_id}\n'
                 f'Planning Time: {planning_time:.3f}s, Path Length: {path_length:.3f}m, '
                 f'Waypoints: {waypoint_count}', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 저장
    output_file = None
    if save_image:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        output_dir = project_root / "data" / "visualized" / "trajectory"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{env_name}_pair_{pair_id}_trajectory.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 궤적 시각화 저장: {output_file}")
    
    plt.show()
    
    return str(output_file) if output_file else None

def main():
    """메인 함수"""
    
    if len(sys.argv) != 3:
        print("사용법: python simple_trajectory_visualizer.py <env_name> <pair_id>")
        print("예시: python simple_trajectory_visualizer.py circle_env_000006 1")
        return 1
    
    env_name = sys.argv[1]
    pair_id = sys.argv[2]
    
    try:
        print(f"🎨 궤적 시각화: {env_name} / Pair {pair_id}")
        
        output_file = visualize_trajectory(env_name, pair_id, save_image=True)
        
        print("✅ 시각화 완료!")
        return 0
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
