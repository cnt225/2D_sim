#!/usr/bin/env python3
"""
간단한 궤적 시각화 스크립트
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_ply_file(ply_path):
    """PLY 파일에서 2D 포인트들 로드"""
    points = []
    try:
        with open(ply_path, 'r') as f:
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
        print(f"PLY 파일 로드 실패: {e}")
        return np.array([])
    
    return np.array(points)

def visualize_trajectory(trajectory_json_path):
    """궤적과 환경을 함께 시각화"""
    
    # JSON 파일 로드
    with open(trajectory_json_path, 'r') as f:
        data = json.load(f)
    
    # 궤적 데이터 추출
    trajectory_poses = data['path']['data']  # [[x, y, z, roll, pitch, yaw], ...]
    start_pose = data['start_pose']
    goal_pose = data['goal_pose']
    environment_info = data['environment']
    rigid_body_info = data['rigid_body']
    
    print(f"📊 궤적 정보:")
    print(f"   포즈 개수: {len(trajectory_poses)}")
    print(f"   시작점: [{start_pose[0]:.3f}, {start_pose[1]:.3f}, {start_pose[5]:.3f}°]")
    print(f"   목표점: [{goal_pose[0]:.3f}, {goal_pose[1]:.3f}, {goal_pose[5]:.3f}°]")
    print(f"   환경: {environment_info['name']}")
    print(f"   로봇: {rigid_body_info['type']} (ID: {rigid_body_info['id']})")
    
    # 환경 포인트클라우드 로드
    env_points = load_ply_file(environment_info['ply_file'])
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 환경 포인트클라우드 그리기
    if len(env_points) > 0:
        plt.scatter(env_points[:, 0], env_points[:, 1], 
                   c='red', s=1, alpha=0.6, label='Environment')
        print(f"   환경 포인트 개수: {len(env_points)}")
    else:
        print("   ⚠️ 환경 데이터 없음")
    
    # 궤적 경로 그리기
    x_coords = [pose[0] for pose in trajectory_poses]
    y_coords = [pose[1] for pose in trajectory_poses]
    
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Trajectory Path')
    plt.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.6, label='Waypoints')
    
    # 시작점과 목표점 강조
    plt.scatter(start_pose[0], start_pose[1], c='green', s=100, marker='o', 
               label='Start', edgecolors='black', linewidth=2)
    plt.scatter(goal_pose[0], goal_pose[1], c='red', s=100, marker='*', 
               label='Goal', edgecolors='black', linewidth=2)
    
    # 로봇 크기 표시 (시작점과 목표점에)
    # Rigid body ID 3 (elongated_ellipse): semi_major=1.2m, semi_minor=0.4m
    from matplotlib.patches import Ellipse
    
    # 시작점 로봇 표시
    start_ellipse = Ellipse((start_pose[0], start_pose[1]), 
                           width=2.4, height=0.8, angle=np.degrees(start_pose[5]),
                           facecolor='green', alpha=0.3, edgecolor='black', linewidth=2)
    plt.gca().add_patch(start_ellipse)
    
    # 목표점 로봇 표시
    goal_ellipse = Ellipse((goal_pose[0], goal_pose[1]), 
                          width=2.4, height=0.8, angle=np.degrees(goal_pose[5]),
                          facecolor='red', alpha=0.3, edgecolor='black', linewidth=2)
    plt.gca().add_patch(goal_ellipse)
    
    # 중간 몇 개 포즈도 표시
    step = max(1, len(trajectory_poses) // 8)  # 최대 8개 중간 포즈
    for i in range(step, len(trajectory_poses)-step, step):
        pose = trajectory_poses[i]
        mid_ellipse = Ellipse((pose[0], pose[1]), 
                             width=2.4, height=0.8, angle=np.degrees(pose[5]),
                             facecolor='orange', alpha=0.2, edgecolor='gray', linewidth=1)
        plt.gca().add_patch(mid_ellipse)
    
    # 그래프 설정
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'추론된 궤적 시각화\n{rigid_body_info["type"]} in {environment_info["name"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 저장
    output_path = f"visualized_{data['trajectory_id']}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 시각화 저장됨: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        trajectory_file = sys.argv[1]
    else:
        trajectory_file = "inference_results/inferred_traj_rb3_20250820_092809.json"
    visualize_trajectory(trajectory_file)
