#!/usr/bin/env python3
"""
Quick Trajectory Check Script
생성된 궤적의 몇 개 waypoint만 빠르게 확인하는 스크립트
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def load_trajectory(trajectory_file):
    """궤적 JSON 파일 로드"""
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    return data


def forward_kinematics(joint_angles, link_lengths):
    """3-link 로봇의 순기구학 계산"""
    θ1, θ2, θ3 = joint_angles
    L1, L2, L3 = link_lengths
    
    # 각 조인트의 절대 각도
    angle1 = θ1
    angle2 = θ1 + θ2
    angle3 = θ1 + θ2 + θ3
    
    # 각 조인트/링크 끝의 위치 계산
    x0, y0 = 0.0, 0.0  # Base
    x1 = L1 * np.cos(angle1)
    y1 = L1 * np.sin(angle1)
    x2 = x1 + L2 * np.cos(angle2)
    y2 = y1 + L2 * np.sin(angle2)
    x3 = x2 + L3 * np.cos(angle3)  # End-effector
    y3 = y2 + L3 * np.sin(angle3)
    
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]


def quick_check_trajectory(trajectory_file):
    """궤적의 첫 몇 개 waypoint 빠른 확인"""
    print(f"🔍 Quick checking: {trajectory_file}")
    
    data = load_trajectory(trajectory_file)
    trajectory = data['trajectory']['joint_angles']
    robot_config = data['robot']
    link_lengths = robot_config['link_lengths']
    start_goal = data['start_goal']
    
    print(f"   Total waypoints: {len(trajectory)}")
    print(f"   Start: {[f'{x:.3f}' for x in start_goal['start_config']]}")
    print(f"   Goal:  {[f'{x:.3f}' for x in start_goal['goal_config']]}")
    
    # 첫 5개와 마지막 5개 waypoint의 end-effector 위치 계산
    print(f"\n📍 First 5 waypoints (end-effector positions):")
    for i in range(min(5, len(trajectory))):
        positions = forward_kinematics(trajectory[i], link_lengths)
        end_eff_pos = positions[-1]
        print(f"   {i}: [{end_eff_pos[0]:.3f}, {end_eff_pos[1]:.3f}] | joints: {[f'{x:.3f}' for x in trajectory[i]]}")
    
    print(f"\n📍 Last 5 waypoints (end-effector positions):")
    for i in range(max(0, len(trajectory)-5), len(trajectory)):
        positions = forward_kinematics(trajectory[i], link_lengths)
        end_eff_pos = positions[-1]
        print(f"   {i}: [{end_eff_pos[0]:.3f}, {end_eff_pos[1]:.3f}] | joints: {[f'{x:.3f}' for x in trajectory[i]]}")
    
    # End-effector 경로 분석
    end_effector_path = []
    for joint_angles in trajectory:
        positions = forward_kinematics(joint_angles, link_lengths)
        end_effector_path.append(positions[-1])
    
    end_effector_path = np.array(end_effector_path)
    distances = np.linalg.norm(np.diff(end_effector_path, axis=0), axis=1)
    total_distance = np.sum(distances)
    max_step = np.max(distances)
    
    print(f"\n📊 End-effector Path Analysis:")
    print(f"   Total travel distance: {total_distance:.3f}m")
    print(f"   Max single step: {max_step:.3f}m")
    print(f"   Average step: {total_distance/(len(trajectory)-1):.3f}m")
    
    # 비정상적으로 큰 스텝 찾기
    large_steps = np.where(distances > 0.5)[0]  # 0.5m 이상 스텝
    if len(large_steps) > 0:
        print(f"   ⚠️  Large steps (>0.5m): {len(large_steps)} found")
        for step_idx in large_steps[:5]:  # 처음 5개만 출력
            print(f"      Step {step_idx}->{step_idx+1}: {distances[step_idx]:.3f}m")
    
    # 간단한 시각화
    plt.figure(figsize=(10, 5))
    
    # End-effector 경로
    plt.subplot(1, 2, 1)
    plt.plot(end_effector_path[:, 0], end_effector_path[:, 1], 'b-', linewidth=2)
    plt.plot(end_effector_path[0, 0], end_effector_path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(end_effector_path[-1, 0], end_effector_path[-1, 1], 'ro', markersize=10, label='Goal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('End-Effector Path')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    # 스텝 크기
    plt.subplot(1, 2, 2)
    plt.plot(distances, 'b-', linewidth=1)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='0.5m threshold')
    plt.xlabel('Step Index')
    plt.ylabel('Step Size (m)')
    plt.title('End-Effector Step Sizes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{trajectory_file.replace(".json", "_quick_check.png")}', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'total_distance': total_distance,
        'max_step': max_step,
        'large_steps': len(large_steps),
        'waypoints': len(trajectory)
    }


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python quick_trajectory_check.py <trajectory_file>")
        return
    
    trajectory_file = sys.argv[1]
    result = quick_check_trajectory(trajectory_file)
    
    print(f"\n📋 Summary: {result}")


if __name__ == "__main__":
    main() 