#!/usr/bin/env python3
"""
학습 데이터의 twist vector 크기와 분포 분석
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os
import sys
from pathlib import Path

sys.path.append('.')
from loaders.trajectory_dataset import TrajectoryDataset

def analyze_training_data():
    """학습 데이터의 twist vector 분석"""
    print("🔍 학습 데이터 분석")
    print("=" * 50)
    
    # 데이터셋 로드
    trajectory_root = "../../../data/trajectories/circle_envs_10k_bsplined"
    pointcloud_root = "../../../data/pointcloud/circle_envs_10k/circle_envs_10k"
    
    print(f"📂 궤적 데이터: {trajectory_root}")
    print(f"📂 포인트클라우드: {pointcloud_root}")
    
    # 작은 서브셋으로 빠른 분석
    dataset = TrajectoryDataset(
        trajectory_root=trajectory_root,
        pointcloud_root=pointcloud_root,
        split='train',
        max_trajectories=50,  # 빠른 분석을 위해 50개만
        use_bsplined=True,
        augment_data=False,
        num_points=300
    )
    
    print(f"✅ 데이터셋 로드 완료: {len(dataset)} 샘플")
    print()
    
    # Twist vector 통계 수집
    twist_norms = []
    linear_norms = []
    angular_norms = []
    timestamps = []
    waypoint_distances = []
    
    print("📊 샘플 분석 중...")
    
    for i in range(min(len(dataset), 1000)):  # 1000개 샘플만 분석
        if i % 100 == 0:
            print(f"   진행률: {i}/{min(len(dataset), 1000)}")
        
        try:
            sample = dataset[i]
            T_dot = sample['T_dot'].numpy()
            
            # Angular part (first 3)
            angular = T_dot[:3]
            angular_norm = np.linalg.norm(angular)
            
            # Linear part (last 3)
            linear = T_dot[3:]
            linear_norm = np.linalg.norm(linear)
            
            # Total norm
            total_norm = np.linalg.norm(T_dot)
            
            twist_norms.append(total_norm)
            linear_norms.append(linear_norm)
            angular_norms.append(angular_norm)
            
            # 시간 정보
            time_t = sample['time_t'].item()
            timestamps.append(time_t)
            
            # 현재-목표 거리
            current_T = sample['current_T'].numpy()
            target_T = sample['target_T'].numpy()
            distance = np.linalg.norm(target_T[:3, 3] - current_T[:3, 3])
            waypoint_distances.append(distance)
            
        except Exception as e:
            continue
    
    # 통계 계산
    twist_norms = np.array(twist_norms)
    linear_norms = np.array(linear_norms)
    angular_norms = np.array(angular_norms)
    timestamps = np.array(timestamps)
    waypoint_distances = np.array(waypoint_distances)
    
    print(f"\n📈 Twist Vector 통계 (총 {len(twist_norms)}개 샘플):")
    print(f"   🔄 Angular velocity:")
    print(f"      평균: {np.mean(angular_norms):.6f} rad/s")
    print(f"      표준편차: {np.std(angular_norms):.6f}")
    print(f"      최대: {np.max(angular_norms):.6f}")
    print(f"      최소: {np.min(angular_norms):.6f}")
    print()
    
    print(f"   📐 Linear velocity:")
    print(f"      평균: {np.mean(linear_norms):.6f} m/s")
    print(f"      표준편차: {np.std(linear_norms):.6f}")
    print(f"      최대: {np.max(linear_norms):.6f}")
    print(f"      최소: {np.min(linear_norms):.6f}")
    print()
    
    print(f"   📊 Total twist:")
    print(f"      평균: {np.mean(twist_norms):.6f}")
    print(f"      표준편차: {np.std(twist_norms):.6f}")
    print(f"      최대: {np.max(twist_norms):.6f}")
    print(f"      최소: {np.min(twist_norms):.6f}")
    print()
    
    print(f"   📏 거리 정보:")
    print(f"      평균 현재-목표 거리: {np.mean(waypoint_distances):.6f} m")
    print(f"      표준편차: {np.std(waypoint_distances):.6f}")
    print(f"      최대: {np.max(waypoint_distances):.6f}")
    print(f"      최소: {np.min(waypoint_distances):.6f}")
    
    return {
        'twist_norms': twist_norms,
        'linear_norms': linear_norms,
        'angular_norms': angular_norms,
        'timestamps': timestamps,
        'waypoint_distances': waypoint_distances
    }

def analyze_single_trajectory():
    """단일 궤적 상세 분석"""
    print("\n🔍 단일 궤적 상세 분석")
    print("=" * 50)
    
    # 하나의 궤적 파일 직접 로드  
    traj_file = "../../../data/trajectories/circle_envs_10k_bsplined/circle_env_000000_pair_1_traj_rb3_bsplined.json"
    
    if not os.path.exists(traj_file):
        print(f"❌ 파일 없음: {traj_file}")
        return
    
    with open(traj_file, 'r') as f:
        traj_data = json.load(f)
    
    path_data = traj_data.get('path', {})
    poses_flat = path_data.get('data', [])
    timestamps = path_data.get('timestamps', [])
    
    print(f"📄 파일: {os.path.basename(traj_file)}")
    print(f"📊 웨이포인트 수: {len(poses_flat)}")
    print(f"⏱️ 타임스탬프 수: {len(timestamps)}")
    
    # 각 웨이포인트 간 거리와 시간 간격 분석
    if len(poses_flat) < 2:
        print("❌ 웨이포인트가 부족합니다")
        return
    
    step_distances = []
    time_intervals = []
    computed_velocities = []
    
    for i in range(len(poses_flat) - 1):
        # 현재와 다음 포즈
        pose1 = poses_flat[i][:3]  # [x, y, z]
        pose2 = poses_flat[i + 1][:3]
        
        # 거리
        distance = np.linalg.norm(np.array(pose2) - np.array(pose1))
        step_distances.append(distance)
        
        # 시간 간격
        if i < len(timestamps) - 1:
            dt = timestamps[i + 1] - timestamps[i]
        else:
            dt = 0.1  # 기본값
        time_intervals.append(dt)
        
        # 속도
        if dt > 0:
            velocity = distance / dt
            computed_velocities.append(velocity)
    
    step_distances = np.array(step_distances)
    time_intervals = np.array(time_intervals)
    computed_velocities = np.array(computed_velocities)
    
    print(f"\n📏 웨이포인트 간 거리:")
    print(f"   평균: {np.mean(step_distances):.6f} m")
    print(f"   표준편차: {np.std(step_distances):.6f}")
    print(f"   최대: {np.max(step_distances):.6f}")
    print(f"   최소: {np.min(step_distances):.6f}")
    
    print(f"\n⏱️ 시간 간격:")
    print(f"   평균: {np.mean(time_intervals):.6f} s")
    print(f"   표준편차: {np.std(time_intervals):.6f}")
    print(f"   최대: {np.max(time_intervals):.6f}")
    print(f"   최소: {np.min(time_intervals):.6f}")
    
    print(f"\n🚀 계산된 속도:")
    print(f"   평균: {np.mean(computed_velocities):.6f} m/s")
    print(f"   표준편차: {np.std(computed_velocities):.6f}")
    print(f"   최대: {np.max(computed_velocities):.6f}")
    print(f"   최소: {np.min(computed_velocities):.6f}")
    
    return {
        'step_distances': step_distances,
        'time_intervals': time_intervals,
        'computed_velocities': computed_velocities
    }

def check_normalization_in_training():
    """학습 코드에서 정규화 여부 확인"""
    print("\n🔍 학습 과정 정규화 확인")
    print("=" * 50)
    
    # Train 스크립트 확인
    train_files = glob.glob("train*.py")
    
    for train_file in train_files:
        print(f"📄 {train_file} 확인 중...")
        
        with open(train_file, 'r') as f:
            content = f.read()
        
        # 정규화 관련 키워드 검색
        keywords = ['normalize', 'norm', 'scale', 'std', 'mean']
        found_lines = []
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword.lower() in line.lower() and 'T_dot' in line:
                    found_lines.append(f"   Line {i+1}: {line.strip()}")
        
        if found_lines:
            print(f"   🔍 정규화 관련 코드 발견:")
            for line in found_lines[:5]:  # 최대 5개만 표시
                print(line)
        else:
            print(f"   ❌ 정규화 관련 코드 없음")
        print()

def visualize_data_distribution(data):
    """데이터 분포 시각화"""
    print("\n📊 데이터 분포 시각화")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Linear velocity 분포
    axes[0, 0].hist(data['linear_norms'], bins=50, alpha=0.7)
    axes[0, 0].set_title('Linear Velocity Distribution')
    axes[0, 0].set_xlabel('Speed (m/s)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(data['linear_norms']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(data["linear_norms"]):.4f}')
    axes[0, 0].legend()
    
    # 2. Angular velocity 분포
    axes[0, 1].hist(data['angular_norms'], bins=50, alpha=0.7)
    axes[0, 1].set_title('Angular Velocity Distribution')
    axes[0, 1].set_xlabel('Angular Speed (rad/s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(data['angular_norms']), color='red', linestyle='--',
                       label=f'Mean: {np.mean(data["angular_norms"]):.4f}')
    axes[0, 1].legend()
    
    # 3. 거리 vs Linear velocity
    axes[1, 0].scatter(data['waypoint_distances'], data['linear_norms'], alpha=0.5)
    axes[1, 0].set_xlabel('Distance to Target (m)')
    axes[1, 0].set_ylabel('Linear Velocity (m/s)')
    axes[1, 0].set_title('Distance vs Linear Velocity')
    
    # 4. Time vs Linear velocity
    axes[1, 1].scatter(data['timestamps'], data['linear_norms'], alpha=0.5)
    axes[1, 1].set_xlabel('Normalized Time')
    axes[1, 1].set_ylabel('Linear Velocity (m/s)')
    axes[1, 1].set_title('Time vs Linear Velocity')
    
    plt.tight_layout()
    plt.savefig('training_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 시각화 저장: training_data_analysis.png")

if __name__ == "__main__":
    print("🚀 학습 데이터 분석 시작")
    print()
    
    # 1. 학습 데이터 전체 분석
    data = analyze_training_data()
    
    # 2. 단일 궤적 상세 분석
    single_traj_data = analyze_single_trajectory()
    
    # 3. 학습 코드 정규화 확인
    check_normalization_in_training()
    
    # 4. 시각화
    if data and len(data['twist_norms']) > 0:
        visualize_data_distribution(data)
    
    print("\n" + "="*60)
    print("🎯 학습 데이터 분석 결과")
    print("="*60)
    
    if data:
        avg_linear = np.mean(data['linear_norms'])
        avg_angular = np.mean(data['angular_norms'])
        
        print(f"📊 핵심 발견:")
        print(f"   1. 학습 데이터 평균 선속도: {avg_linear:.6f} m/s")
        print(f"   2. 모델 예측 평균 선속도: ~0.06 m/s")
        print(f"   3. 차이 비율: {0.06/avg_linear:.2f}배")
        print()
        
        if avg_linear > 0.1:
            print("✅ 학습 데이터는 충분히 큰 속도를 가지고 있음")
            print("❌ 모델이 학습 데이터보다 훨씬 작은 속도 예측")
            print("🔧 가능한 원인:")
            print("   - 손실 함수에서 과도한 정규화")
            print("   - 학습률이나 배치 크기 문제") 
            print("   - 모델 용량 부족")
        else:
            print("❌ 학습 데이터 자체가 작은 속도를 가짐")
            print("🔧 데이터 스케일 조정 필요")
    
    print("\n✅ 분석 완료!")
