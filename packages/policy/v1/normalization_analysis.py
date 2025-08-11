#!/usr/bin/env python3
"""
정규화 파이프라인 완전 분석
"""

import torch
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.append('.')
from loaders.trajectory_dataset import TrajectoryDataset

def analyze_normalization_pipeline():
    """정규화가 필요한 모든 단계 분석"""
    print("🔍 정규화 파이프라인 완전 분석")
    print("=" * 60)
    
    # 1. 학습 데이터에서 통계 추출
    print("📊 1단계: 학습 데이터 통계 추출")
    print("-" * 40)
    
    trajectory_root = "../../../data/trajectories/circle_envs_10k_bsplined"
    pointcloud_root = "../../../data/pointcloud/circle_envs_10k/circle_envs_10k"
    
    dataset = TrajectoryDataset(
        trajectory_root=trajectory_root,
        pointcloud_root=pointcloud_root,
        split='train',
        max_trajectories=100,  # 통계용으로 충분
        use_bsplined=True,
        augment_data=False,
        num_points=300
    )
    
    # Twist vector 통계 수집
    all_twists = []
    all_positions = []
    all_distances = []
    
    print(f"📈 {len(dataset)} 샘플에서 통계 수집 중...")
    
    for i in range(min(len(dataset), 2000)):  # 2000개 샘플
        if i % 500 == 0:
            print(f"   진행률: {i}/{min(len(dataset), 2000)}")
        
        try:
            sample = dataset[i]
            T_dot = sample['T_dot'].numpy()
            current_T = sample['current_T'].numpy()
            target_T = sample['target_T'].numpy()
            
            all_twists.append(T_dot)
            all_positions.append(current_T[:3, 3])
            
            # 현재-목표 거리
            distance = np.linalg.norm(target_T[:3, 3] - current_T[:3, 3])
            all_distances.append(distance)
            
        except Exception as e:
            continue
    
    all_twists = np.array(all_twists)  # [N, 6]
    all_positions = np.array(all_positions)  # [N, 3]
    all_distances = np.array(all_distances)  # [N]
    
    # 통계 계산
    twist_stats = {
        'angular': {
            'mean': np.mean(all_twists[:, :3], axis=0),
            'std': np.std(all_twists[:, :3], axis=0),
            'overall_mean': np.mean(np.linalg.norm(all_twists[:, :3], axis=1)),
            'overall_std': np.std(np.linalg.norm(all_twists[:, :3], axis=1))
        },
        'linear': {
            'mean': np.mean(all_twists[:, 3:], axis=0),
            'std': np.std(all_twists[:, 3:], axis=0),
            'overall_mean': np.mean(np.linalg.norm(all_twists[:, 3:], axis=1)),
            'overall_std': np.std(np.linalg.norm(all_twists[:, 3:], axis=1))
        },
        'total': {
            'mean': np.mean(all_twists, axis=0),
            'std': np.std(all_twists, axis=0),
            'overall_mean': np.mean(np.linalg.norm(all_twists, axis=1)),
            'overall_std': np.std(np.linalg.norm(all_twists, axis=1))
        }
    }
    
    position_stats = {
        'mean': np.mean(all_positions, axis=0),
        'std': np.std(all_positions, axis=0)
    }
    
    distance_stats = {
        'mean': np.mean(all_distances),
        'std': np.std(all_distances),
        'min': np.min(all_distances),
        'max': np.max(all_distances)
    }
    
    print(f"\n📊 추출된 통계:")
    print(f"   Angular velocity:")
    print(f"      평균 크기: {twist_stats['angular']['overall_mean']:.4f} ± {twist_stats['angular']['overall_std']:.4f} rad/s")
    print(f"      성분별 평균: {twist_stats['angular']['mean']}")
    print(f"      성분별 표준편차: {twist_stats['angular']['std']}")
    
    print(f"   Linear velocity:")
    print(f"      평균 크기: {twist_stats['linear']['overall_mean']:.4f} ± {twist_stats['linear']['overall_std']:.4f} m/s")
    print(f"      성분별 평균: {twist_stats['linear']['mean']}")
    print(f"      성분별 표준편차: {twist_stats['linear']['std']}")
    
    print(f"   Position:")
    print(f"      평균: {position_stats['mean']}")
    print(f"      표준편차: {position_stats['std']}")
    
    print(f"   Distance to target:")
    print(f"      평균: {distance_stats['mean']:.4f} ± {distance_stats['std']:.4f} m")
    print(f"      범위: [{distance_stats['min']:.4f}, {distance_stats['max']:.4f}] m")
    
    return twist_stats, position_stats, distance_stats

def create_normalization_configs(twist_stats, position_stats, distance_stats):
    """정규화 설정 파일 생성"""
    print("\n📁 2단계: 정규화 설정 파일 생성")
    print("-" * 40)
    
    # 정규화 설정
    norm_config = {
        'twist_normalization': {
            'method': 'standardization',  # (x - mean) / std
            'angular': {
                'mean': twist_stats['angular']['mean'].tolist(),
                'std': twist_stats['angular']['std'].tolist(),
                'overall_mean': float(twist_stats['angular']['overall_mean']),
                'overall_std': float(twist_stats['angular']['overall_std'])
            },
            'linear': {
                'mean': twist_stats['linear']['mean'].tolist(),
                'std': twist_stats['linear']['std'].tolist(),
                'overall_mean': float(twist_stats['linear']['overall_mean']),
                'overall_std': float(twist_stats['linear']['overall_std'])
            },
            'total': {
                'mean': twist_stats['total']['mean'].tolist(),
                'std': twist_stats['total']['std'].tolist(),
                'overall_mean': float(twist_stats['total']['overall_mean']),
                'overall_std': float(twist_stats['total']['overall_std'])
            }
        },
        'position_normalization': {
            'method': 'standardization',
            'mean': position_stats['mean'].tolist(),
            'std': position_stats['std'].tolist()
        },
        'distance_normalization': {
            'method': 'standardization',
            'mean': float(distance_stats['mean']),
            'std': float(distance_stats['std']),
            'min': float(distance_stats['min']),
            'max': float(distance_stats['max'])
        }
    }
    
    # 저장
    config_path = "configs/normalization_stats.json"
    with open(config_path, 'w') as f:
        json.dump(norm_config, f, indent=2)
    
    print(f"✅ 정규화 설정 저장: {config_path}")
    
    return norm_config

def demonstrate_normalization_pipeline(norm_config):
    """정규화 파이프라인 시연"""
    print("\n🔄 3단계: 정규화 파이프라인 시연")
    print("-" * 40)
    
    # 원본 데이터 (예시)
    original_twist = np.array([0.5, -0.2, 0.1, 3.2, 1.8, -0.5])  # [wx, wy, wz, vx, vy, vz]
    
    print(f"📥 원본 twist: {original_twist}")
    
    # 1. 학습 시 정규화 (데이터셋에서)
    twist_mean = np.array(norm_config['twist_normalization']['total']['mean'])
    twist_std = np.array(norm_config['twist_normalization']['total']['std'])
    
    normalized_twist = (original_twist - twist_mean) / twist_std
    print(f"📤 정규화된 twist: {normalized_twist}")
    print(f"   크기: {np.linalg.norm(normalized_twist):.4f} (원본: {np.linalg.norm(original_twist):.4f})")
    
    # 2. 추론 시 역정규화 (모델 출력 → 실제 twist)
    model_output = normalized_twist * 0.1  # 모델이 작게 예측한다고 가정
    denormalized_twist = model_output * twist_std + twist_mean
    
    print(f"🤖 모델 출력 (정규화됨): {model_output}")
    print(f"🔄 역정규화된 twist: {denormalized_twist}")
    print(f"   크기: {np.linalg.norm(denormalized_twist):.4f}")
    
    return norm_config

def analyze_what_needs_normalization():
    """어떤 부분에 정규화가 필요한지 분석"""
    print("\n🎯 4단계: 정규화 필요 부분 분석")
    print("-" * 40)
    
    print("📋 정규화가 필요한 단계들:")
    print()
    
    print("1️⃣ **학습 데이터 (Dataset)**")
    print("   📍 위치: trajectory_dataset.py")
    print("   🔄 정규화: T_dot (twist vector)")
    print("   📊 방법: (T_dot - mean) / std")
    print("   ⚠️ 주의: 통계는 전체 학습셋에서 미리 계산")
    print()
    
    print("2️⃣ **모델 출력 (Loss 계산)**")
    print("   📍 위치: train.py 또는 trainer")
    print("   🔄 정규화: 이미 정규화된 상태로 loss 계산")
    print("   📊 방법: MSE(predicted_normalized, target_normalized)")
    print()
    
    print("3️⃣ **추론 시 역정규화 (Inference)**")
    print("   📍 위치: inference.py")
    print("   🔄 역정규화: model_output → 실제 twist")
    print("   📊 방법: real_twist = normalized_output * std + mean")
    print("   ⚠️ 중요: 이 단계가 없으면 모델 출력이 작게 나옴!")
    print()
    
    print("4️⃣ **설정 파일 (Config)**")
    print("   📍 위치: configs/normalization_stats.json")
    print("   🔄 저장: 학습 데이터에서 계산된 통계")
    print("   📊 내용: mean, std for angular/linear velocity")
    print()
    
    print("❌ **현재 상황 분석:**")
    print("   1단계: ❌ Dataset에서 정규화 안함")
    print("   2단계: ❌ Loss 계산 시 정규화 안함")
    print("   3단계: ❌ Inference에서 역정규화 안함")
    print("   4단계: ❌ 정규화 통계 없음")
    print()
    
    print("✅ **해결 방법:**")
    print("   → 4단계 모두 구현 필요!")
    print("   → 단순히 학습 데이터만 정규화하면 추론 시 문제 발생")
    print("   → 전체 파이프라인을 일관되게 정규화해야 함")

def create_implementation_plan():
    """구현 계획 제시"""
    print("\n🚀 5단계: 구현 계획")
    print("-" * 40)
    
    print("📋 **즉시 구현 계획 (우선순위 순):**")
    print()
    
    print("🥇 **1순위: 추론 시 스케일링 (임시 해결)**")
    print("   📁 파일: inference.py")
    print("   🔧 방법: velocity_scale=50 적용")
    print("   ⏱️ 시간: 10분")
    print("   💡 효과: 즉시 100배 개선")
    print()
    
    print("🥈 **2순위: 정규화 통계 생성**")
    print("   📁 파일: configs/normalization_stats.json")
    print("   🔧 방법: 위 스크립트 실행")
    print("   ⏱️ 시간: 5분")
    print("   💡 효과: 정확한 통계 확보")
    print()
    
    print("🥉 **3순위: Dataset 정규화 구현**")
    print("   📁 파일: loaders/trajectory_dataset.py")
    print("   🔧 방법: __getitem__에서 T_dot 정규화")
    print("   ⏱️ 시간: 30분")
    print("   💡 효과: 학습 데이터 정규화")
    print()
    
    print("🏅 **4순위: Inference 역정규화 구현**")
    print("   📁 파일: inference.py")
    print("   🔧 방법: _predict_twist에서 역정규화")
    print("   ⏱️ 시간: 20분")
    print("   💡 효과: 정확한 스케일 복원")
    print()
    
    print("🎯 **재학습 (장기)**")
    print("   📁 과정: 정규화된 데이터로 전체 재학습")
    print("   ⏱️ 시간: 2-3시간")
    print("   💡 효과: 근본적 해결")
    print()
    
    print("💡 **추천 접근법:**")
    print("   1. 먼저 1순위(임시 스케일링)로 즉시 개선")
    print("   2. 그 다음 2-4순위 구현하여 정규화 파이프라인 완성")
    print("   3. 마지막에 재학습으로 근본 해결")

if __name__ == "__main__":
    print("🚀 정규화 파이프라인 완전 분석 시작")
    print()
    
    # 1. 학습 데이터 통계 분석
    twist_stats, position_stats, distance_stats = analyze_normalization_pipeline()
    
    # 2. 정규화 설정 생성
    norm_config = create_normalization_configs(twist_stats, position_stats, distance_stats)
    
    # 3. 정규화 파이프라인 시연
    demonstrate_normalization_pipeline(norm_config)
    
    # 4. 필요한 정규화 단계 분석
    analyze_what_needs_normalization()
    
    # 5. 구현 계획
    create_implementation_plan()
    
    print("\n" + "="*60)
    print("🎯 결론")
    print("="*60)
    print("❌ 학습 데이터 정규화만으로는 부족함!")
    print("✅ 전체 파이프라인 (학습 → 추론) 정규화 필요")
    print("🚀 우선 임시 스케일링으로 즉시 개선 후")
    print("🔧 완전한 정규화 파이프라인 구현 권장")
    print()
    print("✅ 분석 완료!")

