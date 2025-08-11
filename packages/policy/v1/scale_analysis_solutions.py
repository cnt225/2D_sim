#!/usr/bin/env python3
"""
모델 출력 스케일 문제 해결책들
"""

import torch
import numpy as np
import json
import sys
import os

sys.path.append('.')
from inference import MotionRFMInference, InferenceConfigs

def solution_1_scale_dt():
    """해결책 1: dt 크기 조정으로 즉시 개선"""
    print("🔧 해결책 1: dt 스케일 조정")
    print("=" * 50)
    
    engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
    
    # 테스트 설정
    device = engine.device
    start = torch.eye(4, dtype=torch.float32, device=device)
    target = torch.eye(4, dtype=torch.float32, device=device)
    target[:3, 3] = torch.tensor([2.0, 2.0, 0.0], device=device)
    pc = torch.randn(300, 3, device=device)
    
    # 현재 설정 vs 개선된 설정
    configs = {
        "현재": {"dt": 0.02, "max_steps": 500},
        "개선 1단계": {"dt": 0.1, "max_steps": 200},  # 5배 증가
        "개선 2단계": {"dt": 0.2, "max_steps": 100},  # 10배 증가
        "적극적": {"dt": 0.5, "max_steps": 50}      # 25배 증가
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n📊 {name} 설정 테스트:")
        print(f"   dt: {config['dt']}, max_steps: {config['max_steps']}")
        
        result = engine.generate_trajectory(start, target, pc, config)
        
        # 궤적 분석
        trajectory = result['trajectory']
        total_distance = 0
        for i in range(1, len(trajectory)):
            dist = torch.norm(trajectory[i][:3, 3] - trajectory[i-1][:3, 3]).item()
            total_distance += dist
        
        avg_step_size = total_distance / max(1, len(trajectory) - 1)
        
        print(f"   스텝 수: {result['steps']}")
        print(f"   총 이동: {total_distance:.4f}m")
        print(f"   평균 스텝: {avg_step_size*1000:.1f}mm")
        print(f"   시간: {result['generation_time']:.3f}초")
        print(f"   성공: {result['success']}")
        
        results[name] = {
            'dt': config['dt'],
            'steps': result['steps'],
            'total_distance': total_distance,
            'avg_step_size_mm': avg_step_size * 1000,
            'time': result['generation_time'],
            'success': result['success']
        }
    
    print(f"\n📈 결과 요약:")
    for name, res in results.items():
        print(f"   {name:12s}: {res['avg_step_size_mm']:6.1f}mm/step, "
              f"{res['total_distance']:6.3f}m 이동, {res['time']:5.2f}초")
    
    return results

def solution_2_velocity_scaling():
    """해결책 2: 추론 시 velocity 스케일링"""
    print("\n🔧 해결책 2: Velocity 스케일링")
    print("=" * 50)
    
    # 개선된 추론 엔진 (velocity scaling 포함)
    class ScaledMotionRFMInference(MotionRFMInference):
        def __init__(self, *args, velocity_scale=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.velocity_scale = velocity_scale
            print(f"✅ Velocity 스케일 팩터: {velocity_scale}")
        
        def _predict_twist(self, current_pose, target_pose, progress, pointcloud):
            # 원본 twist 예측
            twist = super()._predict_twist(current_pose, target_pose, progress, pointcloud)
            
            # 스케일링 적용
            scaled_twist = twist * self.velocity_scale
            
            return scaled_twist
    
    # 다양한 스케일 팩터 테스트
    scale_factors = [1.0, 10.0, 50.0, 100.0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = torch.eye(4, dtype=torch.float32, device=device)
    target = torch.eye(4, dtype=torch.float32, device=device)
    target[:3, 3] = torch.tensor([2.0, 2.0, 0.0], device=device)
    pc = torch.randn(300, 3, device=device)
    
    results = {}
    
    for scale in scale_factors:
        print(f"\n📊 스케일 팩터 {scale:.1f} 테스트:")
        
        engine = ScaledMotionRFMInference(
            'checkpoints/motion_rcfm_final_epoch10.pth', 
            'configs/motion_rcfm.yml',
            velocity_scale=scale
        )
        
        config = {"dt": 0.05, "max_steps": 200}  # 적당한 설정
        result = engine.generate_trajectory(start, target, pc, config)
        
        # 궤적 분석
        trajectory = result['trajectory']
        total_distance = 0
        for i in range(1, len(trajectory)):
            dist = torch.norm(trajectory[i][:3, 3] - trajectory[i-1][:3, 3]).item()
            total_distance += dist
        
        avg_step_size = total_distance / max(1, len(trajectory) - 1)
        
        print(f"   스텝 수: {result['steps']}")
        print(f"   총 이동: {total_distance:.4f}m")
        print(f"   평균 스텝: {avg_step_size*1000:.1f}mm")
        print(f"   성공: {result['success']}")
        
        results[f"scale_{scale}"] = {
            'scale': scale,
            'steps': result['steps'],
            'total_distance': total_distance,
            'avg_step_size_mm': avg_step_size * 1000,
            'success': result['success']
        }
    
    return results

def solution_3_adaptive_inference():
    """해결책 3: 적응적 추론 (거리 기반 스케일링)"""
    print("\n🔧 해결책 3: 적응적 추론")
    print("=" * 50)
    
    class AdaptiveMotionRFMInference(MotionRFMInference):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def _predict_twist(self, current_pose, target_pose, progress, pointcloud):
            # 원본 twist 예측
            twist = super()._predict_twist(current_pose, target_pose, progress, pointcloud)
            
            # 거리 기반 스케일링
            distance = torch.norm(target_pose[:3, 3] - current_pose[:3, 3])
            
            # 거리가 클수록 더 큰 속도 (하지만 제한)
            distance_scale = torch.clamp(distance * 10.0, min=1.0, max=100.0)
            
            # Progress 기반 조정 (시작할 때 더 빠르게)
            progress_scale = torch.clamp(2.0 - progress, min=0.5, max=2.0)
            
            total_scale = distance_scale * progress_scale
            scaled_twist = twist * total_scale
            
            return scaled_twist
    
    print("📊 적응적 추론 테스트:")
    
    engine = AdaptiveMotionRFMInference(
        'checkpoints/motion_rcfm_final_epoch10.pth', 
        'configs/motion_rcfm.yml'
    )
    
    device = engine.device
    start = torch.eye(4, dtype=torch.float32, device=device)
    target = torch.eye(4, dtype=torch.float32, device=device)
    target[:3, 3] = torch.tensor([3.0, 3.0, 0.0], device=device)  # 더 먼 거리
    pc = torch.randn(300, 3, device=device)
    
    config = {"dt": 0.05, "max_steps": 200}
    result = engine.generate_trajectory(start, target, pc, config)
    
    # 궤적 분석
    trajectory = result['trajectory']
    total_distance = 0
    step_sizes = []
    for i in range(1, len(trajectory)):
        dist = torch.norm(trajectory[i][:3, 3] - trajectory[i-1][:3, 3]).item()
        total_distance += dist
        step_sizes.append(dist)
    
    print(f"   스텝 수: {result['steps']}")
    print(f"   총 이동: {total_distance:.4f}m")
    print(f"   평균 스텝: {np.mean(step_sizes)*1000:.1f}mm")
    print(f"   최대 스텝: {np.max(step_sizes)*1000:.1f}mm")
    print(f"   최소 스텝: {np.min(step_sizes)*1000:.1f}mm")
    print(f"   성공: {result['success']}")
    
    return {
        'total_distance': total_distance,
        'avg_step_size_mm': np.mean(step_sizes) * 1000,
        'step_variation': np.std(step_sizes) * 1000
    }

def recommend_best_solution():
    """최적 해결책 추천"""
    print("\n" + "="*60)
    print("🎯 최적 해결책 추천")
    print("="*60)
    
    print("📋 단기 해결책 (즉시 적용 가능):")
    print("   1. dt 조정: 0.02 → 0.1 (5배 증가)")
    print("      ✅ 코드 수정 없음")
    print("      ✅ 즉시 효과")
    print("      ❌ 궤적 품질 약간 저하")
    print()
    
    print("   2. Velocity 스케일링: 50-100배")
    print("      ✅ 학습 데이터 스케일에 맞춤")
    print("      ✅ 궤적 품질 유지")
    print("      ❌ 추론 코드 수정 필요")
    print()
    
    print("📋 중기 해결책 (모델 개선):")
    print("   1. Loss 정규화 추가")
    print("      - Twist vector를 학습 데이터 평균으로 정규화")
    print("      - MSE Loss 전에 스케일 조정")
    print()
    
    print("   2. 더 나은 Loss 함수")
    print("      - Weighted MSE (linear vs angular)")
    print("      - Progress-aware loss")
    print("      - Distance-aware loss")
    print()
    
    print("📋 장기 해결책 (재학습):")
    print("   1. 정규화된 데이터로 재학습")
    print("   2. 더 큰 모델 (velocity field 네트워크 확장)")
    print("   3. 다른 학습률 스케줄")
    print()
    
    print("🚀 권장 즉시 적용:")
    print("   InferenceConfigs에 velocity_scale=50.0 추가")
    print("   dt=0.05로 조정")
    print("   → 학습 데이터 스케일에 근접한 성능 기대")

if __name__ == "__main__":
    print("🚀 모델 출력 스케일 문제 해결책 분석")
    print()
    
    # 1. dt 조정 테스트
    dt_results = solution_1_scale_dt()
    
    # 2. velocity 스케일링 테스트  
    scale_results = solution_2_velocity_scaling()
    
    # 3. 적응적 추론 테스트
    adaptive_results = solution_3_adaptive_inference()
    
    # 4. 최적 해결책 추천
    recommend_best_solution()
    
    print("\n✅ 모든 해결책 분석 완료!")
    print("📁 다음 단계: 선택한 해결책 구현 및 테스트")

