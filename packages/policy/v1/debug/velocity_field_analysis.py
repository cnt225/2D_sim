#!/usr/bin/env python3
"""
Velocity Field 분석 - 정확한 문제 진단
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('.')
from inference import MotionRFMInference, InferenceConfigs

def analyze_twist_vector():
    """Twist vector 형식과 크기 분석"""
    print("🔍 TWIST VECTOR 분석")
    print("=" * 50)
    
    # 엔진 로드
    engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
    
    # 간단한 테스트 케이스 (CUDA 디바이스로)
    device = engine.device
    start = torch.eye(4, dtype=torch.float32, device=device)
    target = torch.eye(4, dtype=torch.float32, device=device)
    target[:3, 3] = torch.tensor([1.0, 0.0, 0.0], device=device)  # 1m X방향 이동
    pc = torch.randn(300, 3, device=device)
    
    print(f"📍 시작 위치: {start[:3, 3].cpu().numpy()}")
    print(f"📍 목표 위치: {target[:3, 3].cpu().numpy()}")
    print(f"📏 직선 거리: {torch.norm(target[:3, 3] - start[:3, 3]).item():.3f}m")
    print()
    
    # 여러 progress 값에서 twist 분석
    progress_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for prog in progress_values:
        progress = torch.tensor(prog, device=device)
        twist = engine._predict_twist(start, target, progress, pc)
        
        w = twist[:3]  # angular velocity [rad/s]
        v = twist[3:]  # linear velocity [m/s]
        
        w_norm = torch.norm(w).item()
        v_norm = torch.norm(v).item()
        
        print(f"⏱️ Progress: {prog:.2f}")
        print(f"   🔄 Angular velocity: [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}] (norm: {w_norm:.4f} rad/s)")
        print(f"   📐 Linear velocity:  [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}] (norm: {v_norm:.4f} m/s)")
        print(f"   📊 Total twist norm: {torch.norm(twist).item():.4f}")
        print()
    
    return engine

def analyze_integration_step():
    """적분 스텝 크기 분석"""
    print("🔍 INTEGRATION STEP 분석")
    print("=" * 50)
    
    # 다양한 dt 값으로 테스트
    dt_values = [0.001, 0.01, 0.02, 0.05, 0.1]
    
    for dt in dt_values:
        # 단위 twist (정규화된)
        twist = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # 1 m/s X방향
        
        # 적분 후 이동 거리
        w = twist[:3] * dt
        v = twist[3:] * dt
        
        distance_moved = torch.norm(v).item()
        
        print(f"⏱️ dt = {dt:.3f}s")
        print(f"   📏 이동 거리: {distance_moved:.6f}m = {distance_moved*1000:.3f}mm")
        print(f"   🔄 회전각: {torch.norm(w).item():.6f} rad = {torch.rad2deg(torch.norm(w)).item():.6f}°")
        print()

def test_different_distances():
    """다양한 거리에서 twist 크기 분석"""
    print("🔍 거리별 TWIST 크기 분석")
    print("=" * 50)
    
    engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
    
    # 다양한 거리로 테스트
    distances = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for dist in distances:
        device = engine.device
        start = torch.eye(4, dtype=torch.float32, device=device)
        target = torch.eye(4, dtype=torch.float32, device=device) 
        target[:3, 3] = torch.tensor([dist, 0.0, 0.0], device=device)
        pc = torch.randn(300, 3, device=device)
        
        progress = torch.tensor(0.0, device=device)  # 초기 상태
        twist = engine._predict_twist(start, target, progress, pc)
        
        v_norm = torch.norm(twist[3:]).item()
        
        print(f"📏 목표 거리: {dist:.1f}m")
        print(f"   📐 예측된 선속도 크기: {v_norm:.6f} m/s")
        print(f"   📊 거리 대비 속도 비율: {v_norm/dist:.6f}")
        print()

def diagnose_problem():
    """문제점 종합 진단"""
    print("\n" + "="*60)
    print("🎯 문제 진단 요약")
    print("="*60)
    
    # 실제 추론으로 확인
    engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
    
    device = engine.device
    start = torch.eye(4, dtype=torch.float32, device=device)
    target = torch.eye(4, dtype=torch.float32, device=device)
    target[:3, 3] = torch.tensor([2.0, 2.0, 0.0], device=device)
    pc = np.random.randn(300, 3)
    
    result = engine.generate_trajectory(start, target, pc, InferenceConfigs.fast())
    
    # 궤적 분석
    trajectory = result['trajectory']
    
    # 스텝 거리들 계산
    step_distances = []
    for i in range(1, len(trajectory)):
        dist = torch.norm(trajectory[i][:3, 3] - trajectory[i-1][:3, 3]).item()
        step_distances.append(dist)
    
    avg_step = np.mean(step_distances)
    max_step = np.max(step_distances)
    min_step = np.min(step_distances)
    
    total_dist = sum(step_distances)
    direct_dist = torch.norm(trajectory[-1][:3, 3] - trajectory[0][:3, 3]).item()
    
    print(f"📊 궤적 통계:")
    print(f"   총 스텝 수: {len(trajectory)-1}")
    print(f"   평균 스텝 거리: {avg_step:.6f}m ({avg_step*1000:.3f}mm)")
    print(f"   최대 스텝 거리: {max_step:.6f}m ({max_step*1000:.3f}mm)")
    print(f"   최소 스텝 거리: {min_step:.6f}m ({min_step*1000:.3f}mm)")
    print(f"   총 이동 거리: {total_dist:.6f}m")
    print(f"   직선 거리: {direct_dist:.6f}m")
    print(f"   효율성: {direct_dist/total_dist*100:.2f}%")
    print()
    
    # 기본 설정 확인
    print(f"🔧 현재 설정:")
    print(f"   dt (적분 스텝): {engine.default_config['dt']}")
    print(f"   max_steps: {engine.default_config['max_steps']}")
    print()
    
    # 문제 진단
    print("❌ 발견된 문제점들:")
    
    if avg_step < 0.01:
        print(f"   1. 스텝 크기가 너무 작음 ({avg_step*1000:.1f}mm)")
        print(f"      → 모델이 너무 작은 velocity를 예측하거나")
        print(f"      → dt가 너무 작을 가능성")
    
    if max_step == min_step:
        print(f"   2. 모든 스텝이 동일한 크기")
        print(f"      → 모델이 일정한 velocity만 출력")
        print(f"      → 학습 부족이나 모델 문제")
    
    if result['generation_time'] > 1.0:
        print(f"   3. 생성 시간이 오래 걸림 ({result['generation_time']:.1f}s)")
        print(f"      → 너무 많은 스텝이 필요")
        
    return result

if __name__ == "__main__":
    print("🚀 Velocity Field 분석 시작")
    print()
    
    # 1. Twist vector 분석
    engine = analyze_twist_vector()
    
    # 2. 적분 스텝 분석
    analyze_integration_step()
    
    # 3. 거리별 분석
    test_different_distances()
    
    # 4. 종합 진단
    result = diagnose_problem()
    
    print("\n✅ 분석 완료!")
    print(f"📁 결과 저장됨: velocity_field_analysis_result.json")
    
    # 결과 저장
    import json
    analysis_result = {
        "avg_step_size_mm": np.mean([torch.norm(result['trajectory'][i][:3, 3] - result['trajectory'][i-1][:3, 3]).item() for i in range(1, len(result['trajectory']))]) * 1000,
        "total_steps": len(result['trajectory']) - 1,
        "generation_time": result['generation_time'],
        "success": result['success'],
        "final_error": result['final_error']
    }
    
    with open('velocity_field_analysis_result.json', 'w') as f:
        json.dump(analysis_result, f, indent=2)
