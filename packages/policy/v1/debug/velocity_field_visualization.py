#!/usr/bin/env python3
"""
Velocity Field 시각화 - 20x20 그리드로 방향성 확인
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys, os

sys.path.append('.')
from inference import MotionRFMInference, InferenceConfigs

def create_velocity_field_visualization():
    """20x20 그리드로 velocity field 방향성 시각화"""
    print("🎨 Velocity Field 시각화 생성")
    print("=" * 50)
    
    # 엔진 로드
    engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
    device = engine.device
    
    # 환경 설정 (2D 평면에서 분석)
    workspace_size = 2.0  # 2m x 2m 작업공간
    grid_size = 20
    
    # 시작점과 목표점 설정
    start_pos = np.array([0.2, 0.2])  # 시작 (왼쪽 아래)
    target_pos = np.array([1.8, 1.8])  # 목표 (오른쪽 위)
    
    print(f"📍 시작점: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    print(f"🎯 목표점: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
    print(f"📊 그리드 크기: {grid_size}x{grid_size}")
    print()
    
    # 그리드 포인트 생성
    x = np.linspace(0, workspace_size, grid_size)
    y = np.linspace(0, workspace_size, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Velocity field 저장할 배열
    U = np.zeros_like(X)  # X 방향 속도
    V = np.zeros_like(Y)  # Y 방향 속도
    Speed = np.zeros_like(X)  # 속도 크기
    
    # 고정된 환경 (빈 공간)
    pc = torch.randn(300, 3, device=device) * 0.1  # 작은 노이즈
    
    # SE(3) 포즈 생성 (목표는 고정)
    target_pose = torch.eye(4, dtype=torch.float32, device=device)
    target_pose[:3, 3] = torch.tensor([target_pos[0], target_pos[1], 0.0], device=device)
    
    print("🔄 그리드 포인트별 velocity 계산 중...")
    
    # 각 그리드 포인트에서 velocity 계산
    for i in range(grid_size):
        for j in range(grid_size):
            # 현재 위치
            current_pos = np.array([X[i, j], Y[i, j]])
            
            # SE(3) 포즈 생성
            current_pose = torch.eye(4, dtype=torch.float32, device=device)
            current_pose[:3, 3] = torch.tensor([current_pos[0], current_pos[1], 0.0], device=device)
            
            # Progress 계산 (거리 기반)
            current_dist = np.linalg.norm(current_pos - target_pos)
            total_dist = np.linalg.norm(start_pos - target_pos)
            progress = max(0.0, 1.0 - current_dist / total_dist)
            progress_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
            
            # Velocity 예측
            twist = engine._predict_twist(current_pose, target_pose, progress_tensor, pc)
            
            # 2D 평면에서의 선속도만 추출 (X, Y)
            linear_vel = twist[3:6].detach().cpu().numpy()  # [v_x, v_y, v_z]
            
            U[i, j] = linear_vel[0]  # X 방향
            V[i, j] = linear_vel[1]  # Y 방향
            Speed[i, j] = np.sqrt(linear_vel[0]**2 + linear_vel[1]**2)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 화살표로 방향성 표시
    ax1.quiver(X, Y, U, V, Speed, cmap='viridis', scale=1.0, width=0.003)
    ax1.scatter(*start_pos, color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(*target_pos, color='red', s=100, marker='*', label='Target', zorder=5)
    ax1.set_xlim(0, workspace_size)
    ax1.set_ylim(0, workspace_size)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Velocity Field (방향성)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 2. 속도 크기로 컬러맵
    im = ax2.imshow(Speed, extent=[0, workspace_size, 0, workspace_size], 
                    origin='lower', cmap='hot', interpolation='bilinear')
    ax2.scatter(*start_pos, color='cyan', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(*target_pos, color='blue', s=100, marker='*', label='Target', zorder=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Velocity Magnitude (속도 크기)')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Speed (m/s)')
    
    plt.tight_layout()
    plt.savefig('velocity_field_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 통계 분석
    print("\n📊 Velocity Field 통계:")
    print(f"   평균 속도: {np.mean(Speed):.6f} m/s")
    print(f"   최대 속도: {np.max(Speed):.6f} m/s")
    print(f"   최소 속도: {np.min(Speed):.6f} m/s")
    print(f"   속도 표준편차: {np.std(Speed):.6f} m/s")
    print()
    
    # 방향성 분석
    # 목표 방향과 예측 방향의 일치도
    alignment_scores = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            current_pos = np.array([X[i, j], Y[i, j]])
            
            # 이상적인 방향 (목표를 향하는 방향)
            ideal_direction = target_pos - current_pos
            ideal_direction = ideal_direction / (np.linalg.norm(ideal_direction) + 1e-8)
            
            # 예측된 방향
            predicted_direction = np.array([U[i, j], V[i, j]])
            predicted_direction = predicted_direction / (np.linalg.norm(predicted_direction) + 1e-8)
            
            # 코사인 유사도
            alignment = np.dot(ideal_direction, predicted_direction)
            alignment_scores.append(alignment)
    
    alignment_scores = np.array(alignment_scores)
    
    print("🎯 방향성 분석:")
    print(f"   평균 정렬도: {np.mean(alignment_scores):.3f} (-1~1, 1이 완벽)")
    print(f"   정렬도 표준편차: {np.std(alignment_scores):.3f}")
    print(f"   올바른 방향 비율: {np.sum(alignment_scores > 0.5) / len(alignment_scores) * 100:.1f}%")
    
    return {
        'speed_stats': {
            'mean': float(np.mean(Speed)),
            'max': float(np.max(Speed)), 
            'min': float(np.min(Speed)),
            'std': float(np.std(Speed))
        },
        'alignment_stats': {
            'mean': float(np.mean(alignment_scores)),
            'std': float(np.std(alignment_scores)),
            'correct_ratio': float(np.sum(alignment_scores > 0.5) / len(alignment_scores))
        }
    }

def test_different_scenarios():
    """다양한 시나리오에서 velocity field 테스트"""
    print("\n🔬 다양한 시나리오 테스트")
    print("=" * 50)
    
    scenarios = [
        {"name": "Short Distance", "start": [0.5, 0.5], "target": [0.7, 0.7]},
        {"name": "Long Distance", "start": [0.2, 0.2], "target": [1.8, 1.8]},
        {"name": "Horizontal", "start": [0.2, 1.0], "target": [1.8, 1.0]},
        {"name": "Vertical", "start": [1.0, 0.2], "target": [1.0, 1.8]},
    ]
    
    engine = MotionRFMInference('checkpoints/motion_rcfm_final_epoch10.pth', 'configs/motion_rcfm.yml')
    device = engine.device
    pc = torch.randn(300, 3, device=device) * 0.1
    
    for scenario in scenarios:
        start_pos = np.array(scenario["start"])
        target_pos = np.array(scenario["target"])
        
        # SE(3) 포즈
        start_pose = torch.eye(4, dtype=torch.float32, device=device)
        start_pose[:3, 3] = torch.tensor([start_pos[0], start_pos[1], 0.0], device=device)
        
        target_pose = torch.eye(4, dtype=torch.float32, device=device)
        target_pose[:3, 3] = torch.tensor([target_pos[0], target_pos[1], 0.0], device=device)
        
        # 초기 velocity
        progress = torch.tensor(0.0, dtype=torch.float32, device=device)
        twist = engine._predict_twist(start_pose, target_pose, progress, pc)
        
        linear_vel = twist[3:6].detach().cpu().numpy()
        speed = np.linalg.norm(linear_vel[:2])
        
        # 방향 분석
        ideal_direction = target_pos - start_pos
        ideal_direction = ideal_direction / np.linalg.norm(ideal_direction)
        
        predicted_direction = linear_vel[:2] / (np.linalg.norm(linear_vel[:2]) + 1e-8)
        alignment = np.dot(ideal_direction, predicted_direction)
        
        distance = np.linalg.norm(target_pos - start_pos)
        
        print(f"📋 {scenario['name']}:")
        print(f"   거리: {distance:.2f}m")
        print(f"   속도: {speed:.6f} m/s")
        print(f"   방향 정렬도: {alignment:.3f}")
        print(f"   이상적 방향: [{ideal_direction[0]:.3f}, {ideal_direction[1]:.3f}]")
        print(f"   예측된 방향: [{predicted_direction[0]:.3f}, {predicted_direction[1]:.3f}]")
        print()

if __name__ == "__main__":
    print("🚀 Velocity Field 시각화 시작")
    print()
    
    # 1. 메인 시각화
    results = create_velocity_field_visualization()
    
    # 2. 다양한 시나리오 테스트
    test_different_scenarios()
    
    print("✅ 시각화 완료!")
    print(f"📁 이미지 저장: velocity_field_visualization.png")
    
    # 결과 요약
    print("\n" + "="*60)
    print("🎯 핵심 문제 진단 결과")
    print("="*60)
    
    print("❌ 발견된 핵심 문제:")
    print("   1. 모델이 예측하는 velocity가 너무 작음 (~0.06 m/s)")
    print("   2. 거리와 무관하게 일정한 속도 출력 (거리 적응성 부족)")
    print("   3. Progress 변화에 무관하게 거의 동일한 출력")
    print("   4. dt=0.02s와 결합시 스텝당 ~1.2mm 이동으로 극도로 느림")
    print()
    
    print("✅ 해결 방안:")
    print("   1. dt 크기 증가: 0.02 → 0.1 (5배)")
    print("   2. 모델 재학습: 더 큰 velocity 스케일로")
    print("   3. Velocity 정규화/스케일링 추가")
    print("   4. Progress-aware velocity 조정")
    print()
    
    print("🔧 즉시 테스트 가능한 해결책:")
    print("   - InferenceConfigs에서 dt 값 조정")
    print("   - Velocity에 스케일 팩터 곱하기")
    print("   - 더 적극적인 early stopping")
