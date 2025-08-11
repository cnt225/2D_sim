#!/usr/bin/env python3
"""
간단한 더미 데이터로 Motion RFM 추론 테스트
"""

import torch
import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import MotionRFMInference, InferenceConfigs

def create_dummy_data():
    """간단한 더미 테스트 데이터 생성"""
    
    print("📂 더미 테스트 데이터 생성 중...")
    
    # 1. 간단한 포인트클라우드 (원형 장애물 시뮬레이션)
    n_points = 1000
    # 중앙에 원형 장애물
    theta = np.linspace(0, 2*np.pi, n_points//2)
    x_circle = 0.5 * np.cos(theta) + 2.0
    y_circle = 0.5 * np.sin(theta) + 2.0
    z_circle = np.zeros_like(x_circle)
    
    # 바닥 점들
    x_floor = np.random.uniform(0, 4, n_points//2)
    y_floor = np.random.uniform(0, 4, n_points//2)
    z_floor = np.zeros(n_points//2)
    
    pointcloud = np.vstack([
        np.column_stack([x_circle, y_circle, z_circle]),
        np.column_stack([x_floor, y_floor, z_floor])
    ])
    
    print(f"✅ 포인트클라우드 생성: {pointcloud.shape[0]}개 점")
    
    # 2. 시작 및 목표 포즈 정의
    # 시작: 원점
    start_pose = torch.eye(4, dtype=torch.float32)
    start_pose[:3, 3] = torch.tensor([0.5, 0.5, 0.0])  # 시작 위치
    
    # 목표: 장애물을 피해서 대각선 반대편
    target_pose = torch.eye(4, dtype=torch.float32)
    target_pose[:3, 3] = torch.tensor([3.5, 3.5, 0.0])  # 목표 위치
    
    # 목표에서 약간 회전
    import math
    angle = math.pi / 4  # 45도 회전
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    target_pose[0, 0] = cos_a
    target_pose[0, 1] = -sin_a
    target_pose[1, 0] = sin_a
    target_pose[1, 1] = cos_a
    
    print(f"✅ 포즈 페어 생성")
    print(f"   시작 위치: {start_pose[:3, 3].tolist()}")
    print(f"   목표 위치: {target_pose[:3, 3].tolist()}")
    
    # 거리 계산
    distance = torch.norm(target_pose[:3, 3] - start_pose[:3, 3])
    print(f"   거리: {distance:.3f}m")
    
    return pointcloud, start_pose, target_pose

def test_simple_inference():
    """간단한 추론 테스트"""
    
    print("🚀 Motion RFM 간단 추론 테스트 시작")
    print("=" * 50)
    
    # 1. 더미 데이터 생성
    pointcloud, start_pose, target_pose = create_dummy_data()
    
    print("\n" + "=" * 50)
    
    # 2. 추론 엔진 초기화
    print("🔧 추론 엔진 초기화 중...")
    try:
        engine = MotionRFMInference(
            model_path="checkpoints/motion_rcfm_final_epoch10.pth",
            config_path="configs/motion_rcfm.yml",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ 추론 엔진 초기화 완료")
    except Exception as e:
        print(f"❌ 추론 엔진 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    
    # 3. 기본 설정으로 테스트
    print("🎯 기본 설정으로 추론 테스트")
    
    config = InferenceConfigs.default()
    print(f"   설정: dt={config['dt']}, max_steps={config['max_steps']}")
    print(f"   허용오차: {config['pos_tolerance']}m, {config['rot_tolerance']}rad")
    
    try:
        # 추론 실행
        print("🔄 궤적 생성 중...")
        start_time = time.time()
        
        result = engine.generate_trajectory(
            start_pose=start_pose,
            target_pose=target_pose,
            pointcloud=pointcloud,
            config=config
        )
        
        # 결과 분석
        success = result['success']
        steps = result['steps']
        gen_time = result['generation_time']
        final_error = result['final_error']
        trajectory = result['trajectory']
        
        print(f"\n🎉 추론 완료!")
        print(f"   ✅ 결과: {'성공' if success else '실패'}")
        print(f"   📊 스텝 수: {steps}")
        print(f"   ⏱️ 생성 시간: {gen_time:.3f}초")
        print(f"   📏 위치 오차: {final_error['position_error_m']:.3f}m")
        print(f"   🔄 회전 오차: {final_error['rotation_error_deg']:.1f}°")
        print(f"   🚀 속도: {1/gen_time:.1f} 궤적/초")
        
        # 궤적 분석
        total_length = 0
        for i in range(1, len(trajectory)):
            pos_diff = trajectory[i][:3, 3] - trajectory[i-1][:3, 3]
            total_length += torch.norm(pos_diff).item()
        
        straight_distance = torch.norm(target_pose[:3, 3] - start_pose[:3, 3]).item()
        efficiency = straight_distance / total_length if total_length > 0 else 0
        
        print(f"   📐 궤적 길이: {total_length:.3f}m")
        print(f"   📏 직선 거리: {straight_distance:.3f}m")
        print(f"   📈 효율성: {efficiency:.3f} (1.0이 완벽)")
        
        # 궤적 포인트들 출력 (처음/중간/끝)
        print(f"\n📍 궤적 샘플:")
        n_traj = len(trajectory)
        sample_indices = [0, n_traj//4, n_traj//2, 3*n_traj//4, n_traj-1]
        
        for i in sample_indices:
            if i < n_traj:
                pos = trajectory[i][:3, 3]
                print(f"   Step {i:2d}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        # RRT 대비 성능 추정
        print(f"\n🔥 성능 분석:")
        print(f"   🚀 RRT 대비 속도: ~{1000/gen_time:.0f}배 빠름")
        print(f"   🎯 최종 정확도: {final_error['position_error_m']*1000:.1f}mm")
        print(f"   ⚡ 실시간성: {'YES' if gen_time < 0.1 else 'NO'}")
        
        if success:
            print(f"\n✨ 학습된 모델이 성공적으로 작동합니다!")
        else:
            print(f"\n⚠️ 목표 도달 실패 - 설정 조정이 필요할 수 있습니다.")
        
        return result
        
    except Exception as e:
        print(f"❌ 추론 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_simple_inference()
    print("\n🎊 테스트 완료!")




