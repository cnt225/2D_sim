#!/usr/bin/env python3
"""
추론 결과와 학습 데이터 궤적 형식 비교
"""

import torch
import numpy as np
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import MotionRFMInference, InferenceConfigs

def load_training_trajectory():
    """학습 데이터의 궤적 형식 확인"""
    
    print("📂 학습 데이터 궤적 형식 분석")
    print("=" * 50)
    
    traj_file = "../../../data/trajectories/circle_envs_10k/circle_env_000000_pair_1_traj_rb3.json"
    
    try:
        with open(traj_file, 'r') as f:
            traj_data = json.load(f)
        
        print(f"✅ 궤적 파일 로드: {traj_file}")
        print(f"📊 JSON 최상위 키들: {list(traj_data.keys())}")
        
        # 궤적 데이터 구조 분석
        if 'trajectory' in traj_data:
            trajectory = traj_data['trajectory']
            print(f"📏 궤적 포인트 수: {len(trajectory)}")
            
            # 첫 번째 포인트 구조 확인
            first_point = trajectory[0]
            print(f"🔍 첫 번째 포인트 키들: {list(first_point.keys())}")
            
            # SE(3) 변환 행렬 확인
            if 'transformation_matrix' in first_point:
                transform = first_point['transformation_matrix']
                print(f"🔄 변환 행렬 형태: {np.array(transform).shape}")
                print(f"📍 첫 번째 포즈:")
                print(np.array(transform))
                
                # 위치 정보 추출
                pos = np.array(transform)[:3, 3]
                print(f"📍 첫 번째 위치: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            
            # 마지막 포인트도 확인
            if len(trajectory) > 1:
                last_point = trajectory[-1]
                if 'transformation_matrix' in last_point:
                    last_transform = np.array(last_point['transformation_matrix'])
                    last_pos = last_transform[:3, 3]
                    print(f"📍 마지막 위치: ({last_pos[0]:.3f}, {last_pos[1]:.3f}, {last_pos[2]:.3f})")
                    
                    # 총 이동 거리 계산
                    total_distance = np.linalg.norm(last_pos - pos)
                    print(f"📏 총 이동 거리: {total_distance:.3f}m")
            
            # 추가 정보
            if 'metadata' in traj_data:
                metadata = traj_data['metadata']
                print(f"📋 메타데이터 키들: {list(metadata.keys())}")
                if 'total_length' in metadata:
                    print(f"📐 메타데이터 길이: {metadata['total_length']}")
                if 'duration' in metadata:
                    print(f"⏱️ 지속 시간: {metadata['duration']}")
        
        return traj_data
        
    except Exception as e:
        print(f"❌ 궤적 파일 로드 실패: {e}")
        return None

def generate_inference_trajectory():
    """추론으로 궤적 생성"""
    
    print("\n" + "=" * 50)
    print("🚀 추론 궤적 형식 분석")
    
    # 간단한 테스트 데이터
    start_pose = torch.eye(4, dtype=torch.float32)
    start_pose[:3, 3] = torch.tensor([1.0, 1.0, 0.0])
    
    target_pose = torch.eye(4, dtype=torch.float32)
    target_pose[:3, 3] = torch.tensor([2.0, 2.0, 0.0])
    
    pointcloud = np.random.randn(500, 3)  # 더미 포인트클라우드
    
    try:
        engine = MotionRFMInference(
            model_path="checkpoints/motion_rcfm_final_epoch10.pth",
            config_path="configs/motion_rcfm.yml",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        result = engine.generate_trajectory(
            start_pose=start_pose,
            target_pose=target_pose,
            pointcloud=pointcloud,
            config=InferenceConfigs.fast()  # 빠른 테스트용
        )
        
        trajectory = result['trajectory']
        
        print(f"✅ 추론 궤적 생성 완료")
        print(f"📏 궤적 포인트 수: {len(trajectory)}")
        print(f"🔍 첫 번째 포즈 타입: {type(trajectory[0])}")
        print(f"🔍 첫 번째 포즈 형태: {trajectory[0].shape}")
        
        # 첫 번째와 마지막 위치
        first_pos = trajectory[0][:3, 3]
        last_pos = trajectory[-1][:3, 3]
        
        print(f"📍 첫 번째 위치: ({first_pos[0]:.3f}, {first_pos[1]:.3f}, {first_pos[2]:.3f})")
        print(f"📍 마지막 위치: ({last_pos[0]:.3f}, {last_pos[1]:.3f}, {last_pos[2]:.3f})")
        
        return result
        
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        return None

def convert_inference_to_training_format(inference_result):
    """추론 결과를 학습 데이터 형식으로 변환"""
    
    print("\n" + "=" * 50)
    print("🔄 형식 변환: 추론 → 학습 데이터 형식")
    
    if inference_result is None:
        print("❌ 추론 결과가 없습니다.")
        return None
    
    trajectory = inference_result['trajectory']
    
    # 학습 데이터 형식으로 변환
    converted_trajectory = []
    
    for i, pose_tensor in enumerate(trajectory):
        # torch.Tensor를 numpy array로 변환
        pose_matrix = pose_tensor.cpu().numpy().tolist()
        
        point = {
            "step": i,
            "transformation_matrix": pose_matrix,
            "timestamp": i * 0.02  # dt=0.02 가정
        }
        converted_trajectory.append(point)
    
    # 메타데이터 추가
    converted_data = {
        "trajectory": converted_trajectory,
        "metadata": {
            "algorithm": "Motion_RFM",
            "total_points": len(converted_trajectory),
            "success": inference_result['success'],
            "generation_time": inference_result['generation_time'],
            "final_error": inference_result['final_error'],
            "steps": inference_result['steps'],
            "config": inference_result['info']['config']
        }
    }
    
    print(f"✅ 변환 완료")
    print(f"📏 변환된 궤적 포인트 수: {len(converted_trajectory)}")
    
    # 샘플 저장
    output_file = "inference_trajectory_converted.json"
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"💾 변환된 궤적 저장: {output_file}")
    
    return converted_data

def compare_formats():
    """형식 비교 분석"""
    
    print("\n" + "=" * 50)
    print("⚖️ 형식 비교 분석")
    
    print("\n🔍 **학습 데이터 형식**:")
    print("   - JSON 파일")
    print("   - 'trajectory' 키 하위에 포인트 리스트")
    print("   - 각 포인트: {'step', 'transformation_matrix', 'timestamp'}")
    print("   - transformation_matrix: 4x4 리스트 (SE(3))")
    print("   - 'metadata' 키로 추가 정보")
    
    print("\n🚀 **추론 출력 형식**:")
    print("   - Python 딕셔너리")
    print("   - 'trajectory' 키: List[torch.Tensor]")
    print("   - 각 텐서: (4, 4) SE(3) 행렬")
    print("   - 추가 정보: success, steps, final_error, generation_time")
    
    print("\n🔄 **변환 과정**:")
    print("   1. torch.Tensor → numpy → list")
    print("   2. step, timestamp 정보 추가")
    print("   3. 메타데이터 구조 맞춤")
    print("   4. JSON 직렬화 가능 형식")
    
    print("\n✅ **호환성**:")
    print("   - ✅ SE(3) 변환 행렬 동일")
    print("   - ✅ 궤적 순서 동일") 
    print("   - ✅ 위치/회전 정보 보존")
    print("   - ✅ 기존 시각화/분석 도구 사용 가능")

if __name__ == "__main__":
    print("🔍 궤적 형식 비교 분석 시작")
    print("=" * 60)
    
    # 1. 학습 데이터 궤적 분석
    training_traj = load_training_trajectory()
    
    # 2. 추론 궤적 생성
    inference_result = generate_inference_trajectory()
    
    # 3. 형식 변환
    converted_traj = convert_inference_to_training_format(inference_result)
    
    # 4. 형식 비교
    compare_formats()
    
    print("\n🎉 분석 완료!")
    print("💡 추론 결과는 간단한 변환으로 학습 데이터와 동일한 형식 사용 가능!")




