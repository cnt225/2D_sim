#!/usr/bin/env python3
"""
SE(3) 행렬을 6D 포즈 벡터로 변환 - 간단 버전
"""

import torch
import numpy as np
import json
from scipy.spatial.transform import Rotation
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import MotionRFMInference, InferenceConfigs

def se3_to_6d(se3_matrix):
    """SE(3) 4x4 → [x,y,z,roll,pitch,yaw]"""
    if isinstance(se3_matrix, torch.Tensor):
        se3_matrix = se3_matrix.cpu().numpy()
    
    # 위치
    pos = se3_matrix[:3, 3]
    
    # 회전 → 오일러각
    rot_matrix = se3_matrix[:3, :3]
    rotation = Rotation.from_matrix(rot_matrix)
    euler = rotation.as_euler('xyz', degrees=False)
    
    return [float(pos[0]), float(pos[1]), float(pos[2]), 
            float(euler[0]), float(euler[1]), float(euler[2])]

def convert_inference_result(inference_result):
    """추론 결과 → 학습 데이터 형식"""
    
    trajectory = inference_result['trajectory']
    
    # SE(3) → 6D 변환
    path_data = [se3_to_6d(pose) for pose in trajectory]
    
    # final_error 딕셔너리의 모든 값을 float로 변환
    final_error_clean = {}
    for key, value in inference_result['final_error'].items():
        if hasattr(value, 'item'):  # Tensor인 경우
            final_error_clean[key] = float(value.item())
        else:
            final_error_clean[key] = float(value)
    
    # 학습 데이터 형식으로 구성
    result = {
        "pair_id": -1,
        "trajectory_id": f"inference_{len(trajectory)}pts",
        "start_pose": path_data[0],
        "goal_pose": path_data[-1],
        "path": {
            "data": path_data,
            "format": "6D pose [x,y,z,roll,pitch,yaw]",
            "length": len(path_data),
            "planning_time": inference_result['generation_time']
        },
        "planning_method": "Motion_RFM",
        "generation_info": {
            "success": bool(inference_result['success'].item() if hasattr(inference_result['success'], 'item') else inference_result['success']),
            "steps": int(inference_result['steps']),
            "final_error": final_error_clean,
            "generation_time": float(inference_result['generation_time'])
        }
    }
    
    return result

if __name__ == "__main__":
    print("🔄 간단 변환 테스트")
    
    # 더미 데이터로 빠른 테스트
    start = torch.eye(4, dtype=torch.float32)
    start[:3, 3] = torch.tensor([0.0, 0.0, 0.0])
    
    target = torch.eye(4, dtype=torch.float32) 
    target[:3, 3] = torch.tensor([2.0, 2.0, 0.0])
    
    pc = np.random.randn(300, 3)
    
    print("🚀 추론 실행...")
    try:
        engine = MotionRFMInference(
            model_path="checkpoints/motion_rcfm_final_epoch10.pth",
            config_path="configs/motion_rcfm.yml"
        )
        
        result = engine.generate_trajectory(
            start_pose=start,
            target_pose=target, 
            pointcloud=pc,
            config=InferenceConfigs.fast()
        )
        
        print(f"✅ 추론 완료: {result['steps']}스텝, {result['generation_time']:.3f}초")
        
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        exit(1)
    
    # 변환
    print("🔄 형식 변환...")
    converted = convert_inference_result(result)
    
    # 저장
    with open("converted_trajectory.json", 'w') as f:
        json.dump(converted, f, indent=2)
    
    print("✅ 변환 완료!")
    print(f"📏 궤적 길이: {len(converted['path']['data'])}")
    print(f"📍 시작: {converted['start_pose'][:3]}")
    print(f"📍 목표: {converted['goal_pose'][:3]}")
    print(f"💾 저장: converted_trajectory.json")
    
    # 기존 학습 데이터와 비교
    print("\n📚 학습 데이터와 비교:")
    try:
        with open("../../../data/trajectories/circle_envs_10k/circle_env_000000_pair_1_traj_rb3.json") as f:
            train_data = json.load(f)
        
        print(f"학습 데이터 키들: {list(train_data.keys())}")
        print(f"변환 데이터 키들: {list(converted.keys())}")
        print(f"✅ 형식 호환: {'path' in train_data and 'path' in converted}")
        
    except:
        print("학습 데이터 비교 실패")
    
    print("\n🎯 결론: 추론 결과를 기존 궤적 형식으로 변환 완료!")
