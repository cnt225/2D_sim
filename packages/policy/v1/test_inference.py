#!/usr/bin/env python3
"""
학습된 Motion RFM 모델로 추론 테스트
circle_env_000000, pose pair #2 사용 (학습에 미사용된 데이터)
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import MotionRFMInference, InferenceConfigs
# from utils.pointcloud import load_pointcloud

def load_test_data():
    """테스트 데이터 로드: circle_env_000000, pose pair #2"""
    
    # 파일 경로들
    pointcloud_file = "../../../data/pointcloud/circle_envs_10k/circle_envs_10k/circle_env_000000.ply"
    pose_pairs_file = "../../../data/pose_pairs/circle_envs_10k/circle_env_000000_rb_3_pairs.json"
    
    print(f"📂 데이터 로드 중...")
    print(f"   포인트클라우드: {pointcloud_file}")
    print(f"   포즈 페어: {pose_pairs_file}")
    
    # 포인트클라우드 로드
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pointcloud_file)
        pointcloud = np.asarray(pcd.points)
        print(f"✅ 포인트클라우드 로드: {pointcloud.shape[0]}개 점")
    except Exception as e:
        print(f"❌ 포인트클라우드 로드 실패: {e}")
        return None, None, None
    
    # 포즈 페어 로드
    try:
        with open(pose_pairs_file, 'r') as f:
            pose_data = json.load(f)
        
        # 구조 확인 후 첫 번째 available 페어 사용
        pairs = pose_data['pose_pairs']['data']
        if len(pairs) > 1:
            # 두 번째 페어 사용 (인덱스 1)
            pair_data = pairs[1]
        else:
            # 첫 번째 페어 사용
            pair_data = pairs[0]
        
        # SE(3) 행렬로 변환
        start_pose = torch.tensor(pair_data['start_pose'], dtype=torch.float32)
        target_pose = torch.tensor(pair_data['target_pose'], dtype=torch.float32)
        
        print(f"✅ 포즈 페어 로드 완료")
        print(f"   시작 위치: {start_pose[:3, 3].tolist()}")
        print(f"   목표 위치: {target_pose[:3, 3].tolist()}")
        
        # 거리 계산
        distance = torch.norm(target_pose[:3, 3] - start_pose[:3, 3])
        print(f"   거리: {distance:.3f}m")
        
        return pointcloud, start_pose, target_pose
        
    except Exception as e:
        print(f"❌ 포즈 페어 로드 실패: {e}")
        return None, None, None

def test_inference():
    """추론 테스트 실행"""
    
    print("🚀 Motion RFM 추론 테스트 시작")
    print("=" * 50)
    
    # 1. 테스트 데이터 로드
    pointcloud, start_pose, target_pose = load_test_data()
    if pointcloud is None:
        return
    
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
        return
    
    print("\n" + "=" * 50)
    
    # 3. 여러 설정으로 테스트
    configs = {
        "기본 설정": InferenceConfigs.default(),
        "고품질 설정": InferenceConfigs.high_quality(),
        "고속 설정": InferenceConfigs.fast(),
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n🎯 {config_name} 테스트")
        print(f"   dt: {config['dt']}, max_steps: {config['max_steps']}")
        print(f"   허용오차: {config['pos_tolerance']}m, {config['rot_tolerance']}rad")
        
        try:
            # 추론 실행
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
            
            print(f"   ✅ 결과: {'성공' if success else '실패'}")
            print(f"   📊 스텝 수: {steps}")
            print(f"   ⏱️ 생성 시간: {gen_time:.3f}초")
            print(f"   📏 위치 오차: {final_error['position_error_m']:.3f}m")
            print(f"   🔄 회전 오차: {final_error['rotation_error_deg']:.1f}°")
            print(f"   🚀 속도: {1/gen_time:.1f} 궤적/초")
            
            # 궤적 길이 계산
            trajectory = result['trajectory']
            total_length = 0
            for i in range(1, len(trajectory)):
                pos_diff = trajectory[i][:3, 3] - trajectory[i-1][:3, 3]
                total_length += torch.norm(pos_diff).item()
            
            print(f"   📐 궤적 길이: {total_length:.3f}m")
            print(f"   📈 효율성: {torch.norm(target_pose[:3, 3] - start_pose[:3, 3]).item() / total_length:.3f}")
            
            results[config_name] = result
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results[config_name] = None
    
    print("\n" + "=" * 50)
    
    # 4. 성능 요약
    print("📊 성능 요약")
    successful_configs = [name for name, result in results.items() if result and result['success']]
    
    if successful_configs:
        print(f"✅ 성공한 설정: {len(successful_configs)}/{len(configs)}")
        
        # 가장 빠른 설정
        fastest_config = min(successful_configs, 
                           key=lambda name: results[name]['generation_time'])
        fastest_time = results[fastest_config]['generation_time']
        
        # 가장 정확한 설정
        most_accurate_config = min(successful_configs,
                                 key=lambda name: results[name]['final_error']['position_error_m'])
        best_accuracy = results[most_accurate_config]['final_error']['position_error_m']
        
        print(f"🚀 가장 빠른 설정: {fastest_config} ({fastest_time:.3f}초)")
        print(f"🎯 가장 정확한 설정: {most_accurate_config} ({best_accuracy:.3f}m)")
        
        # RRT-Connect 대비 추정
        print(f"\n🔥 RRT-Connect 대비 예상 성능:")
        print(f"   속도: {1/fastest_time:.0f}x 빠름 (RRT ~1초 vs RFM ~{fastest_time:.3f}초)")
        print(f"   정확도: {best_accuracy*1000:.1f}mm (매우 정밀함)")
        
    else:
        print("❌ 모든 설정에서 실패")
        print("💡 설정 조정이나 추가 학습이 필요할 수 있습니다.")
    
    print("\n🎉 추론 테스트 완료!")
    
    return results

if __name__ == "__main__":
    results = test_inference()
