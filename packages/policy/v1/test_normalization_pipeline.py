#!/usr/bin/env python3
"""
정규화 파이프라인 완전 검증
Dataset → Model → Inference 전체 흐름 테스트
"""

import torch
import numpy as np
import sys
import json
from pathlib import Path

# 모듈 imports
from loaders.trajectory_dataset import TrajectoryDataset
from inference_normalized import NormalizedMotionRFMInference, NormalizedInferenceConfigs
from utils.normalization import TwistNormalizer

def test_dataset_normalization():
    """Dataset 정규화 테스트"""
    print("🔍 1단계: Dataset 정규화 테스트")
    print("-" * 50)
    
    # 정규화 없는 데이터셋
    dataset_raw = TrajectoryDataset(
        trajectory_root="../../../data/trajectories/circle_envs_10k_bsplined",
        pointcloud_root="../../../data/pointcloud/circle_envs_10k/circle_envs_10k",
        split='train',
        max_trajectories=10,
        normalize_twist=False,
        augment_data=False,
        num_points=300
    )
    
    # 정규화 있는 데이터셋
    dataset_norm = TrajectoryDataset(
        trajectory_root="../../../data/trajectories/circle_envs_10k_bsplined",
        pointcloud_root="../../../data/pointcloud/circle_envs_10k/circle_envs_10k",
        split='train',
        max_trajectories=10,
        normalize_twist=True,
        augment_data=False,
        num_points=300
    )
    
    # 샘플 비교
    sample_raw = dataset_raw[0]
    sample_norm = dataset_norm[0]
    
    T_dot_raw = sample_raw['T_dot'].numpy()
    T_dot_norm = sample_norm['T_dot'].numpy()
    
    print(f"원본 T_dot 크기: {np.linalg.norm(T_dot_raw):.6f}")
    print(f"정규화 T_dot 크기: {np.linalg.norm(T_dot_norm):.6f}")
    print(f"원본 T_dot: {T_dot_raw}")
    print(f"정규화 T_dot: {T_dot_norm}")
    
    # 역정규화 테스트
    normalizer = TwistNormalizer(stats_path="configs/normalization_stats.json")
    T_dot_denorm = normalizer.denormalize_twist(T_dot_norm)
    
    print(f"역정규화 T_dot: {T_dot_denorm}")
    print(f"복원 오차: {np.linalg.norm(T_dot_raw - T_dot_denorm):.8f}")
    
    if np.linalg.norm(T_dot_raw - T_dot_denorm) < 1e-5:
        print("✅ Dataset 정규화 정확!")
    else:
        print("❌ Dataset 정규화 오차 발생!")
    
    return T_dot_raw, T_dot_norm

def test_inference_denormalization():
    """Inference 역정규화 테스트"""
    print("\n🔍 2단계: Inference 역정규화 테스트")
    print("-" * 50)
    
    # 정규화 버전과 비정규화 버전 비교
    try:
        # 정규화 추론 엔진
        engine_norm = NormalizedMotionRFMInference(
            'checkpoints/motion_rcfm_final_epoch10.pth',
            'configs/motion_rcfm.yml',
            normalize_twist=True
        )
        
        # 비정규화 추론 엔진
        engine_raw = NormalizedMotionRFMInference(
            'checkpoints/motion_rcfm_final_epoch10.pth',
            'configs/motion_rcfm.yml',
            normalize_twist=False
        )
        
        # 테스트 데이터
        start_pose = torch.eye(4, dtype=torch.float32)
        target_pose = torch.eye(4, dtype=torch.float32)
        target_pose[:3, 3] = torch.tensor([0.5, 0.5, 0.0])  # 작은 거리
        pointcloud = torch.randn(300, 3)
        
        # 단일 twist 예측 비교
        with torch.no_grad():
            progress = torch.tensor(0.5)
            
            # 정규화 모드 twist
            twist_norm = engine_norm._predict_twist(start_pose, target_pose, progress, pointcloud)
            
            # 비정규화 모드 twist
            twist_raw = engine_raw._predict_twist(start_pose, target_pose, progress, pointcloud)
            
            print(f"정규화 모드 twist 크기: {torch.norm(twist_norm).item():.6f}")
            print(f"비정규화 모드 twist 크기: {torch.norm(twist_raw).item():.6f}")
            print(f"정규화 모드 twist: {twist_norm.numpy()}")
            print(f"비정규화 모드 twist: {twist_raw.numpy()}")
            
            # 정규화 모드에서 더 큰 값이 나와야 함
            if torch.norm(twist_norm) > torch.norm(twist_raw):
                print("✅ Inference 역정규화 작동 중!")
            else:
                print("⚠️ 역정규화 효과 확인 필요")
        
        return True
    
    except Exception as e:
        print(f"❌ Inference 테스트 실패: {e}")
        return False

def test_end_to_end_trajectory():
    """End-to-End 궤적 생성 테스트"""
    print("\n🔍 3단계: End-to-End 궤적 생성 테스트")
    print("-" * 50)
    
    try:
        # 정규화 추론 엔진
        engine = NormalizedMotionRFMInference(
            'checkpoints/motion_rcfm_final_epoch10.pth',
            'configs/motion_rcfm.yml',
            normalize_twist=True
        )
        
        # 테스트 케이스들
        test_cases = [
            {"name": "근거리", "target": [0.2, 0.2, 0.0]},
            {"name": "중거리", "target": [1.0, 1.0, 0.0]},
            {"name": "원거리", "target": [2.0, 2.0, 0.0]}
        ]
        
        results = []
        
        for case in test_cases:
            print(f"\n🎯 {case['name']} 테스트: {case['target']}")
            
            start_pose = torch.eye(4, dtype=torch.float32)
            target_pose = torch.eye(4, dtype=torch.float32)
            target_pose[:3, 3] = torch.tensor(case['target'])
            pointcloud = torch.randn(300, 3)
            
            # 궤적 생성
            result = engine.generate_trajectory(
                start_pose, target_pose, pointcloud,
                NormalizedInferenceConfigs.default()
            )
            
            # 결과 분석
            total_distance = torch.norm(target_pose[:3, 3] - start_pose[:3, 3]).item()
            final_distance = result['final_error']['position_error_m']
            success_rate = result['success']
            
            print(f"   목표 거리: {total_distance:.3f}m")
            print(f"   최종 오차: {final_distance:.3f}m")
            print(f"   성공 여부: {'✅' if success_rate else '❌'}")
            print(f"   스텝 수: {result['steps']}")
            print(f"   시간: {result['generation_time']:.3f}초")
            
            results.append({
                'case': case['name'],
                'target_distance': total_distance,
                'final_error': final_distance,
                'success': success_rate,
                'steps': result['steps'],
                'time': result['generation_time']
            })
        
        # 전체 결과 요약
        print(f"\n📊 전체 테스트 요약:")
        success_count = sum(1 for r in results if r['success'])
        avg_error = np.mean([r['final_error'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        print(f"   성공률: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
        print(f"   평균 오차: {avg_error:.3f}m")
        print(f"   평균 스텝: {avg_steps:.1f}")
        
        return results
    
    except Exception as e:
        print(f"❌ End-to-End 테스트 실패: {e}")
        return []

def test_wandb_logging():
    """wandb 로깅 테스트"""
    print("\n🔍 4단계: wandb 설정 확인")
    print("-" * 50)
    
    # train.py 확인
    train_path = Path("../../train.py")
    if train_path.exists():
        with open(train_path, 'r') as f:
            content = f.read()
            
        if 'wandb.log' in content and 'loss' in content:
            print("✅ train.py에 wandb.log(loss) 발견")
        else:
            print("⚠️ train.py에 loss 로깅 추가 필요")
    else:
        print("❌ train.py 파일 없음")

def main():
    """전체 테스트 실행"""
    print("🚀 정규화 파이프라인 완전 검증 시작")
    print("=" * 60)
    
    # 1. Dataset 정규화 테스트
    T_dot_raw, T_dot_norm = test_dataset_normalization()
    
    # 2. Inference 역정규화 테스트
    inference_ok = test_inference_denormalization()
    
    # 3. End-to-End 궤적 생성 테스트
    trajectory_results = test_end_to_end_trajectory()
    
    # 4. wandb 설정 확인
    test_wandb_logging()
    
    # 최종 결론
    print("\n" + "=" * 60)
    print("🎯 최종 검증 결과")
    print("=" * 60)
    
    if len(trajectory_results) > 0:
        success_count = sum(1 for r in trajectory_results if r['success'])
        if success_count > 0:
            print("✅ 정규화 파이프라인 검증 성공!")
            print("✅ 모델이 의미 있는 속도를 출력하고 있음")
            print("✅ 궤적 생성이 정상 작동함")
        else:
            print("❌ 궤적 생성에서 모든 케이스 실패")
            print("🔧 추가 디버깅 필요")
    else:
        print("❌ 검증 프로세스에서 오류 발생")
    
    print("\n🚀 다음 단계:")
    print("1. 정규화 적용된 학습 실행")
    print("2. save_interval=10으로 설정")
    print("3. wandb에서 loss 모니터링")
    print("4. tmux 백그라운드 실행")

if __name__ == "__main__":
    main()
