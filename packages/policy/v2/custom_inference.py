#!/usr/bin/env python3
"""
커스텀 조건으로 추론 실행
"""
import torch
import numpy as np
from pathlib import Path
import glob
from inference_normalized import NormalizedMotionRCFMInference

def custom_inference():
    """커스텀 조건으로 추론 실행"""
    
    print("🧪 커스텀 조건으로 Motion RCFM 추론...")
    
    # 최신 체크포인트 찾기
    checkpoint_patterns = [
        "train_results/motion_rcfm/*/best_model.pth",
        "train_results/motion_rcfm/*/model_latest.pth", 
        "train_results/motion_rcfm/*/*.pth"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern))
        if checkpoints:
            break
    
    if not checkpoints:
        print("❌ 훈련된 모델을 찾을 수 없습니다.")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda x: Path(x).stat().st_mtime)
    print(f"📦 사용할 체크포인트: {latest_checkpoint}")
    
    # 추론 엔진 초기화
    inference_engine = NormalizedMotionRCFMInference(
        model_path=latest_checkpoint,
        config_path="configs/motion_rcfm.yml",
        normalize_twist=True
    )
    
    print("\n🎯 커스텀 조건으로 추론 실행...")
    
    # 더미 포인트클라우드 (실제로는 circle_env_000000 환경 사용)
    pointcloud = torch.randn(2000, 3)
    
    # 새로운 시작/목표 포즈: [0,0,0] → [3,3,0], 같은 orientation (0도)
    start_pose = np.eye(4)
    start_pose[:3, 3] = [0.0, 0.0, 0.0]  # [x, y, z]
    
    target_pose = np.eye(4) 
    target_pose[:3, 3] = [3.0, 3.0, 0.0]  # [x, y, z]
    
    print(f"   시작점: [0, 0, 0] (orientation: 0°)")
    print(f"   목표점: [3, 3, 0] (orientation: 0°)")
    print(f"   거리: {np.sqrt((3-0)**2 + (3-0)**2):.2f}m")
    
    # 적분 구간을 1/100로 줄여서 더 세밀한 궤적 (원래 20개 → 2000개)
    num_samples = 2000  # 훨씬 더 세밀한 궤적
    print(f"   궤적 포인트 수: {num_samples}개 (1/100 스케일)")
    
    # 궤적 생성
    result = inference_engine.generate_trajectory(
        pointcloud=pointcloud,
        start_pose=start_pose,
        target_pose=target_pose,
        num_samples=num_samples
    )
    
    # 결과 출력
    inference_engine.visualize_trajectory(result)
    
    if result['success']:
        print("✅ 커스텀 추론 성공!")
        
        # 궤적을 JSON 파일로 저장
        print("\n💾 커스텀 궤적 JSON 저장...")
        try:
            saved_path = inference_engine.save_trajectory_json(
                trajectory_poses=result['poses'],
                start_pose=result['start_pose'],
                goal_pose=result['target_pose'],
                environment_name="circle_env_000000",  # 같은 환경 사용
                rigid_body_id=3,
                rigid_body_type="elongated_ellipse",
                output_path="inference_results/custom_traj_0_0_to_3_3_fine.json"
            )
            print(f"✅ 커스텀 궤적 저장 성공: {saved_path}")
            return saved_path
            
        except Exception as e:
            print(f"❌ 궤적 저장 실패: {e}")
            return None
    else:
        print("❌ 커스텀 추론 실패!")
        return None

if __name__ == "__main__":
    saved_path = custom_inference()
    if saved_path:
        print(f"\n🎨 시각화를 위해 다음 명령어 실행:")
        print(f"python simple_visualize.py {saved_path}")

