#!/usr/bin/env python3
"""정규화된 모델로 추론 테스트"""

import sys
sys.path.append('.')
from inference_normalized import NormalizedMotionRFMInference
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import torch

def pose_6d_to_4x4(pose_6d):
    """6차원 포즈 [x, y, z, roll, pitch, yaw]를 4x4 변환 행렬로 변환"""
    x, y, z, roll, pitch, yaw = pose_6d
    
    # 회전 행렬 생성
    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    rotation_matrix = rot.as_matrix()
    
    # 4x4 변환 행렬 생성
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = [x, y, z]
    
    return transform

def load_pointcloud(ply_path):
    """PLY 파일에서 포인트클라우드 로딩"""
    try:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        points = np.asarray(mesh.vertices)
        
        if len(points) == 0:
            print(f"⚠️ 빈 포인트클라우드: {ply_path}")
            # 기본 더미 포인트클라우드 생성
            points = np.random.rand(1000, 3) * 10
        
        # 형태 확인 및 조정
        if points.shape[1] != 3:
            points = points[:, :3]  # 처음 3개 좌표만 사용
            
        print(f"✅ 포인트클라우드 로드: {points.shape}")
        return torch.tensor(points, dtype=torch.float32)
        
    except Exception as e:
        print(f"❌ 포인트클라우드 로딩 실패: {e}")
        # 더미 포인트클라우드 생성
        points = np.random.rand(1000, 3) * 10
        return torch.tensor(points, dtype=torch.float32)

def main():
    # 모델 로드
    print('🔄 정규화된 모델 로딩 중...')
    model = NormalizedMotionRFMInference(
        model_path='checkpoints/model_epoch_10_normalized.pth',
        config_path='configs/motion_rcfm_normalized.yml'
    )

    # pose pairs 파일 로드
    print('📋 환경 데이터 로딩 중...')
    with open('/home/dhkang225/2D_sim/data/pose_pairs/circle_envs_10k/circle_env_000000_rb_3_pairs.json', 'r') as f:
        pairs_data = json.load(f)

    # 두 번째 pair 사용 (인덱스 1 = pair #2)
    pose_pairs_list = pairs_data['pose_pairs']['data']
    if len(pose_pairs_list) >= 2:
        pair_2 = pose_pairs_list[1]  # 두 번째 pair
        start_pose_6d = np.array(pair_2['init'])
        target_pose_6d = np.array(pair_2['target'])
        
        # 6차원 포즈를 4x4 변환 행렬로 변환
        start_pose = pose_6d_to_4x4(start_pose_6d)
        target_pose = pose_6d_to_4x4(target_pose_6d)
        
        print(f'🎯 Pose pair #2:')
        print(f'시작 위치: [{start_pose[0,3]:.3f}, {start_pose[1,3]:.3f}, {start_pose[2,3]:.3f}]')
        print(f'목표 위치: [{target_pose[0,3]:.3f}, {target_pose[1,3]:.3f}, {target_pose[2,3]:.3f}]')
        
        # 포인트클라우드 로드
        pointcloud_path = '/home/dhkang225/2D_sim/data/pointcloud/circle_envs_10k/circle_envs_10k/circle_env_000000.ply'
        pointcloud = load_pointcloud(pointcloud_path)
        
        print('🚀 추론 실행 중...')
        result = model.generate_trajectory(start_pose, target_pose, pointcloud)
        
        print(f"\n📊 추론 결과:")
        print(f"✅ 성공 여부: {result['success']}")
        print(f"📏 스텝 수: {result['steps']}")
        print(f"🎯 최종 오차: {result['final_error']}")
        print(f"⏱️ 생성 시간: {result['generation_time']:.3f}초")
        
        # 궤적 정보
        trajectory = result['trajectory']
        print(f"\n🛤️ 궤적 정보:")
        print(f"📍 웨이포인트 수: {len(trajectory)}")
        
        # 첫 번째와 마지막 위치
        first_pos = trajectory[0][:3, 3]
        last_pos = trajectory[-1][:3, 3]
        print(f"🟢 시작 위치: [{first_pos[0]:.3f}, {first_pos[1]:.3f}, {first_pos[2]:.3f}]")
        print(f"🔴 최종 위치: [{last_pos[0]:.3f}, {last_pos[1]:.3f}, {last_pos[2]:.3f}]")
        
        # 총 이동 거리 계산
        total_distance = 0
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1][:3, 3]
            curr_pos = trajectory[i][:3, 3]
            total_distance += np.linalg.norm(curr_pos - prev_pos)
        
        print(f"📏 총 이동 거리: {total_distance:.6f}m")
        
        # 목표까지의 직선 거리
        target_distance = np.linalg.norm(target_pose[:3, 3] - start_pose[:3, 3])
        print(f"📐 목표까지 직선 거리: {target_distance:.6f}m")
        
        if total_distance > 0:
            print(f"📊 효율성 비율: {target_distance/total_distance:.3f}")
            
            # 궤적 시각화를 위한 위치 정보 출력
            print(f"\n🎨 궤적 시각화 정보:")
            positions = []
            for i, T in enumerate(trajectory):
                pos = T[:3, 3]
                positions.append([pos[0], pos[1], pos[2]])
                if i % 10 == 0 or i == len(trajectory) - 1:  # 10스텝마다 출력
                    print(f"  Step {i:3d}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
            
            # 이동 패턴 분석
            moves = []
            for i in range(1, min(6, len(trajectory))):  # 처음 5 단계만
                prev_pos = trajectory[i-1][:3, 3]
                curr_pos = trajectory[i][:3, 3]
                move = curr_pos - prev_pos
                move_dist = np.linalg.norm(move)
                moves.append(move_dist)
                print(f"  이동 {i}: {move_dist:.6f}m")
            
            avg_move = np.mean(moves) if moves else 0
            print(f"  평균 단계별 이동: {avg_move:.6f}m")
            
        else:
            print(f"⚠️ 이동하지 않음 (정지 상태)")
            
    else:
        print("두 번째 pair가 없습니다.")

if __name__ == "__main__":
    main()
