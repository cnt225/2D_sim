#!/usr/bin/env python3
"""
SE(3) Batch Pose Generator
특정 환경-rigid body 쌍에 대해 collision-free SE(3) pose를 대량 생성하여 저장

사용법:
    python batch_pose_generator.py <environment> <rigid_body_id> --num_poses <n>
    
예시:
    # 기본 환경에 대해
    python batch_pose_generator.py circles_only 0 --num_poses 50
    
    # circle_envs 환경에 대해  
    python batch_pose_generator.py circle_envs_10k/env_0001 2 --num_poses 100
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from .random_pose_generator import SE3RandomPoseGenerator
except ImportError:
    from random_pose_generator import SE3RandomPoseGenerator


class SE3BatchPoseGenerator:
    """SE(3) 배치 포즈 생성기"""
    
    def __init__(self, config_file: str = "config/rigid_body_configs.yaml", seed: int = None):
        """
        Args:
            config_file: rigid body 설정 파일 경로
            seed: 랜덤 시드
        """
        self.pose_generator = SE3RandomPoseGenerator(config_file, seed)
        self.data_dir = Path("../../data/pose")  # 프로젝트 루트의 data/pose
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_and_save_poses(self, 
                               environment: str, 
                               rigid_body_id: int, 
                               num_poses: int = 100,
                               max_attempts: int = 1000,
                               safety_margin: float = 0.05) -> str:
        """
        특정 환경-rigid body 쌍에 대해 SE(3) pose 생성 및 저장
        
        Args:
            environment: 환경 이름 (예: 'circles_only', 'circle_envs_10k/env_0001')
            rigid_body_id: Rigid body ID (0-2)
            num_poses: 생성할 pose 개수
            max_attempts: 최대 시도 횟수 (포즈당)
            safety_margin: 안전 여유거리
            
        Returns:
            저장된 파일 경로
        """
        
        # Environment 경로 해석
        possible_paths = [
            # 프로젝트 루트 기준 경로들
            f"../../../data/pointcloud/{environment}/{environment}.ply",
            f"../../../data/pointcloud/{environment}.ply",
            
            # 직접 경로
            f"{environment}.ply",
            environment  # 직접 경로인 경우
        ]
        
        ply_file = None
        for path in possible_paths:
            if os.path.exists(path):
                ply_file = path
                break
                
        if not ply_file:
            # 환경 이름에서 .ply 확장자 제거 시도
            if environment.endswith('.ply'):
                environment = environment[:-4]
            
            # 다시 시도
            possible_paths = [
                f"../../../data/pointcloud/{environment}.ply",
                f"../../../data/pointcloud/{environment}/{environment}.ply"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ply_file = path
                    break
        
        if not ply_file:
            raise ValueError(f"Environment file not found: {environment}\nTried paths: {possible_paths}")
        
        # Rigid body 설정 확인
        rigid_body_config = self.pose_generator.get_rigid_body_config(rigid_body_id)
        if rigid_body_config is None:
            raise ValueError(f"Rigid body ID {rigid_body_id} not found")
        
        print(f"🚀 Generating SE(3) poses for {environment} with {rigid_body_config.name}")
        print(f"   Target poses: {num_poses}")
        print(f"   PLY file: {ply_file}")
        print(f"   Rigid body: {rigid_body_config.name} ({rigid_body_config.semi_major_axis}×{rigid_body_config.semi_minor_axis}m)")
        
        # 포즈 생성
        start_time = time.time()
        poses = self.pose_generator.generate_multiple_poses(
            rigid_body_id=rigid_body_id,
            ply_file=ply_file,
            num_poses=num_poses,
            max_attempts=max_attempts,
            safety_margin=safety_margin
        )
        generation_time = time.time() - start_time
        
        # 환경 메타데이터 수집
        env_metadata = self._extract_environment_metadata(ply_file, environment)
        
        # Rigid body 메타데이터 수집
        rigid_body_metadata = self._extract_rigid_body_metadata(rigid_body_id)
        
        # 생성 통계
        statistics = {
            'total_generated': len(poses),
            'target_poses': num_poses,
            'success_rate': len(poses) / num_poses * 100 if num_poses > 0 else 0,
            'generation_time': generation_time,
            'poses_per_second': len(poses) / generation_time if generation_time > 0 else 0
        }
        
        # 저장할 데이터 구성
        pose_data = {
            'environment': {
                'name': environment,
                'ply_file': ply_file,
                'metadata': env_metadata
            },
            'rigid_body': {
                'id': rigid_body_id,
                'metadata': rigid_body_metadata
            },
            'poses': {
                'data': poses,
                'count': len(poses),
                'format': 'se3_poses',
                'coordinate_system': 'world_frame',
                'fixed_dof': {'z': 0.0, 'roll': 0.0, 'pitch': 0.0}
            },
            'generation_info': {
                'target_poses': num_poses,
                'achieved_poses': len(poses),
                'generation_time': generation_time,
                'safety_margin': safety_margin,
                'max_attempts': max_attempts,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'statistics': statistics
        }
        
        # 파일명 생성 및 저장
        output_file = self._generate_filename(environment, rigid_body_id)
        self._save_pose_data(pose_data, output_file)
        
        return output_file
    
    def _extract_environment_metadata(self, ply_file: str, env_name: str) -> Dict[str, Any]:
        """환경 메타데이터 추출"""
        metadata = {
            'name': env_name,
            'ply_file': ply_file
        }
        
        # JSON 메타데이터 파일 찾기
        json_paths = [
            ply_file.replace('.ply', '_meta.json'),
            os.path.join(os.path.dirname(ply_file), f"{os.path.basename(ply_file).replace('.ply', '')}_meta.json")
        ]
        
        for json_path in json_paths:
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        json_metadata = json.load(f)
                        metadata.update(json_metadata)
                        break
                except Exception as e:
                    print(f"Warning: Could not load metadata from {json_path}: {e}")
        
        # PLY 파일에서 기본 정보 추출
        try:
            with open(ply_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('element vertex'):
                        metadata['num_points'] = int(line.split()[-1])
                        break
        except Exception as e:
            print(f"Warning: Could not extract PLY info: {e}")
        
        return metadata
    
    def _extract_rigid_body_metadata(self, rigid_body_id: int) -> Dict[str, Any]:
        """Rigid body 메타데이터 추출"""
        config = self.pose_generator.get_rigid_body_config(rigid_body_id)
        
        if config:
            return {
                'id': rigid_body_id,
                'name': config.name,
                'type': config.type,
                'semi_major_axis': config.semi_major_axis,
                'semi_minor_axis': config.semi_minor_axis,
                'mass': config.mass,
                'color': config.color
            }
        else:
            return {
                'id': rigid_body_id,
                'name': 'Unknown',
                'type': 'Unknown',
                'semi_major_axis': 0.0,
                'semi_minor_axis': 0.0,
                'mass': 0.0,
                'color': [0.0, 0.0, 0.0]
            }
    
    def _generate_filename(self, environment: str, rigid_body_id: int) -> str:
        """파일명 생성"""
        
        # circle_envs 형태 처리
        if 'circle_envs' in environment and '/' in environment:
            # circle_envs_10k/env_0001 -> circle_envs/env_0001_rb_0_poses.json
            parts = environment.split('/')
            env_folder = parts[0].replace('_10k', '')  # circle_envs_10k -> circle_envs
            env_name = parts[1]  # env_0001
            
            folder_path = self.data_dir / env_folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"{env_name}_rb_{rigid_body_id}_poses.json"
            return str(folder_path / filename)
        
        # 기본 환경들 처리
        else:
            filename = f"{environment}_rb_{rigid_body_id}_poses.json"
            return str(self.data_dir / filename)
    
    def _save_pose_data(self, data: Dict[str, Any], filename: str) -> None:
        """포즈 데이터 저장"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        achieved = data['poses']['count']
        target = data['generation_info']['target_poses']
        success_rate = data['statistics']['success_rate']
        print(f"💾 Saved {achieved}/{target} SE(3) poses ({success_rate:.1f}% success) to {filename}")


# 호환성을 위한 별칭
BatchPoseGenerator = SE3BatchPoseGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate batch of SE(3) poses')
    parser.add_argument('environment', type=str, help='Environment name')
    parser.add_argument('rigid_body_id', type=int, help='Rigid body ID')
    parser.add_argument('--num_poses', type=int, default=100, help='Number of poses to generate')
    parser.add_argument('--max_attempts', type=int, default=1000, help='Max attempts per pose')
    parser.add_argument('--safety_margin', type=float, default=0.05, help='Safety margin')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='../../../data/pose/circle_envs_10k',
                      help='Output directory for pose files')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # SE3BatchPoseGenerator 초기화
    generator = SE3BatchPoseGenerator()
    
    # 출력 디렉토리 설정
    generator.data_dir = Path(args.output_dir)
    generator.data_dir.mkdir(parents=True, exist_ok=True)
    
    # 포즈 생성 및 저장
    output_file = generator.generate_and_save_poses(
        environment=args.environment,
        rigid_body_id=args.rigid_body_id,
        num_poses=args.num_poses,
        max_attempts=args.max_attempts,
        safety_margin=args.safety_margin
    )
    
    print(f"✅ Poses saved to: {output_file}")

if __name__ == '__main__':
    main() 