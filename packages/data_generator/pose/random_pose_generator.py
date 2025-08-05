#!/usr/bin/env python3
"""
SE(3) Random Pose Generator
Rigid body의 SE(3) 포즈를 생성하는 모듈

포즈: SE(3) [x, y, z=0, roll=0, pitch=0, yaw] (2D 평면에서 3-DOF)
"""

import numpy as np
import math
import random
import yaml
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from .collision_detector import RigidBodyCollisionDetector
except ImportError:
    from collision_detector import RigidBodyCollisionDetector


@dataclass
class SE3PoseGenerationConfig:
    """SE(3) 포즈 생성 설정"""
    workspace_bounds: Tuple[float, float, float, float]  # (min_x, max_x, min_y, max_y)
    yaw_limits: Tuple[float, float]  # (min_yaw, max_yaw)
    safety_margin: float
    max_attempts: int


class SE3RandomPoseGenerator:
    """SE(3) 랜덤 포즈 생성기"""
    
    def __init__(self, config_file: str = "config/rigid_body_configs.yaml", seed: Optional[int] = None):
        """
        Args:
            config_file: rigid body 설정 파일 경로
            seed: 랜덤 시드
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Collision detector 초기화 (rigid body config 공유)
        self.collision_detector = RigidBodyCollisionDetector(config_file)
        self.generation_config = self._load_generation_config(config_file)
        
    def _load_generation_config(self, config_file: str) -> SE3PoseGenerationConfig:
        """config에서 포즈 생성 설정 로드"""
        config_path = Path(config_file)
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / config_file
        
        if not config_path.exists():
            # 기본값 사용
            return SE3PoseGenerationConfig(
                workspace_bounds=(0.0, 10.0, 0.0, 8.0),
                yaw_limits=(-math.pi, math.pi),
                safety_margin=0.05,
                max_attempts=1000
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Simulation 설정에서 값 추출
        sim_config = config.get('simulation', {})
        workspace = sim_config.get('workspace_bounds', {})
        orientation = sim_config.get('orientation_bounds', {})
        
        return SE3PoseGenerationConfig(
            workspace_bounds=(
                workspace.get('x_min', 0.0),
                workspace.get('x_max', 10.0),
                workspace.get('y_min', 0.0),
                workspace.get('y_max', 8.0)
            ),
            yaw_limits=(
                orientation.get('yaw_min', -math.pi),
                orientation.get('yaw_max', math.pi)
            ),
            safety_margin=sim_config.get('collision_margin', 0.05),
            max_attempts=1000
        )
    
    def get_rigid_body_config(self, rigid_body_id: int):
        """특정 rigid body ID의 설정 정보 반환"""
        return self.collision_detector.get_rigid_body_config(rigid_body_id)
    
    def list_available_rigid_bodies(self) -> List[int]:
        """사용 가능한 rigid body ID 목록"""
        return self.collision_detector.list_available_rigid_bodies()
    
    def generate_random_se3_pose(self, 
                                workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
                                yaw_limits: Optional[Tuple[float, float]] = None) -> List[float]:
        """
        랜덤 SE(3) 포즈 생성 (충돌 검사 없이)
        
        Args:
            workspace_bounds: 작업공간 경계 (min_x, max_x, min_y, max_y)
            yaw_limits: yaw 각도 제한 (min_yaw, max_yaw)
            
        Returns:
            [x, y, z=0, roll=0, pitch=0, yaw] SE(3) 포즈
        """
        
        # 기본값 설정
        if workspace_bounds is None:
            workspace_bounds = self.generation_config.workspace_bounds
        if yaw_limits is None:
            yaw_limits = self.generation_config.yaw_limits
        
        min_x, max_x, min_y, max_y = workspace_bounds
        min_yaw, max_yaw = yaw_limits
        
        # 랜덤 SE(3) 포즈 생성
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = 0.0  # 2D 시뮬레이션
        roll = 0.0  # 2D 시뮬레이션
        pitch = 0.0  # 2D 시뮬레이션
        yaw = random.uniform(min_yaw, max_yaw)
        
        return [x, y, z, roll, pitch, yaw]
    
    def generate_collision_free_pose(self, 
                                   rigid_body_id: int,
                                   ply_file: str,
                                   workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
                                   yaw_limits: Optional[Tuple[float, float]] = None,
                                   safety_margin: Optional[float] = None,
                                   max_attempts: int = 1000) -> Optional[List[float]]:
        """
        충돌 없는 SE(3) 포즈 생성
        
        Args:
            rigid_body_id: Rigid body ID
            ply_file: PLY 환경 파일 경로
            workspace_bounds: 작업공간 경계
            yaw_limits: yaw 각도 제한
            safety_margin: 안전 여유 거리
            max_attempts: 최대 시도 횟수
            
        Returns:
            충돌 없는 SE(3) 포즈 또는 None (실패시)
        """
        
        # 기본값 설정
        if safety_margin is None:
            safety_margin = self.generation_config.safety_margin
        
        # 환경 로드
        if not self.collision_detector.load_environment(ply_file):
            print(f"Error: Failed to load environment {ply_file}")
            return None
        
        # Rigid body 설정 확인
        config = self.get_rigid_body_config(rigid_body_id)
        if config is None:
            print(f"Error: Rigid body ID {rigid_body_id} not found")
            return None
        
        print(f"Generating collision-free pose for {config.name}...")
        
        for attempt in range(max_attempts):
            # 랜덤 SE(3) 포즈 생성
            pose = self.generate_random_se3_pose(workspace_bounds, yaw_limits)
            
            # 충돌 검사
            result = self.collision_detector.check_collision(pose, rigid_body_id, safety_margin)
            
            if not result.is_collision:
                print(f"Found collision-free pose after {attempt + 1} attempts")
                return pose
            
            # 진행상황 출력
            if (attempt + 1) % 100 == 0:
                print(f"Attempt {attempt + 1}/{max_attempts}...")
        
        print(f"Warning: Failed to generate collision-free pose after {max_attempts} attempts")
        return None
    
    def generate_multiple_poses(self, 
                               rigid_body_id: int,
                               ply_file: str,
                               num_poses: int,
                               workspace_bounds: Optional[Tuple[float, float, float, float]] = None,
                               yaw_limits: Optional[Tuple[float, float]] = None,
                               safety_margin: Optional[float] = None,
                               max_attempts: int = 1000) -> List[List[float]]:
        """
        여러 개의 충돌 없는 SE(3) 포즈 생성
        
        Args:
            rigid_body_id: Rigid body ID
            ply_file: PLY 환경 파일 경로
            num_poses: 생성할 포즈 개수
            workspace_bounds: 작업공간 경계
            yaw_limits: yaw 각도 제한
            safety_margin: 안전 여유 거리
            max_attempts: 각 포즈당 최대 시도 횟수
            
        Returns:
            생성된 SE(3) 포즈들의 리스트
        """
        
        # 기본값 설정
        if safety_margin is None:
            safety_margin = self.generation_config.safety_margin
        
        # 환경 로드 (한 번만)
        if not self.collision_detector.load_environment(ply_file):
            print(f"Error: Failed to load environment {ply_file}")
            return []
        
        # Rigid body 설정 확인
        config = self.get_rigid_body_config(rigid_body_id)
        if config is None:
            print(f"Error: Rigid body ID {rigid_body_id} not found")
            return []
        
        print(f"Generating {num_poses} collision-free poses for {config.name}...")
        
        poses = []
        total_attempts = 0
        
        for i in range(num_poses):
            for attempt in range(max_attempts):
                total_attempts += 1
                
                # 랜덤 SE(3) 포즈 생성
                pose = self.generate_random_se3_pose(workspace_bounds, yaw_limits)
                
                # 충돌 검사
                result = self.collision_detector.check_collision(pose, rigid_body_id, safety_margin)
                
                if not result.is_collision:
                    poses.append(pose)
                    break
            
            # 진행상황 출력
            if (i + 1) % max(1, num_poses // 10) == 0:
                success_rate = len(poses) / total_attempts * 100
                print(f"Generated {len(poses)}/{i+1} poses (success rate: {success_rate:.1f}%)")
        
        final_success_rate = len(poses) / total_attempts * 100 if total_attempts > 0 else 0
        print(f"Final: {len(poses)} poses generated with {final_success_rate:.1f}% success rate")
        
        return poses
    
    def get_workspace_info(self) -> Dict:
        """작업공간 정보 반환"""
        config = self.generation_config
        return {
            'workspace_bounds': config.workspace_bounds,
            'yaw_limits': config.yaw_limits,
            'safety_margin': config.safety_margin,
            'max_attempts': config.max_attempts
        }
    
    def print_rigid_body_info(self, rigid_body_id: int):
        """Rigid body 정보 출력"""
        self.collision_detector.print_rigid_body_info(rigid_body_id)
    
    def print_generation_config(self):
        """포즈 생성 설정 정보 출력"""
        config = self.generation_config
        print(f"SE(3) Pose Generation Configuration:")
        print(f"  Workspace bounds: x=[{config.workspace_bounds[0]:.1f}, {config.workspace_bounds[1]:.1f}], "
              f"y=[{config.workspace_bounds[2]:.1f}, {config.workspace_bounds[3]:.1f}]")
        print(f"  Yaw limits: [{config.yaw_limits[0]:.2f}, {config.yaw_limits[1]:.2f}] rad")
        print(f"  Safety margin: {config.safety_margin:.3f}m")
        print(f"  Max attempts: {config.max_attempts}")


# 호환성을 위한 별칭
RandomPoseGenerator = SE3RandomPoseGenerator


if __name__ == "__main__":
    # 테스트 실행
    print("🚀 SE(3) Random Pose Generator Test...")
    
    generator = SE3RandomPoseGenerator(seed=42)
    
    # 사용 가능한 rigid body 목록
    print(f"Available rigid bodies: {generator.list_available_rigid_bodies()}")
    
    # Rigid body 정보 출력
    generator.print_rigid_body_info(0)
    
    # 생성 설정 출력
    print()
    generator.print_generation_config()
    
    # 환경 파일 경로
    test_ply = "data/pointcloud/circles_only/circles_only.ply"
    
    print(f"\n📍 Testing with environment: {test_ply}")
    
    # 단일 SE(3) 포즈 생성 테스트
    print("\n1. Random SE(3) pose generation (no collision check):")
    for i in range(3):
        pose = generator.generate_random_se3_pose()
        print(f"  Pose {i}: [x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[5]:.2f}]")
    
    # 충돌 없는 포즈 생성 테스트
    print(f"\n2. Collision-free pose generation:")
    collision_free_pose = generator.generate_collision_free_pose(
        rigid_body_id=0,
        ply_file=test_ply,
        max_attempts=200
    )
    
    if collision_free_pose:
        print(f"  ✅ Success: {collision_free_pose}")
    else:
        print(f"  ❌ Failed to generate collision-free pose")
    
    # 다중 포즈 생성 테스트
    print(f"\n3. Multiple collision-free poses:")
    poses = generator.generate_multiple_poses(
        rigid_body_id=0,
        ply_file=test_ply,
        num_poses=5,
        max_attempts=200
    )
    
    for i, pose in enumerate(poses):
        print(f"  Pose {i}: [x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[5]:.2f}]")
    
    print("\n🎉 Test completed!") 