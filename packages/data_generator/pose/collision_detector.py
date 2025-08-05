#!/usr/bin/env python3
"""
Rigid Body Collision Detector
PLY 포인트클라우드 환경과 SE(3) 포즈의 rigid body 간의 충돌을 검사하는 모듈

충돌 검사 방법:
1. PLY 파일에서 장애물 포인트들 로드
2. SE(3) 포즈의 타원체가 점유하는 공간 계산
3. 포인트-타원체 충돌 검사 수행
"""

import numpy as np
import math
import yaml
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RigidBodyCollisionResult:
    """Rigid body 충돌 검사 결과"""
    is_collision: bool
    rigid_body_pose: List[float]  # SE(3) 포즈 [x, y, z, roll, pitch, yaw]


@dataclass
class RigidBodyConfig:
    """Rigid body 설정 정보"""
    id: int
    name: str
    type: str  # "ellipse"
    semi_major_axis: float
    semi_minor_axis: float
    mass: float
    color: List[float]


class RigidBodyCollisionDetector:
    """SE(3) Rigid Body 충돌 검사기"""
    
    def __init__(self, config_file: str = "config/rigid_body_configs.yaml"):
        """
        Args:
            config_file: rigid body 설정 파일 경로
        """
        self.rigid_body_configs = self._load_rigid_body_configs(config_file)
        self.environment_points = None
        self.environment_bounds = None
    
    def _load_rigid_body_configs(self, config_file: str) -> Dict[int, RigidBodyConfig]:
        """rigid body 설정 파일 로드"""
        config_path = Path(config_file)
        if not config_path.exists():
            # 상대 경로로 다시 시도
            config_path = Path(__file__).parent.parent / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        configs = {}
        
        if 'rigid_bodies' in config:
            for body_id, body_config in config['rigid_bodies'].items():
                configs[int(body_id)] = RigidBodyConfig(
                    id=int(body_id),
                    name=body_config['name'],
                    type=body_config['type'],
                    semi_major_axis=body_config['semi_major_axis'],
                    semi_minor_axis=body_config['semi_minor_axis'],
                    mass=body_config['mass'],
                    color=body_config['color']
                )
        
        return configs
    
    def get_rigid_body_config(self, rigid_body_id: int) -> Optional[RigidBodyConfig]:
        """특정 rigid body ID의 설정 정보 반환"""
        return self.rigid_body_configs.get(rigid_body_id)
    
    def list_available_rigid_bodies(self) -> List[int]:
        """사용 가능한 rigid body ID 목록"""
        return list(self.rigid_body_configs.keys())
    
    def load_environment(self, ply_file: str) -> bool:
        """
        PLY 파일에서 환경 포인트클라우드 로드
        
        Args:
            ply_file: PLY 파일 경로

        Returns:
            로드 성공 여부
        """
        try:
            points = self._read_ply_file(ply_file)
            
            if len(points) == 0:
                print(f"Warning: No points found in {ply_file}")
                return False
            
            self.environment_points = np.array(points)
            
            # 환경 경계 계산
            min_x, min_y = np.min(self.environment_points, axis=0)
            max_x, max_y = np.max(self.environment_points, axis=0)
            self.environment_bounds = (min_x, max_x, min_y, max_y)
            
            print(f"Loaded environment: {len(points)} points")
            print(f"Environment bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")
            
            return True
            
        except Exception as e:
            print(f"Error loading PLY file {ply_file}: {e}")
            return False
    
    def _read_ply_file(self, ply_file: str) -> List[Tuple[float, float]]:
        """PLY 파일에서 2D 포인트 데이터 읽기"""
        points = []
        
        with open(ply_file, 'r') as f:
            # 헤더 스킵
            line = f.readline()
            while not line.startswith('end_header'):
                line = f.readline()
            
            # 포인트 데이터 읽기
            for line in f:
                if line.strip():
                    coords = line.strip().split()
                    if len(coords) >= 3:
                        x, y = float(coords[0]), float(coords[1])
                        points.append((x, y))
        
        return points
    
    def check_collision(self, se3_pose: List[float], rigid_body_id: int, 
                       safety_margin: float = 0.05) -> RigidBodyCollisionResult:
        """
        SE(3) 포즈의 rigid body와 환경 간 충돌 검사
        
        Args:
            se3_pose: [x, y, z, roll, pitch, yaw] SE(3) 포즈
            rigid_body_id: Rigid body ID
            safety_margin: 안전 여유 거리 (타원체 확장 크기)
            
        Returns:
            RigidBodyCollisionResult 객체 (is_collision: bool)
        """
        
        if self.environment_points is None:
            print("Error: Environment not loaded. Call load_environment() first.")
            return RigidBodyCollisionResult(False, se3_pose)
        
        config = self.get_rigid_body_config(rigid_body_id)
        if config is None:
            print(f"Error: Rigid body ID {rigid_body_id} not found")
            return RigidBodyCollisionResult(False, se3_pose)
        
        # SE(3) 포즈에서 SE(2) 추출 (z, roll, pitch 무시)
        x, y = se3_pose[0], se3_pose[1]
        yaw = se3_pose[5] if len(se3_pose) > 5 else 0.0
        
        # 안전 여유 포함한 타원체 크기
        effective_semi_major = config.semi_major_axis + safety_margin
        effective_semi_minor = config.semi_minor_axis + safety_margin
        
        # 충돌 검사 수행 (하나라도 충돌하면 즉시 True 반환)
        is_collision = self._check_ellipse_collision_fast(
            center=(x, y),
            yaw=yaw,
            semi_major=effective_semi_major,
            semi_minor=effective_semi_minor
        )
        
        return RigidBodyCollisionResult(
            is_collision=is_collision,
            rigid_body_pose=se3_pose
        )
    
    def _check_ellipse_collision_fast(self, center: Tuple[float, float], yaw: float,
                                      semi_major: float, semi_minor: float) -> bool:
        """
        SE(2) 포즈의 타원체와 환경 포인트들 간의 빠른 충돌 검사
        하나라도 충돌하면 즉시 True 반환 (early return)
        
        Args:
            center: 타원체 중심 (x, y)
            yaw: 타원체 방향 (라디안)
            semi_major: 장축 반지름
            semi_minor: 단축 반지름
            
        Returns:
            충돌 여부 (bool)
        """
        
        cx, cy = center
        
        # 회전 변환 계산 (역회전)
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        
        # 모든 환경 포인트에 대해 충돌 검사 (early return)
        for point in self.environment_points:
            px, py = point[0], point[1]
            
            if self._point_in_ellipse_se2(px, py, cx, cy, yaw, semi_major, semi_minor,
                                         cos_yaw, sin_yaw):
                return True  # 하나라도 충돌하면 즉시 True 반환
        
        return False  # 모든 점이 충돌하지 않으면 False
    
    def _point_in_ellipse_se2(self, px: float, py: float, 
                             cx: float, cy: float, yaw: float,
                             semi_major: float, semi_minor: float,
                             cos_yaw: float, sin_yaw: float) -> bool:
        """
        점이 SE(2) 포즈의 타원체 내부에 있는지 확인
        
        Args:
            px, py: 검사할 점
            cx, cy: 타원체 중심
            yaw: 타원체 방향 (사용되지 않음, cos_yaw, sin_yaw 사용)
            semi_major: 장축 반지름  
            semi_minor: 단축 반지름
            cos_yaw, sin_yaw: 미리 계산된 회전 변환 값
            
        Returns:
            True if 점이 타원체 내부에 있음
        """
        
        # 점을 타원체 중심 기준 좌표계로 변환
        dx = px - cx
        dy = py - cy
        
        # yaw 회전 적용하여 로컬 좌표 계산 (역회전)
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw
        
        # 표준 타원 방정식 검사: (x/a)^2 + (y/b)^2 <= 1
        return (local_x/semi_major)**2 + (local_y/semi_minor)**2 <= 1
    
    def check_multiple_poses(self, se3_poses: List[List[float]], rigid_body_id: int,
                           safety_margin: float = 0.05) -> List[bool]:
        """
        여러 SE(3) 포즈에 대한 일괄 충돌 검사
        
        Args:
            se3_poses: SE(3) 포즈 리스트
            rigid_body_id: Rigid body ID
            safety_margin: 안전 여유 거리 (타원체 확장 크기)
            
        Returns:
            각 포즈의 충돌 여부 리스트 (bool)
        """
        
        collision_results = []
        
        for i, pose in enumerate(se3_poses):
            result = self.check_collision(pose, rigid_body_id, safety_margin)
            collision_results.append(result.is_collision)
            
            # 진행상황 출력
            if (i + 1) % max(1, len(se3_poses) // 10) == 0:
                collision_count = sum(collision_results)
                print(f"Checked {i+1}/{len(se3_poses)} poses, {collision_count} collisions found")
        
        return collision_results
    
    def get_collision_free_poses(self, se3_poses: List[List[float]], rigid_body_id: int,
                                safety_margin: float = 0.05) -> List[List[float]]:
        """
        충돌 없는 SE(3) 포즈들만 필터링
        
        Args:
            se3_poses: 입력 SE(3) 포즈 리스트
            rigid_body_id: Rigid body ID
            safety_margin: 안전 여유 거리 (타원체 확장 크기)
            
        Returns:
            충돌 없는 SE(3) 포즈들의 리스트
        """
        
        collision_results = self.check_multiple_poses(se3_poses, rigid_body_id, safety_margin)
        
        collision_free_poses = []
        for pose, is_collision in zip(se3_poses, collision_results):
            if not is_collision:
                collision_free_poses.append(pose)
        
        collision_count = len(se3_poses) - len(collision_free_poses)
        print(f"Filtered poses: {len(collision_free_poses)} collision-free out of {len(se3_poses)} total")
        print(f"Collision rate: {collision_count/len(se3_poses)*100:.1f}%")
        
        return collision_free_poses
    
    def print_collision_summary(self, result: RigidBodyCollisionResult):
        """충돌 검사 결과 요약 출력"""
        
        if result.is_collision:
            print(f"❌ COLLISION DETECTED!")
            print(f"  Rigid body pose: {result.rigid_body_pose}")
        else:
            print("✅ No collision detected.")
    
    def print_rigid_body_info(self, rigid_body_id: int):
        """Rigid body 정보 출력"""
        config = self.get_rigid_body_config(rigid_body_id)
        if config is None:
            print(f"Rigid body ID {rigid_body_id} not found")
            return
        
        print(f"Rigid Body ID {rigid_body_id}: {config.name}")
        print(f"  Type: {config.type}")
        print(f"  Semi-major axis: {config.semi_major_axis}m")
        print(f"  Semi-minor axis: {config.semi_minor_axis}m")
        print(f"  Mass: {config.mass}kg")
        print(f"  Color: {config.color}")


if __name__ == "__main__":
    # 간단한 테스트
    print("Starting rigid body collision detection test...")
    
    test_ply = "data/pointcloud/circles_only/circles_only.ply"
    
    # 충돌 검사기 초기화
    detector = RigidBodyCollisionDetector()
    
    # 사용 가능한 rigid body 확인
    print(f"Available rigid bodies: {detector.list_available_rigid_bodies()}")
    
    # Rigid body 정보 출력
    detector.print_rigid_body_info(0)
    
    if detector.load_environment(test_ply):
        print("\nGenerating test SE(3) poses...")
        
        # 테스트 SE(3) 포즈 생성 (x, y, z=0, roll=0, pitch=0, yaw)
        test_poses = []
        for i in range(10):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            yaw = np.random.uniform(-math.pi, math.pi)
            pose = [x, y, 0.0, 0.0, 0.0, yaw]
            test_poses.append(pose)
        
        print(f"\nTesting collision detection on {len(test_poses)} SE(3) poses...")
        
        # 충돌 검사 수행
        collision_results = detector.check_multiple_poses(test_poses, rigid_body_id=0)
        
        # 결과 분석
        collision_count = sum(collision_results)
        print(f"\nCollision Summary:")
        print(f"  Total poses tested: {len(collision_results)}")
        print(f"  Collisions found: {collision_count}")
        print(f"  Collision rate: {collision_count/len(collision_results)*100:.1f}%")
        
        # 각 포즈 결과 출력
        for i, is_collision in enumerate(collision_results):
            status = '❌ COLLISION' if is_collision else '✅ FREE'
            print(f"  Pose {i}: {status}")
        
        # 충돌 없는 포즈 필터링 테스트
        collision_free = detector.get_collision_free_poses(test_poses, rigid_body_id=0)
        print(f"\nFiltered to {len(collision_free)} collision-free SE(3) poses")
    
    else:
        print(f"Could not load test environment: {test_ply}")
        print("Please ensure the PLY file exists for testing.")
    
    print("\nTesting completed!") 