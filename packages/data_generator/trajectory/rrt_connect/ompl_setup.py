#!/usr/bin/env python3
"""
OMPL Setup Module for SE(3) Rigid Body Planning
SE(3) rigid body를 위한 RRT-Connect 플래닝 환경 설정

주요 기능:
- SE(3) state space 설정 (실제로는 SE(2): x, y, yaw)
- collision_detector와 연동된 state validity checking
- Problem definition 생성
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ompl import base as ob
from ompl import geometric as og

print("✅ Using real OMPL library")


@dataclass
class RigidBodyConfig:
    """SE(3) Rigid Body 설정 정보"""
    rigid_body_id: int
    name: str                      # "small_ellipse", "medium_ellipse", etc.
    semi_major_axis: float         # 장축 반지름
    semi_minor_axis: float         # 단축 반지름
    mass: float                    # 질량
    workspace_bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    orientation_bounds: Tuple[float, float]              # (yaw_min, yaw_max)


class SE3OMPLSetup:
    """SE(3) Rigid Body를 위한 OMPL 설정 및 관리 클래스"""
    
    def __init__(self, rigid_body_config: RigidBodyConfig, pointcloud_file: str = None):
        """
        Args:
            rigid_body_config: SE(3) rigid body 설정
            pointcloud_file: 환경 PLY 파일 경로 (None이면 빈 환경)
        """
        self.config = rigid_body_config
        self.pointcloud_file = pointcloud_file
        
        # collision_detector import 및 초기화
        try:
            import sys
            import os
            
            # pose 모듈 경로를 sys.path에 추가
            pose_path = os.path.join(os.path.dirname(__file__), "..", "..", "pose")
            if pose_path not in sys.path:
                sys.path.insert(0, pose_path)
            
            from collision_detector import RigidBodyCollisionDetector
            self.collision_detector = RigidBodyCollisionDetector()
            
            # 환경 파일이 있으면 로드
            if pointcloud_file:
                self.collision_detector.load_environment(pointcloud_file)
                print(f"✅ Environment loaded: {pointcloud_file}")
            
            print(f"✅ Collision detector initialized for rigid body {rigid_body_config.rigid_body_id}")
        except ImportError as e:
            print(f"❌ Failed to import collision detector: {e}")
            self.collision_detector = None
        except Exception as e:
            print(f"❌ Failed to load environment: {e}")
            self.collision_detector = None
        
        # OMPL components 초기화
        self._setup_state_space()
        self._setup_space_information()
        
    def _setup_state_space(self):
        """SE(3) state space 설정 (실제로는 SE(2): x, y, yaw)"""
        # SE(3) state space 생성 (z, roll, pitch는 고정)
        self.state_space = ob.SE3StateSpace()
        
        # Bounds 설정
        bounds = ob.RealVectorBounds(3)  # x, y, z
        x_min, x_max, y_min, y_max = self.config.workspace_bounds
        bounds.setLow(0, x_min)    # x_min
        bounds.setHigh(0, x_max)   # x_max
        bounds.setLow(1, y_min)    # y_min
        bounds.setHigh(1, y_max)   # y_max
        bounds.setLow(2, 0.0)      # z (fixed)
        bounds.setHigh(2, 0.0)     # z (fixed)
        
        # Position bounds 설정
        self.state_space.setBounds(bounds)
        
        print(f"✅ SE(3) state space configured:")
        print(f"   - Position bounds: x[{x_min}, {x_max}], y[{y_min}, {y_max}], z=0")
        print(f"   - Orientation bounds: yaw[{self.config.orientation_bounds[0]:.2f}, {self.config.orientation_bounds[1]:.2f}]")
    
    def _setup_space_information(self):
        """Space information 및 state validity checker 설정"""
        self.space_info = ob.SpaceInformation(self.state_space)
        
        # State validity checker 설정
        self.space_info.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        self.space_info.setup()
        
        print("✅ Space information configured with SE(3) collision checking")
    
    def _is_state_valid(self, state) -> bool:
        """
        State validity checking using collision_detector
        
        Args:
            state: OMPL SE(3) state
            
        Returns:
            True if state is collision-free, False otherwise
        """
        if self.collision_detector is None:
            # Simple bounds checking when no collision detector available
            try:
                x = state.getX()
                y = state.getY()
                x_min, x_max, y_min, y_max = self.config.workspace_bounds
                
                # Check if state is within workspace bounds
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"⚠️ State validation error: {e}")
                return False
        
        try:
            # OMPL SE(3) state → SE(3) pose 변환
            x = state.getX()
            y = state.getY()
            z = 0.0  # 2D 시뮬레이션
            
            # Quaternion → yaw 변환 (2D 회전만 고려)
            rot = state.rotation()
            qw = rot.w
            qx = rot.x  
            qy = rot.y
            qz = rot.z
            yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            
            pose = [x, y, z, 0.0, 0.0, yaw]
            
            # Collision checking (환경은 이미 로드됨)
            collision_result = self.collision_detector.check_collision(
                pose, 
                self.config.rigid_body_id,
                safety_margin=0.05  # 기본 collision margin으로 테스트
            )
            
            # Debug output for first few calls
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 3:
                print(f"🔍 State validity check {self._debug_count}: pose={pose}, collision={collision_result.is_collision}")
                self._debug_count += 1
            
            return not collision_result.is_collision  # True if no collision
            
        except Exception as e:
            print(f"❌ Error in state validity checking: {e}")
            return False
    
    def create_problem_definition(self, start_pose: List[float], goal_pose: List[float]) -> ob.ProblemDefinition:
        """
        Problem definition 생성
        
        Args:
            start_pose: SE(3) start pose [x, y, z, roll, pitch, yaw]
            goal_pose: SE(3) goal pose [x, y, z, roll, pitch, yaw]
            
        Returns:
            OMPL problem definition
        """
        pdef = ob.ProblemDefinition(self.space_info)
        
        # Start state 설정
        start_state = self.state_space.allocState()
        self._set_state_from_pose(start_state, start_pose)
        pdef.setStartAndGoalStates(start_state, self._create_goal_state(goal_pose))
        
        print(f"✅ Problem definition created:")
        print(f"   - Start: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {start_pose[5]:.2f}]")
        print(f"   - Goal:  [{goal_pose[0]:.2f}, {goal_pose[1]:.2f}, {goal_pose[5]:.2f}]")
        
        return pdef
    
    def _set_state_from_pose(self, state, pose: List[float]):
        """SE(3) pose를 OMPL state로 변환"""
        x, y, z, roll, pitch, yaw = pose
        
        # Position 설정 (z는 항상 0)
        state.setXYZ(x, y, 0.0)
        
        # Orientation 설정 (yaw만 사용, roll=pitch=0)
        if abs(yaw) < 1e-6:
            # yaw가 0에 가까우면 identity quaternion 사용
            state.rotation().setIdentity()
        else:
            # Z축 회전 (yaw)
            state.rotation().setAxisAngle(0, 0, 1, yaw)
    
    def _create_goal_state(self, goal_pose: List[float]):
        """Goal state 생성"""
        goal_state = self.state_space.allocState()
        self._set_state_from_pose(goal_state, goal_pose)
        return goal_state
    
    def state_to_pose(self, state) -> List[float]:
        """OMPL state를 SE(3) pose로 변환"""
        try:
            # Position 추출
            x = state.getX()
            y = state.getY()
            z = state.getZ()
            
            # Rotation (quaternion) 추출
            rot = state.rotation()
            qw = rot.w
            qx = rot.x
            qy = rot.y
            qz = rot.z
            
            # Quaternion → yaw 변환 (Z축 회전만 고려)
            yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
            
            return [x, y, z, 0.0, 0.0, yaw]
            
        except Exception as e:
            print(f"❌ Error converting state to pose: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def get_space_information(self) -> ob.SpaceInformation:
        """Space information 반환"""
        return self.space_info
    
    def get_state_space(self) -> ob.SE3StateSpace:
        """State space 반환"""
        return self.state_space


def create_rigid_body_config(rigid_body_id: int, workspace_bounds: Tuple[float, float, float, float] = None) -> RigidBodyConfig:
    """
    Rigid body configuration 생성 편의 함수
    
    Args:
        rigid_body_id: Rigid body ID (0-3)
        workspace_bounds: Optional workspace bounds
        
    Returns:
        RigidBodyConfig instance
    """
    # Import rigid body config
    try:
        from config_loader import get_rigid_body_config
        rb_config = get_rigid_body_config(rigid_body_id)
        if rb_config is None:
            raise ValueError(f"Rigid body config not found for ID {rigid_body_id}")
    except ImportError:
        # Fallback configurations
        configs = {
            0: {"name": "small_ellipse", "semi_major_axis": 0.4, "semi_minor_axis": 0.2, "mass": 1.0},
            1: {"name": "medium_ellipse", "semi_major_axis": 0.7, "semi_minor_axis": 0.4, "mass": 1.8},
            2: {"name": "large_ellipse", "semi_major_axis": 1.0, "semi_minor_axis": 0.6, "mass": 2.5},
            3: {"name": "elongated_ellipse", "semi_major_axis": 1.2, "semi_minor_axis": 0.4, "mass": 2.3}
        }
        rb_config = configs.get(rigid_body_id, configs[0])
    
    # Default workspace bounds
    if workspace_bounds is None:
        workspace_bounds = (-1.0, 11.0, -1.0, 11.0)
    
    return RigidBodyConfig(
        rigid_body_id=rigid_body_id,
        name=rb_config['name'],
        semi_major_axis=rb_config['semi_major_axis'],
        semi_minor_axis=rb_config['semi_minor_axis'], 
        mass=rb_config['mass'],
        workspace_bounds=workspace_bounds,
        orientation_bounds=(-math.pi, math.pi)  # Full rotation
    )


if __name__ == "__main__":
    # 테스트 코드
    print("Testing OMPL Setup...")
    
    # 샘플 로봇 설정
    robot_config = create_rigid_body_config(0) # 예시로 0번 로봇 사용
    
    # OMPL 설정
    ompl_setup = SE3OMPLSetup(robot_config)
    
    # 환경 로드 (예시)
    # ompl_setup.load_environment("data/pointcloud/circles_only/circles_only.ply")
    
    # 문제 정의 (예시)
    # ompl_setup.create_problem_definition([0, 0, 0], [1, 0.5, -0.5])
    
    ompl_setup.print_summary()
    print("✅ OMPL Setup test completed!") 