#!/usr/bin/env python3
"""
SE(3) RRT-Connect Planner Module  
SE(3) rigid body를 위한 OMPL RRT-Connect 알고리즘 궤적 계획

주요 기능:
- SE(3) RRT-Connect 플래너 실행 (실제로는 SE(2): x, y, yaw)
- collision_detector와 연동된 isStateValid 구현
- SE(3) trajectory 후처리 (interpolation, smoothing)
- 결과 데이터 저장
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from ompl import base as ob
from ompl import geometric as og

from .ompl_setup import SE3OMPLSetup, RigidBodyConfig, create_rigid_body_config


@dataclass 
class SE3TrajectoryResult:
    """SE(3) 궤적 계획 결과"""
    success: bool
    trajectory: List[List[float]]  # [[x, y, z, roll, pitch, yaw], ...]
    timestamps: List[float]        # [t0, t1, t2, ...]
    planning_time: float
    path_length: float
    num_waypoints: int
    planner_settings: Dict[str, Any]
    metadata: Dict[str, Any]


class SE3RRTConnectPlanner:
    """SE(3) Rigid Body를 위한 RRT-Connect 플래너 클래스"""
    
    def __init__(self, rigid_body_config: RigidBodyConfig, pointcloud_file: str = None):
        """
        Args:
            rigid_body_config: SE(3) rigid body 설정
            pointcloud_file: 환경 PLY 파일 경로
        """
        self.config = rigid_body_config
        self.pointcloud_file = pointcloud_file
        
        # OMPL setup 초기화
        self.ompl_setup = SE3OMPLSetup(rigid_body_config, pointcloud_file)
        self.planner = None
        
        # 플래너 설정
        self.planner_settings = {
            "max_planning_time": 15.0,
            "range": 0.25,  # RRT-Connect range parameter (default: 0.25 for balanced performance)
            "goal_bias": 0.05,
            "interpolate": True,
            "simplify": True
        }
        
        print(f"✅ SE(3) RRT-Connect planner initialized for rigid body {rigid_body_config.rigid_body_id}")
    
    def plan_trajectory(self, start_pose: List[float], goal_pose: List[float], 
                       max_planning_time: float = None) -> SE3TrajectoryResult:
        """
        SE(3) 궤적 계획 실행
        
        Args:
            start_pose: SE(3) start pose [x, y, z, roll, pitch, yaw]
            goal_pose: SE(3) goal pose [x, y, z, roll, pitch, yaw]
            max_planning_time: 최대 계획 시간 (초)
            
        Returns:
            SE3TrajectoryResult
        """
        if max_planning_time is not None:
            self.planner_settings["max_planning_time"] = max_planning_time
        
        print(f"🚀 Starting SE(3) trajectory planning...")
        print(f"   Start: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {start_pose[5]:.2f}]")
        print(f"   Goal:  [{goal_pose[0]:.2f}, {goal_pose[1]:.2f}, {goal_pose[5]:.2f}]")
        
        start_time = time.time()
        
        try:
            # Problem definition 생성
            pdef = self.ompl_setup.create_problem_definition(start_pose, goal_pose)
            
            # RRT-Connect 플래너 생성
            self._setup_planner(pdef)
            
            # 계획 실행
            solved = self.planner.solve(self.planner_settings["max_planning_time"])
            planning_time = time.time() - start_time
            
            if solved:
                print(f"✅ Planning successful in {planning_time:.3f}s")
                return self._process_solution(pdef, planning_time)
            else:
                print(f"❌ Planning failed after {planning_time:.3f}s")
                return SE3TrajectoryResult(
                    success=False,
                    trajectory=[],
                    timestamps=[],
                    planning_time=planning_time,
                    path_length=0.0,
                    num_waypoints=0,
                    planner_settings=self.planner_settings.copy(),
                    metadata={"error": "Planning timeout or no solution found"}
                )
                
        except Exception as e:
            planning_time = time.time() - start_time
            print(f"❌ Planning error: {e}")
            return SE3TrajectoryResult(
                success=False,
                trajectory=[],
                timestamps=[],
                planning_time=planning_time,
                path_length=0.0,
                num_waypoints=0,
                planner_settings=self.planner_settings.copy(),
                metadata={"error": str(e)}
            )
    
    def _setup_planner(self, pdef: ob.ProblemDefinition):
        """RRT-Connect 플래너 설정"""
        # RRT-Connect 플래너 생성
        self.planner = og.RRTConnect(self.ompl_setup.get_space_information())
        
        # 플래너 파라미터 설정
        self.planner.setRange(self.planner_settings["range"])
        
        # Problem definition 설정
        self.planner.setProblemDefinition(pdef)
        self.planner.setup()
        
        print(f"✅ RRT-Connect planner configured:")
        print(f"   - Range: {self.planner_settings['range']}")
        print(f"   - Max time: {self.planner_settings['max_planning_time']}s")
    
    def _process_solution(self, pdef: ob.ProblemDefinition, planning_time: float) -> SE3TrajectoryResult:
        """계획 결과 처리 및 후처리"""
        # Solution path 가져오기
        solution_path = pdef.getSolutionPath()
        
        # Path interpolation (더 부드러운 경로)
        if self.planner_settings["interpolate"]:
            solution_path.interpolate()
        
        # Path simplification (OMPL 버전 호환성)
        if self.planner_settings["simplify"]:
            try:
                solution_path.simplify()
            except AttributeError:
                # 일부 OMPL 버전에서는 simplify 메서드가 없을 수 있음
                print("⚠️ Path simplification not available in this OMPL version")
        
        # SE(3) trajectory 추출
        trajectory = self._extract_se3_trajectory(solution_path)
        
        # Path length 계산
        path_length = self._calculate_path_length(trajectory)
        
        # Timestamps 생성 (uniform spacing)
        timestamps = self._generate_timestamps(len(trajectory))
        
        print(f"✅ Solution processed:")
        print(f"   - Waypoints: {len(trajectory)}")
        print(f"   - Path length: {path_length:.3f}")
        
        return SE3TrajectoryResult(
            success=True,
            trajectory=trajectory,
            timestamps=timestamps,
            planning_time=planning_time,
            path_length=path_length,
            num_waypoints=len(trajectory),
            planner_settings=self.planner_settings.copy(),
            metadata={
                "rigid_body_id": self.config.rigid_body_id,
                "rigid_body_name": self.config.name,
                "environment_file": self.pointcloud_file
            }
        )
    
    def _extract_se3_trajectory(self, solution_path) -> List[List[float]]:
        """OMPL solution path에서 SE(3) trajectory 추출"""
        trajectory = []
        
        for i in range(solution_path.getStateCount()):
            state = solution_path.getState(i)
            pose = self.ompl_setup.state_to_pose(state)
            trajectory.append(pose)
        
        return trajectory
    
    def _calculate_path_length(self, trajectory: List[List[float]]) -> float:
        """SE(3) 경로 길이 계산 (position만 고려)"""
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            prev_pose = trajectory[i-1]
            curr_pose = trajectory[i]
            
            # Position distance
            dx = curr_pose[0] - prev_pose[0]  # x
            dy = curr_pose[1] - prev_pose[1]  # y
            
            # Orientation difference (yaw)
            dyaw = abs(curr_pose[5] - prev_pose[5])  # yaw
            if dyaw > np.pi:
                dyaw = 2*np.pi - dyaw  # Shortest angular distance
            
            # Combined distance (weighted)
            position_dist = np.sqrt(dx*dx + dy*dy)
            orientation_dist = dyaw * 0.5  # Weight for orientation
            
            total_length += position_dist + orientation_dist
        
        return total_length
    
    def _generate_timestamps(self, num_waypoints: int) -> List[float]:
        """Uniform timestamp 생성"""
        if num_waypoints <= 1:
            return [0.0]
        
        return [i / (num_waypoints - 1) for i in range(num_waypoints)]
    

    def update_planner_settings(self, **kwargs):
        """플래너 설정 업데이트"""
        for key, value in kwargs.items():
            if key in self.planner_settings:
                self.planner_settings[key] = value
                print(f"✅ Updated {key} = {value}")
            else:
                print(f"⚠️ Unknown setting: {key}")


def create_se3_planner(rigid_body_id: int, pointcloud_file: str = None, 
                       workspace_bounds: Tuple[float, float, float, float] = None) -> SE3RRTConnectPlanner:
    """
    SE(3) RRT-Connect 플래너 생성 편의 함수
    
    Args:
        rigid_body_id: Rigid body ID (0-3)
        pointcloud_file: 환경 PLY 파일 경로
        workspace_bounds: Optional workspace bounds
        
    Returns:
        SE3RRTConnectPlanner instance
    """
    rigid_body_config = create_rigid_body_config(rigid_body_id, workspace_bounds)
    return SE3RRTConnectPlanner(rigid_body_config, pointcloud_file)


def plan_se3_trajectory_from_poses(rigid_body_id: int, start_pose: List[float], 
                                  goal_pose: List[float], pointcloud_file: str = None,
                                  max_planning_time: float = 15.0) -> SE3TrajectoryResult:
    """
    SE(3) poses로부터 직접 trajectory 계획
    
    Args:
        rigid_body_id: Rigid body ID
        start_pose: Start SE(3) pose
        goal_pose: Goal SE(3) pose 
        pointcloud_file: Environment file
        max_planning_time: Max planning time
        
    Returns:
        SE3TrajectoryResult
    """
    planner = create_se3_planner(rigid_body_id, pointcloud_file)
    return planner.plan_trajectory(start_pose, goal_pose, max_planning_time)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 SE(3) RRT-Connect Planner Test")
    
    # Test configuration
    rigid_body_id = 3  # elongated_ellipse
    
    # Test poses
    start_pose = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # [x, y, z, roll, pitch, yaw]
    goal_pose = [9.0, 9.0, 0.0, 0.0, 0.0, 1.57]   # 90도 회전
    
    # Environment file (optional)
    pointcloud_file = "packages/simulation/robot_simulation/legacy/simple_endeffector_sim/data/pointcloud/circles_only/circles_only.ply"
    
    try:
        # Create planner
        print("Creating SE(3) planner...")
        planner = create_se3_planner(rigid_body_id, pointcloud_file)
        
        # Plan trajectory
        print("Planning trajectory...")
        result = planner.plan_trajectory(start_pose, goal_pose, max_planning_time=3.0)
        
        # Results
        if result.success:
            print(f"✅ Success! Generated {result.num_waypoints} waypoints")
            print(f"   Planning time: {result.planning_time:.3f}s")
            print(f"   Path length: {result.path_length:.3f}")
            
            print("✅ Planning completed successfully!")
        else:
            print("❌ Planning failed")
            print(f"   Error: {result.metadata}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 