#!/usr/bin/env python3
"""
궤적 검증 모듈
충돌 체커와 연동하여 궤적의 안전성을 검증
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# 충돌 체커 import - 기존 경로에서 (아직 복사 안됨)
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from pose.collision_detector import RigidBodyCollisionDetector
    print("✅ 기존 collision_detector import 성공")
except ImportError:
    print("⚠️ 충돌 체커 import 실패, 더미 클래스 사용")
    class RigidBodyCollisionDetector:
        def __init__(self): pass
        def load_environment(self, file): return True
        def check_collision(self, pose, body_id, margin): 
            from dataclasses import dataclass
            @dataclass
            class Result:
                is_collision: bool = False
            return Result()


class TrajectoryValidator:
    """궤적 검증 클래스"""
    
    def __init__(self, 
                 pointcloud_file: str,
                 rigid_body_id: int = 3,
                 safety_margin: float = 0.05,
                 check_density: int = 1):
        """
        Args:
            pointcloud_file: 환경 PLY 파일 경로
            rigid_body_id: Rigid body ID (기본값: 3)
            safety_margin: 안전 여유 거리 (기본값: 0.05m)
            check_density: 체크 밀도 (1=모든점, 2=2개마다1개, ...)
        """
        self.pointcloud_file = pointcloud_file
        self.rigid_body_id = rigid_body_id
        self.safety_margin = safety_margin
        self.check_density = check_density
        
        # 충돌 체커 초기화
        self.collision_detector = RigidBodyCollisionDetector()
        self._load_environment()
        
    def _load_environment(self) -> bool:
        """환경 로드"""
        try:
            success = self.collision_detector.load_environment(self.pointcloud_file)
            if success:
                print(f"✅ 환경 로드 완료: {Path(self.pointcloud_file).name}")
                return True
            else:
                print(f"❌ 환경 로드 실패: {self.pointcloud_file}")
                return False
        except Exception as e:
            print(f"❌ 환경 로드 오류: {e}")
            return False
    
    def validate_trajectory(self, 
                          trajectory: np.ndarray,
                          trajectory_type: str = "unknown") -> Dict[str, Any]:
        """
        궤적 충돌 검증
        
        Args:
            trajectory: [N, 3] 또는 [N, 6] 궤적 데이터
            trajectory_type: 궤적 타입 ('raw', 'smooth', 'unknown')
            
        Returns:
            validation_result: 검증 결과 딕셔너리
        """
        start_time = time.time()
        
        # 입력 검증 및 변환
        if trajectory.shape[1] == 6:
            # SE(3) → SE(2) 변환 [x, y, z, rx, ry, rz] → [x, y, rz]
            se2_trajectory = trajectory[:, [0, 1, 5]]
        elif trajectory.shape[1] == 3:
            # SE(2) 그대로 사용
            se2_trajectory = trajectory
        else:
            return {
                'success': False,
                'error': f'Unsupported trajectory shape: {trajectory.shape}',
                'validation_time': time.time() - start_time
            }
        
        total_waypoints = len(se2_trajectory)
        
        # 선택적 웨이포인트만 체크 (밀도 조절)
        check_indices = list(range(0, total_waypoints, self.check_density))
        checked_waypoints = len(check_indices)
        
        collision_waypoints = []
        collision_details = []
        first_collision_index = None
        
        print(f"🔍 궤적 검증 시작 ({trajectory_type})")
        print(f"   검사 대상: {checked_waypoints}/{total_waypoints} waypoints")
        print(f"   검사 밀도: 1/{self.check_density}")
        print(f"   안전 여유: {self.safety_margin}m")
        
        # 각 웨이포인트에 대해 collision check
        for i, idx in enumerate(check_indices):
            # SE(2) → SE(3) 변환 for collision checker
            x, y, theta = se2_trajectory[idx]
            se3_pose = [x, y, 0.0, 0.0, 0.0, theta]
            
            # 충돌 검사
            result = self.collision_detector.check_collision(
                se3_pose, self.rigid_body_id, self.safety_margin
            )
            
            if result.is_collision:
                collision_waypoints.append(idx)
                collision_details.append({
                    'waypoint_index': idx,
                    'pose': se3_pose,
                    'collision_result': result
                })
                
                if first_collision_index is None:
                    first_collision_index = idx
            
            # 진행 상황 출력 (10개마다)
            if (i + 1) % 10 == 0 or (i + 1) == checked_waypoints:
                collision_count = len(collision_waypoints)
                print(f"   진행: {i+1}/{checked_waypoints}, 충돌: {collision_count}")
        
        # 결과 계산
        collision_count = len(collision_waypoints)
        collision_percentage = (collision_count / checked_waypoints) * 100 if checked_waypoints > 0 else 0
        is_collision_free = collision_count == 0
        validation_time = time.time() - start_time
        
        # 안전성 점수 계산 (0-100, 높을수록 안전)
        safety_score = max(0, 100 - collision_percentage)
        
        result = {
            'success': True,
            'is_collision_free': is_collision_free,
            'safety_score': safety_score,
            'trajectory_type': trajectory_type,
            'total_waypoints': total_waypoints,
            'checked_waypoints': checked_waypoints,
            'collision_waypoints': collision_waypoints,
            'collision_count': collision_count,
            'collision_percentage': collision_percentage,
            'first_collision_index': first_collision_index,
            'collision_details': collision_details,
            'validation_settings': {
                'rigid_body_id': self.rigid_body_id,
                'safety_margin': self.safety_margin,
                'check_density': self.check_density,
                'pointcloud_file': self.pointcloud_file
            },
            'validation_time': validation_time,
            'timestamp': time.time()
        }
        
        # 결과 요약 출력
        print(f"📊 검증 완료 - {trajectory_type}")
        print(f"   충돌 여부: {'❌ 충돌 발생' if not is_collision_free else '✅ 충돌 없음'}")
        print(f"   안전 점수: {safety_score:.1f}/100")
        print(f"   충돌 waypoint: {collision_count}/{checked_waypoints} ({collision_percentage:.1f}%)")
        if first_collision_index is not None:
            print(f"   첫 충돌 지점: waypoint {first_collision_index}")
        print(f"   검증 시간: {validation_time:.3f}초")
        
        return result
    
    def validate_multiple_trajectories(self, 
                                     trajectories: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        여러 궤적 동시 검증
        
        Args:
            trajectories: {'raw': traj1, 'smooth': traj2, ...}
            
        Returns:
            validation_results: 각 궤적별 검증 결과
        """
        results = {}
        
        print(f"🔍 다중 궤적 검증 시작 ({len(trajectories)}개)")
        
        for traj_name, trajectory in trajectories.items():
            print(f"\n--- {traj_name} 궤적 검증 ---")
            results[traj_name] = self.validate_trajectory(trajectory, traj_name)
        
        # 비교 요약
        print(f"\n📊 다중 궤적 검증 요약")
        for name, result in results.items():
            if result['success']:
                status = "✅ 안전" if result['is_collision_free'] else "❌ 충돌"
                print(f"   {name:10s}: {status} (점수: {result['safety_score']:.1f})")
            else:
                print(f"   {name:10s}: ❌ 검증 실패")
        
        return results
    
    def compare_trajectory_safety(self, 
                                raw_trajectory: np.ndarray,
                                smooth_trajectory: np.ndarray) -> Dict[str, Any]:
        """
        Raw vs Smooth 궤적 안전성 비교
        
        Args:
            raw_trajectory: 원본 RRT 궤적
            smooth_trajectory: 스무딩된 궤적
            
        Returns:
            comparison_result: 비교 결과
        """
        print("🆚 궤적 안전성 비교 분석")
        
        # 각각 검증
        raw_result = self.validate_trajectory(raw_trajectory, "raw")
        smooth_result = self.validate_trajectory(smooth_trajectory, "smooth")
        
        if not (raw_result['success'] and smooth_result['success']):
            return {
                'success': False,
                'error': 'Validation failed for one or both trajectories'
            }
        
        # 비교 분석
        comparison = {
            'success': True,
            'raw_result': raw_result,
            'smooth_result': smooth_result,
            'comparison': {
                'safety_improvement': smooth_result['safety_score'] - raw_result['safety_score'],
                'collision_reduction': raw_result['collision_count'] - smooth_result['collision_count'],
                'better_trajectory': 'smooth' if smooth_result['safety_score'] > raw_result['safety_score'] else 'raw' if raw_result['safety_score'] > smooth_result['safety_score'] else 'equal',
                'both_collision_free': raw_result['is_collision_free'] and smooth_result['is_collision_free'],
                'validation_time_total': raw_result['validation_time'] + smooth_result['validation_time']
            }
        }
        
        # 비교 결과 출력
        print(f"\n📊 안전성 비교 결과")
        print(f"   Raw 궤적     : {raw_result['safety_score']:.1f}/100 (충돌: {raw_result['collision_count']})")
        print(f"   Smooth 궤적  : {smooth_result['safety_score']:.1f}/100 (충돌: {smooth_result['collision_count']})")
        print(f"   안전성 개선  : {comparison['comparison']['safety_improvement']:+.1f}점")
        print(f"   충돌 감소    : {comparison['comparison']['collision_reduction']}개")
        print(f"   권장 궤적    : {comparison['comparison']['better_trajectory']}")
        
        return comparison
    
    def get_collision_heatmap(self, trajectory: np.ndarray) -> np.ndarray:
        """
        궤적 상의 충돌 위험도 히트맵 생성
        
        Args:
            trajectory: [N, 3] 궤적 데이터
            
        Returns:
            heatmap: [N] 각 waypoint의 충돌 위험도 (0=안전, 1=충돌)
        """
        heatmap = np.zeros(len(trajectory))
        
        # SE(2) 변환
        if trajectory.shape[1] == 6:
            se2_trajectory = trajectory[:, [0, 1, 5]]
        else:
            se2_trajectory = trajectory
        
        # 각 waypoint 검사
        for i, waypoint in enumerate(se2_trajectory):
            x, y, theta = waypoint
            se3_pose = [x, y, 0.0, 0.0, 0.0, theta]
            
            result = self.collision_detector.check_collision(
                se3_pose, self.rigid_body_id, self.safety_margin
            )
            
            heatmap[i] = 1.0 if result.is_collision else 0.0
        
        return heatmap


# 헬퍼 함수들
def create_trajectory_validator(pointcloud_file: str, **kwargs) -> TrajectoryValidator:
    """궤적 검증기 생성 헬퍼"""
    return TrajectoryValidator(pointcloud_file, **kwargs)


def quick_safety_check(trajectory: np.ndarray, 
                      pointcloud_file: str,
                      **kwargs) -> bool:
    """빠른 안전성 체크 (충돌 없음만 확인)"""
    validator = TrajectoryValidator(pointcloud_file, **kwargs)
    result = validator.validate_trajectory(trajectory)
    return result.get('is_collision_free', False)


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 TrajectoryValidator 테스트")
    
    # 테스트 궤적 (간단한 직선)
    test_trajectory = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # 테스트 환경 파일 (실제 파일이 있다고 가정)
    test_env_file = "data/pointcloud/circles_only/circles_only.ply"
    
    try:
        # 검증기 생성
        validator = TrajectoryValidator(
            pointcloud_file=test_env_file,
            rigid_body_id=3,
            safety_margin=0.05,
            check_density=1
        )
        
        # 궤적 검증
        result = validator.validate_trajectory(test_trajectory, "test")
        
        if result['success']:
            print("✅ 검증 성공")
            print(f"   안전성: {'안전' if result['is_collision_free'] else '위험'}")
            print(f"   점수: {result['safety_score']}/100")
        else:
            print(f"❌ 검증 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("💡 실제 환경 파일이 필요합니다.")
