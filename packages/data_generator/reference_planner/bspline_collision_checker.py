#!/usr/bin/env python3
"""
B-spline 스무딩된 궤적의 collision checking
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# collision detector import
sys.path.append(str(Path(__file__).parent.parent))
from pose.collision_detector import RigidBodyCollisionDetector


def check_bsplined_trajectory_collision(
    trajectory_file: str,
    pointcloud_file: str,
    rigid_body_id: int = 3,
    safety_margin: float = 0.05,
    check_density: int = 1
) -> Dict[str, Any]:
    """
    B-spline 스무딩된 궤적의 collision checking
    
    Args:
        trajectory_file: B-spline 궤적 JSON 파일 경로
        pointcloud_file: 환경 PLY 파일 경로
        rigid_body_id: Rigid body ID (기본값: 3)
        safety_margin: 안전 여유 거리 (기본값: 0.05m)
        check_density: 체크 밀도 (1=모든점, 2=2개마다1개, ...)
        
    Returns:
        collision_result: {
            'is_collision_free': bool,
            'total_waypoints': int,
            'checked_waypoints': int,
            'collision_waypoints': List[int],
            'collision_count': int,
            'collision_percentage': float,
            'first_collision_index': int or None,
            'safety_margin_used': float
        }
    """
    
    # 궤적 데이터 로드
    with open(trajectory_file, 'r') as f:
        traj_data = json.load(f)
    
    path_data = np.array(traj_data['path']['data'])
    total_waypoints = len(path_data)
    
    # collision detector 초기화
    collision_detector = RigidBodyCollisionDetector()
    
    # 환경 로드
    try:
        collision_detector.load_environment(pointcloud_file)
        print(f"✅ Environment loaded: {Path(pointcloud_file).name}")
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return {
            'is_collision_free': False,
            'error': f'Environment loading failed: {e}',
            'total_waypoints': total_waypoints
        }
    
    # 선택적 웨이포인트만 체크 (밀도 조절)
    check_indices = list(range(0, total_waypoints, check_density))
    checked_waypoints = len(check_indices)
    
    collision_waypoints = []
    first_collision_index = None
    
    print(f"🔍 Checking {checked_waypoints}/{total_waypoints} waypoints (density: 1/{check_density})")
    print(f"   Safety margin: {safety_margin}m")
    
    # 각 웨이포인트에 대해 collision check
    for i, idx in enumerate(check_indices):
        pose = path_data[idx].tolist()  # [x, y, z, rx, ry, rz]
        
        # collision checking
        result = collision_detector.check_collision(pose, rigid_body_id, safety_margin)
        
        if result.is_collision:
            collision_waypoints.append(idx)
            if first_collision_index is None:
                first_collision_index = idx
        
        # 진행 상황 출력 (10개마다)
        if (i + 1) % 10 == 0 or (i + 1) == checked_waypoints:
            collision_count = len(collision_waypoints)
            print(f"   Progress: {i+1}/{checked_waypoints}, Collisions: {collision_count}")
    
    collision_count = len(collision_waypoints)
    collision_percentage = (collision_count / checked_waypoints) * 100
    is_collision_free = collision_count == 0
    
    # 결과 반환
    result = {
        'is_collision_free': is_collision_free,
        'total_waypoints': total_waypoints,
        'checked_waypoints': checked_waypoints,
        'collision_waypoints': collision_waypoints,
        'collision_count': collision_count,
        'collision_percentage': collision_percentage,
        'first_collision_index': first_collision_index,
        'safety_margin_used': safety_margin,
        'check_density': check_density,
        'trajectory_file': str(trajectory_file),
        'environment_file': str(pointcloud_file)
    }
    
    return result


def print_collision_result(result: Dict[str, Any]) -> None:
    """Collision checking 결과를 보기 좋게 출력"""
    
    print(f"\n🎯 Collision Checking Results:")
    print(f"   Trajectory: {Path(result['trajectory_file']).name}")
    print(f"   Environment: {Path(result['environment_file']).name}")
    print(f"   Total waypoints: {result['total_waypoints']}")
    print(f"   Checked waypoints: {result['checked_waypoints']} (density: 1/{result['check_density']})")
    print(f"   Safety margin: {result['safety_margin_used']}m")
    
    if result['is_collision_free']:
        print(f"   ✅ COLLISION-FREE! No collisions detected.")
    else:
        print(f"   ❌ COLLISION DETECTED!")
        print(f"      Collision count: {result['collision_count']}")
        print(f"      Collision percentage: {result['collision_percentage']:.1f}%")
        print(f"      First collision at waypoint: {result['first_collision_index']}")
        
        if len(result['collision_waypoints']) <= 10:
            print(f"      Collision waypoints: {result['collision_waypoints']}")
        else:
            print(f"      Collision waypoints: {result['collision_waypoints'][:5]}...{result['collision_waypoints'][-5:]} (showing first/last 5)")


def main():
    """테스트용 메인 함수"""
    
    if len(sys.argv) < 3:
        print("Usage: python bspline_collision_checker.py <trajectory_file> <pointcloud_file> [rigid_body_id] [safety_margin] [check_density]")
        print("Example: python bspline_collision_checker.py trajectory.json environment.ply 3 0.05 1")
        sys.exit(1)
    
    trajectory_file = sys.argv[1]
    pointcloud_file = sys.argv[2]
    rigid_body_id = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    safety_margin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    check_density = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    
    # Collision checking 실행
    result = check_bsplined_trajectory_collision(
        trajectory_file=trajectory_file,
        pointcloud_file=pointcloud_file,
        rigid_body_id=rigid_body_id,
        safety_margin=safety_margin,
        check_density=check_density
    )
    
    # 결과 출력
    print_collision_result(result)
    
    # JSON 결과 저장
    output_file = f"{Path(trajectory_file).stem}_collision_check.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Detailed results saved: {output_file}")


if __name__ == '__main__':
    main()