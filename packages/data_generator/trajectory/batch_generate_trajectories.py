#!/usr/bin/env python3
"""
배치 궤적 생성 스크립트
HDF5 기반 환경별 궤적 데이터 배치 생성 시스템

주요 기능:
- 환경별 궤적 배치 생성
- RRT → B-spline 파이프라인 자동화
- 충돌 검증 시스템 연동
- 멀티프로세싱 지원
"""

import sys
import os
import argparse
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 모듈 import
from trajectory_data_manager import (
    TrajectoryDataManager, 
    PosePairMetadata, 
    create_environment_info, 
    create_generation_config
)
from trajectory_validator import TrajectoryValidator
from utils.trajectory_smoother import BSplineTrajectoryProcessor

# 복사된 모듈 import (로컬 경로 사용)
try:
    # trajectory 하위에 복사된 모듈들 사용
    from rrt_connect import create_se3_planner, SE3TrajectoryResult
    
    # pose 관련 함수들 - 기존 경로에서 import (아직 복사 안됨)
    sys.path.append(str(project_root / "packages" / "data_generator"))
    from pose.random_pose_generator import generate_collision_free_poses
    
    print("✅ 로컬 RRT 모듈 import 성공")
except ImportError as e:
    print(f"⚠️ 모듈 import 실패: {e}")
    generate_collision_free_poses = None
    create_se3_planner = None


class TrajectoryBatchGenerator:
    """배치 궤적 생성기"""
    
    def __init__(self, 
                 env_name: str,
                 pointcloud_file: str,
                 rigid_body_id: int = 3,
                 safety_margin: float = 0.05,
                 rrt_config: Optional[Dict[str, Any]] = None,
                 bspline_config: Optional[Dict[str, Any]] = None,
                 validation_enabled: bool = True):
        """
        Args:
            env_name: 환경 이름
            pointcloud_file: 환경 PLY 파일 경로
            rigid_body_id: Rigid body ID
            safety_margin: 안전 여유 거리
            rrt_config: RRT 설정
            bspline_config: B-spline 설정
            validation_enabled: 충돌 검증 활성화
        """
        self.env_name = env_name
        self.pointcloud_file = pointcloud_file
        self.rigid_body_id = rigid_body_id
        self.safety_margin = safety_margin
        self.validation_enabled = validation_enabled
        
        # 기본 설정
        self.rrt_config = rrt_config or {
            'range': 0.5,
            'max_planning_time': 5.0,
            'interpolate': True,
            'simplify': True
        }
        
        self.bspline_config = bspline_config or {
            'degree': 3,
            'smoothing_factor': 0.0,
            'density_multiplier': 2
        }
        
        # 데이터 매니저 초기화
        self.data_manager = TrajectoryDataManager(env_name)
        
        # RRT 플래너 초기화
        try:
            self.rrt_planner = create_se3_planner(rigid_body_id, pointcloud_file)
            print(f"✅ RRT 플래너 초기화 완료")
        except Exception as e:
            print(f"❌ RRT 플래너 초기화 실패: {e}")
            self.rrt_planner = None
        
        # B-spline 프로세서 초기화
        self.bspline_processor = BSplineTrajectoryProcessor(
            degree=self.bspline_config['degree'],
            smoothing_factor=self.bspline_config['smoothing_factor']
        )
        
        # 검증기 초기화 (선택적)
        self.validator = None
        if validation_enabled:
            try:
                self.validator = TrajectoryValidator(
                    pointcloud_file=pointcloud_file,
                    rigid_body_id=rigid_body_id,
                    safety_margin=safety_margin
                )
                print(f"✅ 궤적 검증기 초기화 완료")
            except Exception as e:
                print(f"⚠️ 궤적 검증기 초기화 실패: {e}")
                self.validation_enabled = False
        
        # 통계
        self.stats = {
            'total_attempts': 0,
            'successful_rrt': 0,
            'successful_smooth': 0,
            'collision_free': 0,
            'total_time': 0.0
        }
    
    def initialize_data_manager(self, workspace_bounds: List[float]) -> bool:
        """데이터 매니저 HDF5 파일 초기화"""
        env_info = create_environment_info(
            env_name=self.env_name,
            env_type="auto_detected",
            pointcloud_file=self.pointcloud_file,
            workspace_bounds=workspace_bounds
        )
        
        gen_config = create_generation_config(
            rigid_body_id=self.rigid_body_id,
            safety_margin=self.safety_margin,
            rrt_range=self.rrt_config['range'],
            rrt_max_time=self.rrt_config['max_planning_time'],
            bspline_degree=self.bspline_config['degree'],
            bspline_smoothing=self.bspline_config['smoothing_factor'],
            validation_enabled=self.validation_enabled
        )
        
        return self.data_manager.initialize_h5_file(env_info, gen_config)
    
    def generate_pose_pairs(self, count: int) -> List[Tuple[List[float], List[float]]]:
        """충돌 없는 pose pair 생성"""
        print(f"🎯 Pose pair 생성 중... (목표: {count}개)")
        
        try:
            # 충돌 없는 pose 생성
            poses = generate_collision_free_poses(
                environment_file=self.pointcloud_file,
                robot_geometry=self.rigid_body_id,
                num_poses=count * 3,  # 여유분을 두고 생성
                workspace_bounds=(-5, 5, -5, 5),  # 기본 workspace
                max_attempts=count * 10
            )
            
            if len(poses) < count * 2:
                print(f"⚠️ 충분한 pose를 생성하지 못했습니다: {len(poses)} < {count * 2}")
                return []
            
            # pose pair 생성
            pose_pairs = []
            for i in range(0, min(len(poses) - 1, count * 2), 2):
                start_pose = poses[i][:3]  # [x, y, theta]
                end_pose = poses[i + 1][:3]
                pose_pairs.append((start_pose, end_pose))
            
            print(f"✅ Pose pair 생성 완료: {len(pose_pairs)}개")
            return pose_pairs[:count]
            
        except Exception as e:
            print(f"❌ Pose pair 생성 실패: {e}")
            return []
    
    def generate_single_trajectory(self, 
                                 start_pose: List[float], 
                                 end_pose: List[float],
                                 pair_id: str) -> bool:
        """단일 궤적 생성 (RRT → B-spline → 검증)"""
        start_time = time.time()
        
        try:
            # 1. RRT 궤적 계획
            print(f"🛤️ RRT 궤적 계획 중... ({pair_id})")
            
            # SE(2) → SE(3) 변환
            start_se3 = [start_pose[0], start_pose[1], 0.0, 0.0, 0.0, start_pose[2]]
            end_se3 = [end_pose[0], end_pose[1], 0.0, 0.0, 0.0, end_pose[2]]
            
            # RRT 계획
            rrt_result = self.rrt_planner.plan_trajectory(
                start_se3, end_se3, 
                max_planning_time=self.rrt_config['max_planning_time']
            )
            
            if not rrt_result.success:
                print(f"❌ RRT 실패: {pair_id}")
                return False
            
            self.stats['successful_rrt'] += 1
            
            # SE(3) → SE(2) 변환
            raw_trajectory = np.array(rrt_result.trajectory)
            raw_se2 = raw_trajectory[:, [0, 1, 5]]  # [x, y, rz]
            
            rrt_time = time.time() - start_time
            
            # 2. B-spline 스무딩
            print(f"🌊 B-spline 스무딩 중... ({pair_id})")
            smooth_start_time = time.time()
            
            num_points = int(len(raw_se2) * self.bspline_config['density_multiplier'])
            smooth_trajectory, smooth_info = self.bspline_processor.smooth_trajectory(
                raw_se2, num_points=num_points
            )
            
            smooth_time = time.time() - smooth_start_time
            
            if not smooth_info['success']:
                print(f"❌ 스무딩 실패: {pair_id} - {smooth_info['error']}")
                smooth_trajectory = raw_se2  # 원본 사용
                smooth_time = 0.0
            else:
                self.stats['successful_smooth'] += 1
            
            # 3. 충돌 검증 (선택적)
            validation_results = None
            validation_time = 0.0
            is_collision_free = True
            
            if self.validation_enabled and self.validator is not None:
                print(f"🔍 충돌 검증 중... ({pair_id})")
                validation_start_time = time.time()
                
                # Raw와 Smooth 궤적 모두 검증
                validation_results = self.validator.compare_trajectory_safety(
                    raw_se2, smooth_trajectory
                )
                
                validation_time = time.time() - validation_start_time
                
                if validation_results['success']:
                    is_collision_free = validation_results['smooth_result']['is_collision_free']
                    if is_collision_free:
                        self.stats['collision_free'] += 1
                
            # 4. 메타데이터 생성
            metadata = PosePairMetadata(
                start_pose=start_pose,
                end_pose=end_pose,
                generation_method="rrt_connect",
                smoothing_method="bspline" if smooth_info['success'] else "none",
                collision_free=is_collision_free,
                path_length=float(np.sum(np.linalg.norm(
                    np.diff(smooth_trajectory[:, :2], axis=0), axis=1
                ))),
                generation_time=rrt_time,
                smoothing_time=smooth_time,
                validation_time=validation_time
            )
            
            # 5. HDF5에 저장
            success = self.data_manager.add_pose_pair(
                pair_id=pair_id,
                metadata=metadata,
                raw_trajectory=raw_se2,
                smooth_trajectory=smooth_trajectory,
                validation_results=validation_results
            )
            
            total_time = time.time() - start_time
            self.stats['total_time'] += total_time
            
            if success:
                print(f"✅ 궤적 생성 완료: {pair_id} ({total_time:.2f}초)")
                return True
            else:
                print(f"❌ 저장 실패: {pair_id}")
                return False
                
        except Exception as e:
            print(f"❌ 궤적 생성 오류 ({pair_id}): {e}")
            return False
    
    def generate_batch(self, 
                      pair_count: int,
                      use_existing_poses: bool = False,
                      existing_poses_file: Optional[str] = None) -> Dict[str, Any]:
        """배치 궤적 생성"""
        print(f"🚀 배치 궤적 생성 시작")
        print(f"   환경: {self.env_name}")
        print(f"   목표 궤적 수: {pair_count}")
        print(f"   기존 pose 사용: {use_existing_poses}")
        
        batch_start_time = time.time()
        
        # 1. HDF5 파일 초기화
        if not self.initialize_data_manager([-5.0, 5.0, -5.0, 5.0]):
            return {'success': False, 'error': 'Failed to initialize data manager'}
        
        # 2. Pose pair 준비
        if use_existing_poses and existing_poses_file:
            print(f"📁 기존 pose 파일 로드: {existing_poses_file}")
            # TODO: 기존 pose 파일 로드 로직 구현
            pose_pairs = self.generate_pose_pairs(pair_count)
        else:
            pose_pairs = self.generate_pose_pairs(pair_count)
        
        if not pose_pairs:
            return {'success': False, 'error': 'Failed to generate pose pairs'}
        
        # 3. 궤적 생성 루프
        successful_count = 0
        
        for i, (start_pose, end_pose) in enumerate(pose_pairs):
            pair_id = f"pair_{i+1:06d}"
            
            print(f"\n--- 궤적 {i+1}/{len(pose_pairs)} ---")
            
            self.stats['total_attempts'] += 1
            
            success = self.generate_single_trajectory(start_pose, end_pose, pair_id)
            if success:
                successful_count += 1
            
            # 진행상황 출력
            if (i + 1) % 10 == 0:
                success_rate = (successful_count / (i + 1)) * 100
                avg_time = self.stats['total_time'] / (i + 1)
                print(f"\n📊 중간 통계 ({i+1}/{len(pose_pairs)})")
                print(f"   성공률: {success_rate:.1f}%")
                print(f"   평균 시간: {avg_time:.2f}초/궤적")
        
        # 4. 최종 결과
        batch_time = time.time() - batch_start_time
        
        final_stats = self.data_manager.get_summary_stats()
        
        result = {
            'success': True,
            'env_name': self.env_name,
            'total_attempts': self.stats['total_attempts'],
            'successful_trajectories': successful_count,
            'success_rate': (successful_count / self.stats['total_attempts']) * 100,
            'batch_time': batch_time,
            'avg_time_per_trajectory': batch_time / self.stats['total_attempts'],
            'rrt_success_rate': (self.stats['successful_rrt'] / self.stats['total_attempts']) * 100,
            'smooth_success_rate': (self.stats['successful_smooth'] / self.stats['total_attempts']) * 100,
            'collision_free_rate': (self.stats['collision_free'] / self.stats['total_attempts']) * 100 if self.validation_enabled else None,
            'h5_file_path': str(self.data_manager.h5_file_path),
            'final_stats': final_stats
        }
        
        print(f"\n🎉 배치 생성 완료!")
        print(f"   성공한 궤적: {successful_count}/{self.stats['total_attempts']}")
        print(f"   성공률: {result['success_rate']:.1f}%")
        print(f"   총 소요시간: {batch_time:.1f}초")
        print(f"   HDF5 파일: {self.data_manager.h5_file_path}")
        
        return result


def generate_trajectories_for_environment(env_name: str,
                                        pointcloud_file: str,
                                        pair_count: int = 100,
                                        **kwargs) -> Dict[str, Any]:
    """환경별 궤적 생성 메인 함수"""
    
    generator = TrajectoryBatchGenerator(
        env_name=env_name,
        pointcloud_file=pointcloud_file,
        **kwargs
    )
    
    return generator.generate_batch(pair_count)


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(description="HDF5 기반 배치 궤적 생성")
    
    parser.add_argument('--env-name', required=True, 
                       help='환경 이름 (예: circle_env_000001)')
    parser.add_argument('--pointcloud-file', required=True,
                       help='환경 PLY 파일 경로')
    parser.add_argument('--pair-count', type=int, default=100,
                       help='생성할 궤적 쌍 개수 (기본값: 100)')
    parser.add_argument('--rigid-body-id', type=int, default=3,
                       help='Rigid body ID (기본값: 3)')
    parser.add_argument('--safety-margin', type=float, default=0.05,
                       help='안전 여유 거리 (기본값: 0.05)')
    parser.add_argument('--use-existing-poses', action='store_true',
                       help='기존 pose 파일 사용')
    parser.add_argument('--existing-poses-file', 
                       help='기존 pose 파일 경로')
    parser.add_argument('--no-collision-check', action='store_true',
                       help='충돌 검증 비활성화')
    parser.add_argument('--rrt-range', type=float, default=0.5,
                       help='RRT range 설정')
    parser.add_argument('--rrt-max-time', type=float, default=5.0,
                       help='RRT 최대 계획 시간')
    parser.add_argument('--bspline-degree', type=int, default=3,
                       help='B-spline 차수')
    parser.add_argument('--output-json', action='store_true',
                       help='JSON 형태로도 내보내기')
    
    args = parser.parse_args()
    
    # 환경 파일 존재 확인
    if not Path(args.pointcloud_file).exists():
        print(f"❌ 환경 파일을 찾을 수 없습니다: {args.pointcloud_file}")
        return 1
    
    # RRT 설정
    rrt_config = {
        'range': args.rrt_range,
        'max_planning_time': args.rrt_max_time,
        'interpolate': True,
        'simplify': True
    }
    
    # B-spline 설정
    bspline_config = {
        'degree': args.bspline_degree,
        'smoothing_factor': 0.0,
        'density_multiplier': 2
    }
    
    # 궤적 생성 실행
    try:
        result = generate_trajectories_for_environment(
            env_name=args.env_name,
            pointcloud_file=args.pointcloud_file,
            pair_count=args.pair_count,
            rigid_body_id=args.rigid_body_id,
            safety_margin=args.safety_margin,
            rrt_config=rrt_config,
            bspline_config=bspline_config,
            validation_enabled=not args.no_collision_check
        )
        
        if result['success']:
            print(f"\n✅ 궤적 생성 성공!")
            
            # JSON 내보내기 (선택적)
            if args.output_json:
                generator = TrajectoryBatchGenerator(args.env_name, args.pointcloud_file)
                generator.data_manager = TrajectoryDataManager(args.env_name)
                success = generator.data_manager.export_to_json()
                if success:
                    print(f"📁 JSON 파일 내보내기 완료")
            
            return 0
        else:
            print(f"❌ 궤적 생성 실패: {result.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 중단됨")
        return 130
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
