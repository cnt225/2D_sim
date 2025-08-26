#!/usr/bin/env python3
"""
Raw Trajectory Batch Generator
환경 묶음별로 체계적인 RRT 궤적 대량 생성

사용법:
    python batch_generate_raw_trajectories.py --env-set circles_only \
        --pose-file circles_only_poses.h5 --env-count 3 --pair-count 5
        
구조:
    root/data/trajectory/circles_only_trajs.h5
    ├── circle_env_000000/
    │   ├── 0/ (pose pair index)
    │   │   ├── raw_trajectory
    │   │   └── metadata
    │   └── 1/
    ├── circle_env_000001/
    └── circle_env_000002/
"""

import os
import sys
import argparse
import time
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 필수 모듈 import (실패 시 명확한 오류)
from rrt_connect import create_se3_planner, SE3TrajectoryResult

# pose 데이터 로드를 위한 모듈 (필수)
sys.path.append(str(project_root / "packages" / "data_generator" / "pose"))
from unified_pose_manager import UnifiedPoseManager

print("✅ 필수 모듈 import 완료")

class RawTrajectoryBatchGenerator:
    """환경 묶음별 Raw 궤적 대량 생성기"""
    
    def __init__(self, 
                 env_set_name: str,
                 pose_file: str,
                 rrt_config: Dict[str, Any] = None,
                 validation_enabled: bool = False):
        """
        초기화
        
        Args:
            env_set_name: 환경 묶음 이름 (예: 'circles_only')
            pose_file: Pose 데이터 HDF5 파일명 (root/data/pose/ 기준, 필수)
            rrt_config: RRT 설정
            validation_enabled: 충돌 검증 사용 여부
        """
        self.env_set_name = env_set_name
        self.validation_enabled = validation_enabled
        
        # Pose 파일 경로 (필수, 기본값 없음)
        self.pose_file = str(project_root / "data" / "pose" / pose_file)
        if not Path(self.pose_file).exists():
            raise FileNotFoundError(f"❌ Pose 파일을 찾을 수 없습니다: {self.pose_file}")
        
        # RRT 설정
        self.rrt_config = rrt_config or {
            'rigid_body_id': 3,
            'max_planning_time': 15.0,  # 복잡한 케이스도 안정적으로 해결하기 위해 15초로 설정
            'range': 0.25
        }
        
        print(f"🏗️ RawTrajectoryBatchGenerator 초기화:")
        print(f"   환경 묶음: {env_set_name}")
        print(f"   Pose 파일: {self.pose_file}")
        print(f"   RRT Range: {self.rrt_config['range']}")
        print(f"   검증 활성화: {validation_enabled}")
        
        # Pose 매니저 초기화
        try:
            self.pose_manager = UnifiedPoseManager(self.pose_file)
            print(f"✅ Pose 매니저 초기화 완료")
        except Exception as e:
            raise RuntimeError(f"❌ Pose 매니저 초기화 실패: {e}")
        
        # Trajectory 저장소 초기화 (env_set_name_trajs.h5)
        trajectory_dir = project_root / "data" / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_file = trajectory_dir / f"{env_set_name}_trajs.h5"
        
        # 통계 초기화
        self.stats = {
            'total_environments': 0,
            'total_pairs': 0,
            'successful_pairs': 0,
            'total_time': 0.0
        }
        
        # 사용 가능한 환경 목록 로드
        self._load_available_environments()
        
        print(f"✅ 초기화 완료")
        print(f"   궤적 저장소: {self.trajectory_file}")
        print(f"   사용 가능한 환경: {len(self.available_environments)}개")
    
    def _load_available_environments(self):
        """Pose 파일에서 사용 가능한 환경 목록 로드"""
        try:
            import h5py
            self.available_environments = []
            
            with h5py.File(self.pose_file, 'r') as f:
                if 'environments' in f:
                    for env_name in f['environments'].keys():
                        # RB 3의 pose pairs가 있는 환경만 포함
                        env_path = f'environments/{env_name}/pose_pairs/rb_{self.rrt_config["rigid_body_id"]}'
                        if env_path in f:
                            self.available_environments.append(env_name)
            
            self.available_environments.sort()  # 정렬
            print(f"✅ 환경 목록 로드 완료: {len(self.available_environments)}개")
            
            if len(self.available_environments) == 0:
                raise ValueError(f"RB {self.rrt_config['rigid_body_id']}에 대한 환경이 없습니다")
                
        except Exception as e:
            raise RuntimeError(f"❌ 환경 목록 로드 실패: {e}")
    
    def generate_batch(self, env_count: int, pair_count: int, start_env_id: int = 0) -> Dict[str, Any]:
        """
        환경 묶음별 배치 궤적 생성
        
        Args:
            env_count: 생성할 환경 수 (사용 가능한 환경 목록에서 선택)
            pair_count: 각 환경당 생성할 pose pair 수
            start_env_id: 시작 환경 인덱스 (available_environments 리스트의 인덱스)
            
        Returns:
            결과 통계
        """
        print(f"\n🚀 배치 궤적 생성 시작")
        print(f"   환경 묶음: {self.env_set_name}")
        print(f"   사용 가능한 환경: {len(self.available_environments)}개")
        print(f"   처리할 환경 수: {env_count}")
        print(f"   시작 인덱스: {start_env_id}")
        print(f"   각 환경당 pair 수: {pair_count}")
        
        # 인덱스 범위 검증
        if start_env_id >= len(self.available_environments):
            raise ValueError(f"시작 인덱스 {start_env_id}가 사용 가능한 환경 수 {len(self.available_environments)}를 초과합니다")
        
        if start_env_id + env_count > len(self.available_environments):
            print(f"⚠️ 요청된 환경 수가 사용 가능한 환경을 초과합니다. 조정: {env_count} → {len(self.available_environments) - start_env_id}")
            env_count = len(self.available_environments) - start_env_id
        
        batch_start_time = time.time()
        successful_pairs = 0
        
        # HDF5 파일 초기화
        self._initialize_trajectory_file()
        
        # 실제 환경 이름으로 처리
        for env_idx in range(env_count):
            actual_env_idx = start_env_id + env_idx
            env_name = self.available_environments[actual_env_idx]
            
            print(f"\n📁 환경 처리 중: {env_name} ({env_idx + 1}/{env_count})")
            print(f"   환경 인덱스: {actual_env_idx}/{len(self.available_environments)}")
            
            # 환경별 궤적 생성
            env_success = self._generate_environment_trajectories(env_name, pair_count)
            successful_pairs += env_success
            
            self.stats['total_environments'] += 1
            self.stats['total_pairs'] += pair_count
        
        batch_time = time.time() - batch_start_time
        self.stats['total_time'] = batch_time
        self.stats['successful_pairs'] = successful_pairs
        
        # 최종 통계
        final_stats = {
            'env_set_name': self.env_set_name,
            'total_environments': self.stats['total_environments'],
            'total_pairs': self.stats['total_pairs'],
            'successful_pairs': successful_pairs,
            'success_rate': (successful_pairs / self.stats['total_pairs']) * 100,
            'total_time': batch_time,
            'avg_time_per_pair': batch_time / self.stats['total_pairs'],
            'trajectory_file': str(self.trajectory_file)
        }
        
        return final_stats
    
    def _initialize_trajectory_file(self):
        """궤적 저장용 HDF5 파일 초기화"""
        try:
            with h5py.File(self.trajectory_file, 'w') as f:
                # 전역 메타데이터 그룹
                metadata = f.create_group('metadata')
                metadata.attrs['env_set_name'] = self.env_set_name
                metadata.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                metadata.attrs['rrt_range'] = self.rrt_config['range']
                metadata.attrs['rigid_body_id'] = self.rrt_config['rigid_body_id']
                
            print(f"✅ 궤적 파일 초기화 완료: {self.trajectory_file}")
        except Exception as e:
            raise RuntimeError(f"❌ 궤적 파일 초기화 실패: {e}")
    
    def _generate_environment_trajectories(self, env_name: str, pair_count: int) -> int:
        """단일 환경의 궤적 생성"""
        
        # 1. Pose pairs 로드
        pose_pairs, metadata = self.pose_manager.get_pose_pairs(
            env_name, self.rrt_config['rigid_body_id']
        )
        
        if pose_pairs is None or len(pose_pairs) == 0:
            print(f"⚠️ 환경 '{env_name}' RB {self.rrt_config['rigid_body_id']}의 pose pair가 없습니다")
            return 0
        
        print(f"✅ Pose pairs 로드: {len(pose_pairs)}개")
        
        # 2. 환경 파일 경로 구성
        pointcloud_file = project_root / "data" / "pointcloud" / self.env_set_name / f"{env_name}.ply"
        if not pointcloud_file.exists():
            print(f"⚠️ 환경 파일 없음: {pointcloud_file}")
            return 0
        
        # 3. RRT 플래너 초기화
        try:
            planner = create_se3_planner(self.rrt_config['rigid_body_id'], str(pointcloud_file))
            planner.planner_settings['range'] = self.rrt_config['range']
            print(f"✅ RRT 플래너 초기화 완료 (Range: {self.rrt_config['range']})")
        except Exception as e:
            print(f"❌ RRT 플래너 초기화 실패: {e}")
            return 0
        
        # 4. 각 pose pair에 대해 궤적 생성
        successful_count = 0
        
        for pair_idx in range(min(pair_count, len(pose_pairs))):
            pose_pair = pose_pairs[pair_idx]  # (12,) [init_pose(6), target_pose(6)]
            
            # SE(3) → SE(2) 변환 (x, y, yaw만 사용)
            start_pose = np.array([pose_pair[0], pose_pair[1], pose_pair[5]])  # x, y, yaw
            end_pose = np.array([pose_pair[6], pose_pair[7], pose_pair[11]])   # x, y, yaw
            
            print(f"🛤️ Pair {pair_idx}: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {start_pose[2]:.2f}] → [{end_pose[0]:.2f}, {end_pose[1]:.2f}, {end_pose[2]:.2f}]")
            
            # RRT 계획
            success = self._generate_single_trajectory(planner, env_name, pair_idx, start_pose, end_pose)
            if success:
                successful_count += 1
        
        print(f"📊 환경 '{env_name}' 완료: {successful_count}/{pair_count} 성공")
        return successful_count
    
    def _generate_single_trajectory(self, planner, env_name: str, pair_idx: int, 
                                  start_pose: np.ndarray, end_pose: np.ndarray) -> bool:
        """단일 궤적 생성 및 저장"""
        
        try:
            # SE(2) → SE(3) 변환
            start_se3 = [start_pose[0], start_pose[1], 0.0, 0.0, 0.0, start_pose[2]]
            end_se3 = [end_pose[0], end_pose[1], 0.0, 0.0, 0.0, end_pose[2]]
            
            # RRT 계획
            rrt_start_time = time.time()
            result = planner.plan_trajectory(start_se3, end_se3, self.rrt_config['max_planning_time'])
            rrt_time = time.time() - rrt_start_time
            
            if not result.success:
                print(f"❌ RRT 계획 실패")
                return False
            
            # SE(3) → SE(2) 변환
            raw_se2 = np.array([[p[0], p[1], p[5]] for p in result.trajectory])
            
            print(f"✅ RRT 성공: {len(raw_se2)}개 점, {rrt_time:.3f}초")
            
            # HDF5 저장
            self._save_trajectory_to_h5(env_name, pair_idx, raw_se2, {
                'start_pose': start_pose.tolist(),
                'end_pose': end_pose.tolist(),
                'generation_time': rrt_time,
                'path_length': float(np.sum(np.linalg.norm(np.diff(raw_se2[:, :2], axis=0), axis=1))),
                'waypoint_count': len(raw_se2)
            })
            
            return True
            
        except Exception as e:
            print(f"❌ 궤적 생성 실패: {e}")
            return False
    
    def _save_trajectory_to_h5(self, env_name: str, pair_idx: int, 
                              trajectory: np.ndarray, metadata: Dict):
        """궤적 데이터를 HDF5에 저장"""
        
        try:
            with h5py.File(self.trajectory_file, 'a') as f:
                # 환경 그룹 생성 (없으면)
                if env_name not in f:
                    env_group = f.create_group(env_name)
                else:
                    env_group = f[env_name]
                
                # Pose pair 그룹 생성
                pair_group_name = str(pair_idx)
                if pair_group_name in env_group:
                    del env_group[pair_group_name]  # 기존 데이터 삭제
                
                pair_group = env_group.create_group(pair_group_name)
                
                # 궤적 데이터 저장
                pair_group.create_dataset('raw_trajectory', data=trajectory, compression='gzip')
                
                # 메타데이터 저장
                for key, value in metadata.items():
                    pair_group.attrs[key] = value
                
                pair_group.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
        except Exception as e:
            print(f"❌ HDF5 저장 실패: {e}")
            raise

def parse_arguments():
    """명령행 인수 파싱"""
    
    parser = argparse.ArgumentParser(description='Generate raw trajectories for environment sets')
    
    parser.add_argument('--env-set', type=str, required=True,
                       help='Environment set name (e.g., circles_only)')
    parser.add_argument('--pose-file', type=str, required=True,
                       help='Pose data HDF5 filename in root/data/pose/ (e.g., circles_only_poses.h5)')
    parser.add_argument('--env-count', type=int, default=3,
                       help='Number of environments to process')
    parser.add_argument('--pair-count', type=int, default=5,
                       help='Number of pose pairs per environment')
    parser.add_argument('--start-env-id', type=int, default=0,
                       help='Starting environment index (from available environments list)')
    
    # RRT 설정
    parser.add_argument('--rrt-range', type=float, default=0.25,
                       help='RRT extension range (default: 0.25)')
    parser.add_argument('--rrt-max-time', type=float, default=15.0,
                       help='RRT max planning time (default: 15.0)')
    parser.add_argument('--rigid-body-id', type=int, default=3,
                       help='Rigid body ID (default: 3)')
    
    # 기타 옵션
    parser.add_argument('--list-environments', action='store_true',
                       help='List available environments and exit')
    parser.add_argument('--enable-validation', action='store_true',
                       help='Enable collision validation')
    parser.add_argument('--output-stats', type=str, default=None,
                       help='Output statistics to JSON file')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    
    args = parse_arguments()
    
    print("🚀 Raw Trajectory Batch Generator")
    print(f"   환경 묶음: {args.env_set}")
    print(f"   Pose 파일: {args.pose_file}")
    print(f"   환경 수: {args.env_count}")
    print(f"   각 환경당 pair 수: {args.pair_count}")
    print(f"   RRT Range: {args.rrt_range}")
    
    try:
        # RRT 설정
        rrt_config = {
            'rigid_body_id': args.rigid_body_id,
            'max_planning_time': args.rrt_max_time,
            'range': args.rrt_range
        }
        
        # 생성기 초기화
        generator = RawTrajectoryBatchGenerator(
            env_set_name=args.env_set,
            pose_file=args.pose_file,
            rrt_config=rrt_config,
            validation_enabled=args.enable_validation
        )
        
        # 환경 목록 출력 모드
        if args.list_environments:
            print(f"\n📋 사용 가능한 환경 목록 ({len(generator.available_environments)}개):")
            for idx, env_name in enumerate(generator.available_environments):
                print(f"   {idx:3d}: {env_name}")
            print(f"\n사용법 예시:")
            print(f"   --start-env-id 0 --env-count 5    # 처음 5개 환경")
            print(f"   --start-env-id 10 --env-count 3   # 11번째부터 3개 환경")
            return 0
        
        # 배치 생성 실행
        stats = generator.generate_batch(
            env_count=args.env_count,
            pair_count=args.pair_count,
            start_env_id=args.start_env_id
        )
        
        # 결과 출력
        print(f"\n🎉 배치 생성 완료!")
        print(f"   환경 묶음: {stats['env_set_name']}")
        print(f"   처리된 환경: {stats['total_environments']}")
        print(f"   총 pair 수: {stats['total_pairs']}")
        print(f"   성공 pair: {stats['successful_pairs']}")
        print(f"   성공률: {stats['success_rate']:.1f}%")
        print(f"   총 시간: {stats['total_time']:.2f}초")
        print(f"   평균 시간: {stats['avg_time_per_pair']:.3f}초/pair")
        print(f"   궤적 파일: {stats['trajectory_file']}")
        
        # 통계 파일 저장 (옵션)
        if args.output_stats:
            import json
            with open(args.output_stats, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"   통계 저장: {args.output_stats}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
