#!/usr/bin/env python3
"""
Batch Trajectory Generator with SE(3) Smoothing Pipeline
RRT 생성 + SE(3) 스무딩 + 리샘플링 통합 파이프라인
기존 batch_generate_raw_trajectories.py와 일관된 구조 유지
"""

import os
import sys
import argparse
import time
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'packages'))

# 필수 모듈 import
from rrt_connect import create_se3_planner, SE3TrajectoryResult

# pose 데이터 로드를 위한 모듈
sys.path.append(str(project_root / "packages" / "data_generator" / "pose"))
from unified_pose_manager import UnifiedPoseManager

# SE(3) functions
from packages.utils.SE3_functions import (
    traj_smooth_se3_bspline_slerp,
    traj_resample_by_arclength
)

print("✅ 필수 모듈 import 완료")


class TrajectoryBatchGenerator:
    """RRT + SE(3) 스무딩 통합 배치 궤적 생성기"""
    
    def __init__(self, 
                 env_set_name: str,
                 pose_file: str,
                 rrt_config: Dict[str, Any] = None,
                 smoothing_config: Dict[str, Any] = None,
                 validation_enabled: bool = False,
                 output_format: str = 'se2'):
        """
        초기화
        
        Args:
            env_set_name: 환경 묶음 이름 (예: 'circles_only')
            pose_file: Pose 데이터 HDF5 파일명 (root/data/pose/ 기준)
            rrt_config: RRT 설정
            smoothing_config: SE(3) 스무딩 설정
            validation_enabled: 충돌 검증 사용 여부
            output_format: 궤적 출력 형식 ('se2', 'se3', 'se3_6d', 'quaternion_7d')
        """
        self.env_set_name = env_set_name
        self.validation_enabled = validation_enabled
        
        # 출력 형식 검증
        valid_formats = ['se2', 'se3', 'se3_6d', 'quaternion_7d']
        if output_format not in valid_formats:
            raise ValueError(f"❌ 지원하지 않는 출력 형식: {output_format}. 지원 형식: {valid_formats}")
        self.output_format = output_format
        
        # Pose 파일 경로
        self.pose_file = str(project_root / "data" / "pose" / pose_file)
        if not Path(self.pose_file).exists():
            raise FileNotFoundError(f"❌ Pose 파일을 찾을 수 없습니다: {self.pose_file}")
        
        # RRT 설정
        self.rrt_config = rrt_config or {
            'rigid_body_id': 3,
            'max_planning_time': 15.0,
            'range': 0.25
        }
        
        # SE(3) 스무딩 설정
        self.smoothing_config = smoothing_config or {
            'min_samples': 100,
            'max_samples': 500,
            'smooth_factor': 0.01,
            'degree': 3
        }
        
        print(f"🏗️ TrajectoryBatchGenerator 초기화:")
        print(f"   환경 묶음: {env_set_name}")
        print(f"   Pose 파일: {self.pose_file}")
        print(f"   출력 형식: {self.output_format}")
        print(f"   RRT Range: {self.rrt_config['range']}")
        print(f"   샘플 범위: {self.smoothing_config['min_samples']}-{self.smoothing_config['max_samples']}")
        print(f"   검증 활성화: {validation_enabled}")
        
        # Pose 매니저 초기화
        try:
            self.pose_manager = UnifiedPoseManager(self.pose_file)
            print(f"✅ Pose 매니저 초기화 완료")
        except Exception as e:
            raise RuntimeError(f"❌ Pose 매니저 초기화 실패: {e}")
        
        # Trajectory 저장소 초기화
        trajectory_dir = project_root / "data" / "trajectory"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_file = trajectory_dir / f"{env_set_name}_integrated_trajs.h5"
        
        # 통계 초기화
        self.stats = {
            'total_environments': 0,
            'total_pairs': 0,
            'successful_rrt': 0,
            'successful_smoothing': 0,
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
            self.available_environments = []
            
            with h5py.File(self.pose_file, 'r') as f:
                if 'environments' in f:
                    for env_name in f['environments'].keys():
                        # RB의 pose pairs가 있는 환경만 포함
                        env_path = f'environments/{env_name}/pose_pairs/rb_{self.rrt_config["rigid_body_id"]}'
                        if env_path in f:
                            self.available_environments.append(env_name)
            
            self.available_environments.sort()
            print(f"✅ 환경 목록 로드 완료: {len(self.available_environments)}개")
            
            if len(self.available_environments) == 0:
                raise ValueError(f"RB {self.rrt_config['rigid_body_id']}에 대한 환경이 없습니다")
                
        except Exception as e:
            raise RuntimeError(f"❌ 환경 목록 로드 실패: {e}")
    
    def _determine_num_samples(self, raw_length: int) -> int:
        """적절한 샘플 수 결정"""
        min_samples = self.smoothing_config['min_samples']
        max_samples = self.smoothing_config['max_samples']
        
        if raw_length < 50:
            return max(min_samples, raw_length * 3)
        elif raw_length < 100:
            return min(max_samples, raw_length * 2)
        elif raw_length < 200:
            return min(max_samples, int(raw_length * 1.5))
        else:
            return min(max_samples, raw_length)
    
    def _se2_to_se3_matrices(self, se2_traj: np.ndarray) -> torch.Tensor:
        """SE(2) to SE(3) 변환"""
        N = len(se2_traj)
        T_matrices = torch.zeros((N, 4, 4), dtype=torch.float32)
        
        for i in range(N):
            x, y, theta = se2_traj[i]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            T_matrices[i] = torch.tensor([
                [cos_t, -sin_t, 0, x],
                [sin_t,  cos_t, 0, y],
                [0,      0,     1, 0],
                [0,      0,     0, 1]
            ], dtype=torch.float32)
        
        return T_matrices
    
    def _convert_trajectory_format(self, T_matrices: torch.Tensor, format_type: str) -> np.ndarray:
        """궤적을 지정된 형식으로 변환"""
        T_np = T_matrices.cpu().numpy() if isinstance(T_matrices, torch.Tensor) else T_matrices
        N = T_np.shape[0]
        
        if format_type == 'se2':
            # SE(2) [x, y, yaw]
            result = np.zeros((N, 3))
            for i in range(N):
                result[i, 0] = T_np[i, 0, 3]  # x
                result[i, 1] = T_np[i, 1, 3]  # y
                result[i, 2] = np.arctan2(T_np[i, 1, 0], T_np[i, 0, 0])  # yaw
            return result
            
        elif format_type == 'se3':
            # SE(3) 4x4 행렬 그대로
            return T_np
            
        elif format_type == 'se3_6d':
            # SE(3) 6D [x, y, z, rx, ry, rz]
            result = np.zeros((N, 6))
            for i in range(N):
                result[i, 0] = T_np[i, 0, 3]  # x
                result[i, 1] = T_np[i, 1, 3]  # y
                result[i, 2] = T_np[i, 2, 3]  # z
                result[i, 3] = 0.0  # roll (고정)
                result[i, 4] = 0.0  # pitch (고정)
                result[i, 5] = np.arctan2(T_np[i, 1, 0], T_np[i, 0, 0])  # yaw
            return result
            
        elif format_type == 'quaternion_7d':
            # 쿼터니언 7D [x, y, z, qw, qx, qy, qz]
            from packages.utils.SE3_functions import trajectory_euler_to_quaternion
            # 먼저 6D로 변환 후 쿼터니언으로
            se3_6d = self._convert_trajectory_format(T_matrices, 'se3_6d')
            result = trajectory_euler_to_quaternion(se3_6d)
            return result
            
        else:
            raise ValueError(f"지원하지 않는 형식: {format_type}")
    
    def _se3_to_se2(self, T_matrices: torch.Tensor) -> np.ndarray:
        """SE(3) to SE(2) 변환 (호환성 유지)"""
        return self._convert_trajectory_format(T_matrices, 'se2')
    
    def _process_trajectory(self, env_name: str, pair_id: int,
                           start_pose: List[float], end_pose: List[float]) -> Optional[Dict]:
        """단일 궤적 생성 및 스무딩"""
        try:
            # 1. RRT 궤적 생성
            pointcloud_path = project_root / "data" / "pointcloud" / self.env_set_name / f"{env_name}.ply"
            if not pointcloud_path.exists():
                print(f"❌ 포인트클라우드 없음: {pointcloud_path}")
                return None
            
            # RRT 플래너 생성
            planner = create_se3_planner(
                self.rrt_config['rigid_body_id'],
                str(pointcloud_path)
            )
            
            # SE(3) 형식으로 변환
            start_se3 = [start_pose[0], start_pose[1], 0.0, 0.0, 0.0, start_pose[2]]
            end_se3 = [end_pose[0], end_pose[1], 0.0, 0.0, 0.0, end_pose[2]]
            
            # RRT 계획
            rrt_result = planner.plan_trajectory(
                start_se3, end_se3,
                max_planning_time=self.rrt_config['max_planning_time']
            )
            
            if not rrt_result.success:
                print(f"❌ RRT 실패")
                return None
            
            # RRT 결과를 SE(3) 4x4 행렬로 변환
            raw_trajectory = np.array(rrt_result.trajectory)
            raw_se2 = raw_trajectory[:, [0, 1, 5]]  # 임시로 SE(2) 추출
            
            print(f"✅ RRT 성공: {len(raw_se2)}개 점, {rrt_result.planning_time:.3f}초")
            self.stats['successful_rrt'] += 1
            
            # 2. SE(3) 스무딩 및 리샘플링
            num_samples = self._determine_num_samples(len(raw_se2))
            
            # SE(2) → SE(3) 4x4 행렬 변환
            T_raw = self._se2_to_se3_matrices(raw_se2)
            
            # SE(3) 스무딩
            T_smooth = traj_smooth_se3_bspline_slerp(
                T_raw, 
                degree=self.smoothing_config['degree'],
                smooth=self.smoothing_config['smooth_factor']
            )
            
            # 리샘플링
            T_resampled, _ = traj_resample_by_arclength(T_smooth, num_samples)
            
            # 지정된 형식으로 변환
            raw_formatted = self._convert_trajectory_format(T_raw, self.output_format)
            smooth_formatted = self._convert_trajectory_format(T_resampled, self.output_format)
            
            print(f"✅ 스무딩 완료: {len(raw_formatted)} → {len(smooth_formatted)}개 점 ({self.output_format} 형식)")
            self.stats['successful_smoothing'] += 1
            
            # 경로 길이 계산 (위치 좌표 기준)
            def path_length(traj_formatted):
                if self.output_format == 'se2':
                    return np.sum(np.linalg.norm(np.diff(traj_formatted[:, :2], axis=0), axis=1))
                elif self.output_format in ['se3_6d', 'quaternion_7d']:
                    return np.sum(np.linalg.norm(np.diff(traj_formatted[:, :3], axis=0), axis=1))
                elif self.output_format == 'se3':
                    positions = traj_formatted[:, :3, 3]  # 4x4 행렬에서 위치 추출
                    return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                else:
                    return 0.0
            
            return {
                'raw_trajectory': raw_formatted,
                'smooth_trajectory': smooth_formatted,
                'start_pose': start_pose,
                'end_pose': end_pose,
                'generation_time': rrt_result.planning_time,
                'path_length': path_length(smooth_formatted),
                'waypoint_count': len(smooth_formatted),
                'smoothing_degree': self.smoothing_config['degree'],
                'smoothing_factor': self.smoothing_config['smooth_factor'],
                'output_format': self.output_format
            }
            
        except Exception as e:
            print(f"❌ 궤적 생성 오류: {e}")
            return None
    
    def generate_batch(self, env_count: int, pair_count: int, start_env_id: int = 0) -> Dict[str, Any]:
        """
        환경 묶음별 배치 궤적 생성
        
        Args:
            env_count: 생성할 환경 수
            pair_count: 각 환경당 생성할 pose pair 수
            start_env_id: 시작 환경 인덱스
            
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
        end_env_id = min(start_env_id + env_count, len(self.available_environments))
        actual_env_count = end_env_id - start_env_id
        
        if actual_env_count <= 0:
            raise ValueError(f"유효하지 않은 환경 범위: {start_env_id} ~ {end_env_id}")
        
        # HDF5 파일 초기화
        self._initialize_trajectory_file()
        
        batch_start_time = time.time()
        
        # 각 환경 처리
        with h5py.File(self.trajectory_file, 'a') as f:
            for env_idx in range(start_env_id, end_env_id):
                env_name = self.available_environments[env_idx]
                print(f"\n📁 환경 처리 중: {env_name} ({env_idx - start_env_id + 1}/{actual_env_count})")
                print(f"   환경 인덱스: {env_idx}/{len(self.available_environments)}")
                
                # Pose pairs 로드
                pose_pairs = self.pose_manager.get_pose_pairs(
                    env_name, 
                    self.rrt_config['rigid_body_id']
                )
                
                if not pose_pairs:
                    print(f"⚠️ Pose pairs 없음: {env_name}")
                    continue
                
                print(f"✅ Pose pairs 로드: {len(pose_pairs)}개")
                
                # 환경 그룹 생성
                if env_name not in f:
                    env_group = f.create_group(env_name)
                else:
                    env_group = f[env_name]
                
                # 각 pose pair 처리
                successful_pairs = 0
                # pose_pairs가 tuple이면 첫 번째 요소가 실제 데이터
                if isinstance(pose_pairs, tuple):
                    pose_array = pose_pairs[0]  # numpy array [N, 12]
                else:
                    pose_array = pose_pairs
                
                for pair_idx in range(min(pair_count, len(pose_array))):
                    # pose_array의 각 행: [start_x, start_y, 0, 0, 0, start_yaw, end_x, end_y, 0, 0, 0, end_yaw]
                    pair_data = pose_array[pair_idx]
                    start_pose = [pair_data[0], pair_data[1], pair_data[5]]  # [x, y, yaw]
                    end_pose = [pair_data[6], pair_data[7], pair_data[11]]   # [x, y, yaw]
                    
                    print(f"🛤️ Pair {pair_idx}: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {start_pose[2]:.2f}] → "
                          f"[{end_pose[0]:.2f}, {end_pose[1]:.2f}, {end_pose[2]:.2f}]")
                    
                    # 궤적 생성 및 스무딩
                    result = self._process_trajectory(env_name, pair_idx, start_pose, end_pose)
                    
                    if result:
                        # HDF5에 저장
                        pair_group_name = str(pair_idx)
                        if pair_group_name in env_group:
                            del env_group[pair_group_name]
                        
                        pair_group = env_group.create_group(pair_group_name)
                        
                        # 궤적 데이터 저장 (형식에 따른 압축 설정)
                        compression_opts = 6 if self.output_format == 'se3' else 9
                        pair_group.create_dataset('raw_trajectory', 
                                                 data=result['raw_trajectory'],
                                                 compression='gzip',
                                                 compression_opts=compression_opts)
                        pair_group.create_dataset('smooth_trajectory',
                                                 data=result['smooth_trajectory'],
                                                 compression='gzip',
                                                 compression_opts=compression_opts)
                        
                        # 메타데이터 저장
                        pair_group.attrs['start_pose'] = result['start_pose']
                        pair_group.attrs['end_pose'] = result['end_pose']
                        pair_group.attrs['generation_time'] = result['generation_time']
                        pair_group.attrs['path_length'] = result['path_length']
                        pair_group.attrs['waypoint_count'] = result['waypoint_count']
                        # 스무딩된 궤적의 경로 길이 별도 계산
                        def path_length(traj):
                            diffs = np.diff(traj[:, :2], axis=0)
                            return np.sum(np.linalg.norm(diffs, axis=1))
                        
                        pair_group.attrs['raw_path_length'] = path_length(result['raw_trajectory'])
                        pair_group.attrs['raw_waypoint_count'] = len(result['raw_trajectory'])
                        pair_group.attrs['smooth_path_length'] = path_length(result['smooth_trajectory'])
                        pair_group.attrs['smooth_waypoint_count'] = len(result['smooth_trajectory'])
                        pair_group.attrs['smoothing_degree'] = result['smoothing_degree']
                        pair_group.attrs['smoothing_factor'] = result['smoothing_factor']
                        pair_group.attrs['output_format'] = result['output_format']
                        pair_group.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                        
                        successful_pairs += 1
                        self.stats['total_pairs'] += 1
                    
                print(f"📊 환경 '{env_name}' 완료: {successful_pairs}/{pair_count} 성공")
                self.stats['total_environments'] += 1
                
                # 주기적으로 flush
                if self.stats['total_environments'] % 5 == 0:
                    f.flush()
        
        # 최종 통계
        batch_time = time.time() - batch_start_time
        self.stats['total_time'] = batch_time
        
        print(f"\n🎉 배치 생성 완료!")
        print(f"   환경 묶음: {self.env_set_name}")
        print(f"   처리된 환경: {self.stats['total_environments']}")
        print(f"   총 pair 수: {self.stats['total_pairs']}")
        print(f"   RRT 성공: {self.stats['successful_rrt']}")
        print(f"   스무딩 성공: {self.stats['successful_smoothing']}")
        print(f"   성공률: {self.stats['successful_rrt'] / max(1, self.stats['total_pairs']) * 100:.1f}%")
        print(f"   총 시간: {batch_time:.2f}초")
        print(f"   평균 시간: {batch_time / max(1, self.stats['total_pairs']):.3f}초/pair")
        print(f"   궤적 파일: {self.trajectory_file}")
        
        return self.stats
    
    def _initialize_trajectory_file(self):
        """HDF5 파일 초기화"""
        with h5py.File(self.trajectory_file, 'a') as f:
            # 메타데이터 그룹
            if 'metadata' not in f:
                meta_group = f.create_group('metadata')
                meta_group.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                meta_group.attrs['env_set_name'] = self.env_set_name
                meta_group.attrs['output_format'] = self.output_format
                meta_group.attrs['rigid_body_id'] = self.rrt_config['rigid_body_id']
                meta_group.attrs['rrt_range'] = self.rrt_config['range']
                meta_group.attrs['smoothing_min_samples'] = self.smoothing_config['min_samples']
                meta_group.attrs['smoothing_max_samples'] = self.smoothing_config['max_samples']
                meta_group.attrs['smoothing_factor'] = self.smoothing_config['smooth_factor']
            
        print(f"✅ 궤적 파일 초기화 완료: {self.trajectory_file}")


def main():
    parser = argparse.ArgumentParser(description='Batch trajectory generation with SE(3) smoothing')
    
    # 필수 인자
    parser.add_argument('--env-set', type=str, required=True,
                       help='Environment set name (e.g., circles_only)')
    parser.add_argument('--pose-file', type=str, required=True,
                       help='Pose HDF5 file name')
    parser.add_argument('--env-count', type=int, required=True,
                       help='Number of environments to process')
    parser.add_argument('--pair-count', type=int, required=True,
                       help='Number of trajectory pairs per environment')
    
    # 선택 인자
    parser.add_argument('--start-env-id', type=int, default=0,
                       help='Starting environment index (default: 0)')
    parser.add_argument('--rigid-body-id', type=int, default=3,
                       help='Rigid body ID (default: 3)')
    parser.add_argument('--rrt-range', type=float, default=0.25,
                       help='RRT extension range (default: 0.25)')
    parser.add_argument('--rrt-max-time', type=float, default=15.0,
                       help='Maximum RRT planning time (default: 15.0)')
    
    # SE(3) 스무딩 설정
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum samples after resampling (default: 100)')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples after resampling (default: 500)')
    parser.add_argument('--smooth-factor', type=float, default=0.01,
                       help='Smoothing factor (default: 0.01)')
    parser.add_argument('--degree', type=int, default=3,
                       help='B-spline degree (default: 3)')
    
    # 출력 형식
    parser.add_argument('--output-format', type=str, default='se3_6d',
                       choices=['se2', 'se3', 'se3_6d', 'quaternion_7d'],
                       help='Trajectory output format (default: se3_6d)')
    
    # 기타
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip collision validation')
    parser.add_argument('--list-environments', action='store_true',
                       help='List available environments and exit')
    
    args = parser.parse_args()
    
    # RRT 설정
    rrt_config = {
        'rigid_body_id': args.rigid_body_id,
        'max_planning_time': args.rrt_max_time,
        'range': args.rrt_range
    }
    
    # 스무딩 설정
    smoothing_config = {
        'min_samples': args.min_samples,
        'max_samples': args.max_samples,
        'smooth_factor': args.smooth_factor,
        'degree': args.degree
    }
    
    try:
        # 생성기 초기화
        generator = TrajectoryBatchGenerator(
            env_set_name=args.env_set,
            pose_file=args.pose_file,
            rrt_config=rrt_config,
            smoothing_config=smoothing_config,
            validation_enabled=not args.no_validation,
            output_format=args.output_format
        )
        
        # 환경 목록 출력 모드
        if args.list_environments:
            print(f"\n📋 사용 가능한 환경 목록 ({len(generator.available_environments)}개):")
            for i, env_name in enumerate(generator.available_environments[:20]):
                print(f"   {i:4d}: {env_name}")
            if len(generator.available_environments) > 20:
                print(f"   ... 그리고 {len(generator.available_environments) - 20}개 더")
            return 0
        
        # 배치 생성 실행
        stats = generator.generate_batch(
            env_count=args.env_count,
            pair_count=args.pair_count,
            start_env_id=args.start_env_id
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())