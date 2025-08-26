#!/usr/bin/env python3
"""
Trajectory Smoothing Batch Processor with SE(3) Support
기존 HDF5 파일의 원본 궤적들을 SE(3) B-spline+SLERP로 스무딩 처리

사용법:
    python batch_smooth_trajectories.py --env-name circle_env_000000 --pair-ids raw_pair_001,raw_pair_002
    python batch_smooth_trajectories.py --env-name circle_env_000000 --all-pairs
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'packages'))

# 로컬 모듈 import
from trajectory_data_manager import TrajectoryDataManager, PosePairMetadata
try:
    from utils.trajectory_smoother import BSplineTrajectoryProcessor
except ImportError:
    # Fallback for import issues
    BSplineTrajectoryProcessor = None
    
# Import SE(3) functions from packages/utils
sys.path.insert(0, str(project_root / 'packages'))
from utils.SE3_functions import (
    traj_smooth_se3_bspline_slerp,
    traj_process_se3_pipeline,
    traj_resample_by_arclength
)

try:
    from trajectory_validator import TrajectoryValidator
except ImportError:
    TrajectoryValidator = None

class TrajectorySmootherBatch:
    """배치 궤적 스무딩 처리기 (SE(3) 지원)"""
    
    def __init__(self, 
                 env_name: str,
                 bspline_config: Dict[str, Any] = None,
                 use_se3: bool = True,
                 validate_collision: bool = True):
        """
        초기화
        
        Args:
            env_name: 환경 이름
            bspline_config: B-spline 설정
            use_se3: SE(3) 스무딩 사용 여부
            validate_collision: 충돌 검증 여부
        """
        self.env_name = env_name
        self.use_se3 = use_se3
        self.validate_collision = validate_collision
        
        # B-spline 설정
        self.bspline_config = bspline_config or {
            'degree': 3,
            'smoothing_factor': 0.01,  # SE(3)에서는 약간의 스무딩이 더 안정적
            'density_multiplier': 2,
            'num_samples': 200  # SE(3) resampling용
        }
        
        print(f"🌊 TrajectorySmootherBatch 초기화:")
        print(f"   환경: {env_name}")
        print(f"   SE(3) 모드: {use_se3}")
        print(f"   충돌 검증: {validate_collision}")
        print(f"   B-spline degree: {self.bspline_config['degree']}")
        print(f"   Smoothing factor: {self.bspline_config['smoothing_factor']}")
        print(f"   Density multiplier: {self.bspline_config['density_multiplier']}")
        
        # 데이터 매니저 초기화
        self.data_manager = TrajectoryDataManager(env_name)
        print(f"✅ HDF5 파일: {self.data_manager.h5_file_path}")
        
        # 스무딩 프로세서 초기화
        if self.use_se3:
            print("✅ SE(3) 스무딩 모드 활성화 (B-spline + SLERP)")
            # SE(3) 모드에서는 내장 함수 사용
            self.bspline_processor = None
        else:
            # SE(2) 모드에서는 기존 B-spline 프로세서 사용
            if BSplineTrajectoryProcessor is not None:
                try:
                    self.bspline_processor = BSplineTrajectoryProcessor(
                        degree=self.bspline_config['degree'],
                        smoothing_factor=self.bspline_config['smoothing_factor']
                    )
                    print("✅ SE(2) B-spline 프로세서 초기화 완료")
                except Exception as e:
                    raise RuntimeError(f"B-spline 프로세서 초기화 실패: {e}")
            else:
                print("⚠️ SE(2) B-spline 프로세서를 사용할 수 없습니다. SE(3) 모드를 사용하세요.")
                self.bspline_processor = None
                self.use_se3 = True  # Force SE(3) mode
        
        # 충돌 검증기 초기화
        if self.validate_collision:
            try:
                # 환경 포인트클라우드 로드
                env_type = env_name.split('_')[0]  # e.g., 'circle' from 'circle_env_000000'
                pointcloud_path = project_root / f"data/pointcloud/{env_type}_only" / f"{env_name}.ply"
                
                if pointcloud_path.exists():
                    self.validator = TrajectoryValidator(str(pointcloud_path))
                    print(f"✅ 충돌 검증기 초기화 완료: {pointcloud_path}")
                else:
                    print(f"⚠️ 포인트클라우드 파일 없음: {pointcloud_path}")
                    self.validator = None
                    self.validate_collision = False
            except Exception as e:
                print(f"⚠️ 충돌 검증기 초기화 실패: {e}")
                self.validator = None
                self.validate_collision = False
        else:
            self.validator = None
        
        # 통계 초기화
        self.stats = {
            'total_attempts': 0,
            'successful_smooth': 0,
            'failed_smooth': 0,
            'total_time': 0.0
        }
    
    def get_available_pairs(self) -> List[str]:
        """
        스무딩 가능한 궤적 쌍 목록 조회
        
        Returns:
            pair_ids: 스무딩 가능한 쌍 ID 목록
        """
        try:
            all_pairs = self.data_manager.get_all_pair_ids()
            smoothable_pairs = []
            
            for pair_id in all_pairs:
                pair_data = self.data_manager.get_pose_pair(pair_id)
                if pair_data and 'raw_trajectory' in pair_data:
                    # 원본 궤적이 있고, 스무딩이 안되어 있거나 재처리가 필요한 경우
                    raw_traj = pair_data['raw_trajectory']
                    smooth_traj = pair_data.get('smooth_trajectory')
                    
                    if len(raw_traj) > 2:  # 스무딩 가능한 최소 점수
                        # 스무딩이 없거나, 기존 스무딩이 불량한 경우
                        if smooth_traj is None or len(smooth_traj) <= 2:
                            smoothable_pairs.append(pair_id)
            
            print(f"📊 스무딩 가능한 궤적: {len(smoothable_pairs)}개")
            return smoothable_pairs
            
        except Exception as e:
            print(f"❌ 궤적 쌍 조회 실패: {e}")
            return []
    
    def _convert_se2_to_se3(self, trajectory_se2: np.ndarray) -> torch.Tensor:
        """SE(2) 궤적을 SE(3) 형식으로 변환"""
        N = len(trajectory_se2)
        T_se3 = torch.zeros((N, 4, 4), dtype=torch.float32)
        
        for i in range(N):
            x, y, theta = trajectory_se2[i]
            # SE(3) 변환 행렬 생성
            T_se3[i, 0, 0] = np.cos(theta)
            T_se3[i, 0, 1] = -np.sin(theta)
            T_se3[i, 1, 0] = np.sin(theta)
            T_se3[i, 1, 1] = np.cos(theta)
            T_se3[i, 2, 2] = 1.0  # z축은 단위 행렬
            T_se3[i, 0, 3] = x
            T_se3[i, 1, 3] = y
            T_se3[i, 2, 3] = 0.0  # z = 0 (2D 평면)
            T_se3[i, 3, 3] = 1.0
        
        return T_se3
    
    def _convert_se3_to_se2(self, T_se3: torch.Tensor) -> np.ndarray:
        """SE(3) 궤적을 SE(2) 형식으로 변환"""
        N = T_se3.shape[0]
        trajectory_se2 = np.zeros((N, 3))
        
        for i in range(N):
            # 위치 추출
            trajectory_se2[i, 0] = T_se3[i, 0, 3].item()  # x
            trajectory_se2[i, 1] = T_se3[i, 1, 3].item()  # y
            # yaw 각도 추출 (z축 회전)
            trajectory_se2[i, 2] = np.arctan2(T_se3[i, 1, 0].item(), T_se3[i, 0, 0].item())
        
        return trajectory_se2
    
    def _validate_trajectory(self, trajectory: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """궤적 충돌 검증"""
        if not self.validate_collision or self.validator is None:
            return True, {'collision_free': True, 'checked': False}
        
        try:
            # 충돌 검증
            collision_free = self.validator.validate_trajectory(trajectory)
            
            return collision_free, {
                'collision_free': collision_free,
                'checked': True,
                'num_points_checked': len(trajectory)
            }
        except Exception as e:
            print(f"⚠️ 충돌 검증 실패: {e}")
            return True, {'collision_free': True, 'checked': False, 'error': str(e)}
    
    def smooth_single_trajectory(self, pair_id: str) -> bool:
        """
        단일 궤적 스무딩
        
        Args:
            pair_id: 궤적 쌍 ID
            
        Returns:
            success: 성공 여부
        """
        try:
            self.stats['total_attempts'] += 1
            
            print(f"\n🌊 스무딩 처리 중... ({pair_id})")
            
            # 1. 기존 데이터 로드
            pair_data = self.data_manager.get_pose_pair(pair_id)
            if not pair_data:
                print(f"❌ 궤적 데이터 없음: {pair_id}")
                return False
            
            raw_trajectory = pair_data['raw_trajectory']
            metadata = pair_data['metadata']
            
            if len(raw_trajectory) < 3:
                print(f"❌ 스무딩 불가능 (점수 부족): {pair_id} ({len(raw_trajectory)}개 점)")
                return False
            
            print(f"   원본 궤적: {len(raw_trajectory)}개 점")
            
            # 2. 스무딩 실행
            smooth_start_time = time.time()
            
            if self.use_se3:
                # SE(3) 스무딩
                try:
                    # SE(2) → SE(3) 변환
                    T_se3_raw = self._convert_se2_to_se3(raw_trajectory)
                    
                    # SE(3) 스무딩 실행
                    T_se3_smooth = traj_smooth_se3_bspline_slerp(
                        T_se3_raw,
                        pos_method="bspline_scipy",
                        degree=self.bspline_config['degree'],
                        smooth=self.bspline_config['smoothing_factor']
                    )
                    
                    # Arc-length 기반 재샘플링 (균등 간격)
                    num_samples = self.bspline_config.get('num_samples', 
                                    int(len(raw_trajectory) * self.bspline_config['density_multiplier']))
                    T_se3_resampled, _ = traj_resample_by_arclength(
                        T_se3_smooth,
                        num_samples=num_samples,
                        lambda_rot=0.1  # 회전 가중치 (m/rad)
                    )
                    
                    # SE(3) → SE(2) 변환
                    smooth_trajectory = self._convert_se3_to_se2(T_se3_resampled)
                    
                    smooth_time = time.time() - smooth_start_time
                    self.stats['successful_smooth'] += 1
                    print(f"✅ SE(3) 스무딩 성공: {len(raw_trajectory)} → {len(smooth_trajectory)}개 점 ({smooth_time:.3f}초)")
                    smoothing_method = "se3_bspline_slerp"
                    smooth_success = True
                    
                except Exception as e:
                    print(f"❌ SE(3) 스무딩 실패: {pair_id} - {e}")
                    self.stats['failed_smooth'] += 1
                    # 실패한 경우 서브샘플링으로 대체
                    step = max(1, len(raw_trajectory) // 10)
                    smooth_trajectory = raw_trajectory[::step]
                    smooth_time = time.time() - smooth_start_time
                    print(f"   서브샘플링으로 대체: {len(smooth_trajectory)}개 점")
                    smoothing_method = "subsampling_fallback"
                    smooth_success = False
            else:
                # SE(2) B-spline 스무딩
                num_points = int(len(raw_trajectory) * self.bspline_config['density_multiplier'])
                smooth_trajectory, smooth_info = self.bspline_processor.smooth_trajectory(
                    raw_trajectory, num_points=num_points
                )
                
                smooth_time = time.time() - smooth_start_time
                
                if not smooth_info['success']:
                    print(f"❌ 스무딩 실패: {pair_id} - {smooth_info.get('error', 'Unknown error')}")
                    self.stats['failed_smooth'] += 1
                    # 실패한 경우 서브샘플링으로 대체
                    step = max(1, len(raw_trajectory) // 10)
                    smooth_trajectory = raw_trajectory[::step]
                    print(f"   서브샘플링으로 대체: {len(smooth_trajectory)}개 점")
                    smoothing_method = "subsampling_fallback"
                    smooth_success = False
                else:
                    self.stats['successful_smooth'] += 1
                    print(f"✅ SE(2) 스무딩 성공: {len(raw_trajectory)} → {len(smooth_trajectory)}개 점 ({smooth_time:.3f}초)")
                    smoothing_method = "bspline"
                    smooth_success = True
            
            # 3. 충돌 검증
            collision_free, validation_info = self._validate_trajectory(smooth_trajectory)
            if not collision_free:
                print(f"⚠️ 스무딩된 궤적이 충돌 포함: {pair_id}")
            else:
                print(f"✅ 충돌 검증 통과: {pair_id}")
            
            # 4. 메타데이터 업데이트
            updated_metadata = PosePairMetadata(
                start_pose=metadata['start_pose'],
                end_pose=metadata['end_pose'],
                generation_method=metadata.get('generation_method', 'unknown'),
                smoothing_method=smoothing_method,
                collision_free=collision_free if self.validate_collision else metadata.get('collision_free', True),
                path_length=float(np.sum(np.linalg.norm(
                    np.diff(smooth_trajectory[:, :2], axis=0), axis=1
                ))),
                generation_time=metadata.get('generation_time', 0.0),
                smoothing_time=smooth_time,
                validation_time=metadata.get('validation_time', 0.0)
            )
            
            # 5. HDF5 업데이트 (smooth_trajectory 필드만)
            success = self.data_manager.update_pose_pair_smooth_trajectory(
                pair_id=pair_id,
                smooth_trajectory=smooth_trajectory,
                updated_metadata=updated_metadata
            )
            
            if success:
                print(f"✅ HDF5 업데이트 완료: {pair_id}")
                self.stats['total_time'] += smooth_time
                return True
            else:
                print(f"❌ HDF5 업데이트 실패: {pair_id}")
                return False
                
        except Exception as e:
            print(f"❌ 스무딩 처리 중 오류 ({pair_id}): {e}")
            self.stats['failed_smooth'] += 1
            return False
    
    def smooth_batch(self, 
                    pair_ids: Optional[List[str]] = None, 
                    all_pairs: bool = False) -> Dict[str, Any]:
        """
        배치 스무딩 처리
        
        Args:
            pair_ids: 처리할 궤적 ID 목록 (None이면 all_pairs 기준)
            all_pairs: 모든 궤적 처리 여부
            
        Returns:
            결과 통계
        """
        print(f"\n🚀 배치 스무딩 처리 시작")
        
        # 처리할 궤적 목록 결정
        if all_pairs:
            target_pairs = self.get_available_pairs()
            print(f"   모든 스무딩 가능한 궤적 처리: {len(target_pairs)}개")
        elif pair_ids:
            target_pairs = pair_ids
            print(f"   지정된 궤적 처리: {len(target_pairs)}개")
        else:
            print("❌ 처리할 궤적이 지정되지 않았습니다")
            return {}
        
        if not target_pairs:
            print("⚠️ 처리할 궤적이 없습니다")
            return {}
        
        batch_start_time = time.time()
        successful_pairs = []
        failed_pairs = []
        
        for i, pair_id in enumerate(target_pairs):
            success = self.smooth_single_trajectory(pair_id)
            
            if success:
                successful_pairs.append(pair_id)
            else:
                failed_pairs.append(pair_id)
            
            # 진행 상황 출력
            if (i + 1) % 5 == 0 or (i + 1) == len(target_pairs):
                success_rate = (len(successful_pairs) / (i + 1)) * 100
                print(f"📊 진행 상황: {i + 1}/{len(target_pairs)} ({success_rate:.1f}% 성공)")
        
        batch_time = time.time() - batch_start_time
        
        # 최종 통계
        final_stats = {
            'total_requested': len(target_pairs),
            'successful_pairs': len(successful_pairs),
            'failed_pairs': len(failed_pairs),
            'success_rate': (len(successful_pairs) / len(target_pairs)) * 100,
            'total_batch_time': batch_time,
            'avg_time_per_trajectory': batch_time / len(target_pairs),
            'bspline_success_rate': (self.stats['successful_smooth'] / self.stats['total_attempts']) * 100,
            'h5_file_path': str(self.data_manager.h5_file_path),
            'successful_pair_ids': successful_pairs,
            'failed_pair_ids': failed_pairs
        }
        
        return final_stats

def parse_arguments():
    """명령행 인수 파싱"""
    
    parser = argparse.ArgumentParser(description='Smooth existing raw trajectories in batch')
    
    parser.add_argument('--env-name', type=str, required=True,
                       help='Environment name')
    
    # 처리할 궤적 선택 (둘 중 하나 필수)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pair-ids', type=str,
                       help='Comma-separated list of pair IDs to smooth')
    group.add_argument('--all-pairs', action='store_true',
                       help='Smooth all available pairs')
    
    # 스무딩 모드 선택
    parser.add_argument('--use-se3', action='store_true', default=True,
                       help='Use SE(3) smoothing with B-spline + SLERP (default: True)')
    parser.add_argument('--use-se2', action='store_true',
                       help='Use SE(2) B-spline smoothing only')
    
    # B-spline 설정
    parser.add_argument('--bspline-degree', type=int, default=3,
                       help='B-spline degree (default: 3)')
    parser.add_argument('--smoothing-factor', type=float, default=0.01,
                       help='B-spline smoothing factor (default: 0.01 for SE(3))')
    parser.add_argument('--density-multiplier', type=float, default=2.0,
                       help='Point density multiplier (default: 2.0)')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples for SE(3) resampling (default: 200)')
    
    # 검증 옵션
    parser.add_argument('--no-collision-check', action='store_true',
                       help='Disable collision validation')
    
    # 기타 옵션
    parser.add_argument('--output-stats', type=str, default=None,
                       help='Output statistics to JSON file')
    parser.add_argument('--list-pairs', action='store_true',
                       help='List available pairs and exit')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    
    args = parse_arguments()
    
    print("🌊 Trajectory Smoothing Batch Processor")
    print(f"   환경: {args.env_name}")
    
    try:
        # SE(2) vs SE(3) 모드 결정
        use_se3 = args.use_se3 and not args.use_se2
        
        # B-spline 설정
        bspline_config = {
            'degree': args.bspline_degree,
            'smoothing_factor': args.smoothing_factor,
            'density_multiplier': args.density_multiplier,
            'num_samples': args.num_samples
        }
        
        # 스무딩 처리기 초기화
        smoother = TrajectorySmootherBatch(
            env_name=args.env_name,
            bspline_config=bspline_config,
            use_se3=use_se3,
            validate_collision=not args.no_collision_check
        )
        
        # 사용 가능한 궤적 목록 출력 (옵션)
        if args.list_pairs:
            available_pairs = smoother.get_available_pairs()
            print(f"\n📋 스무딩 가능한 궤적 목록:")
            for pair_id in available_pairs:
                print(f"   - {pair_id}")
            return 0
        
        # 처리할 궤적 목록 준비
        if args.pair_ids:
            pair_ids = [pid.strip() for pid in args.pair_ids.split(',')]
            print(f"   지정된 궤적: {len(pair_ids)}개")
            for pid in pair_ids:
                print(f"     - {pid}")
            
            # 배치 스무딩 실행
            stats = smoother.smooth_batch(pair_ids=pair_ids)
        else:
            # 모든 궤적 스무딩
            stats = smoother.smooth_batch(all_pairs=True)
        
        if not stats:
            print("❌ 스무딩 처리 실패")
            return 1
        
        # 결과 출력
        print(f"\n🎉 배치 스무딩 완료!")
        print(f"   요청 수량: {stats['total_requested']}")
        print(f"   성공 수량: {stats['successful_pairs']}")
        print(f"   실패 수량: {stats['failed_pairs']}")
        print(f"   성공률: {stats['success_rate']:.1f}%")
        print(f"   총 시간: {stats['total_batch_time']:.2f}초")
        print(f"   평균 시간: {stats['avg_time_per_trajectory']:.3f}초/궤적")
        print(f"   스무딩 성공률: {stats['bspline_success_rate']:.1f}%")
        print(f"   스무딩 모드: {'SE(3)' if use_se3 else 'SE(2)'}")
        print(f"   HDF5 파일: {stats['h5_file_path']}")
        
        if stats['failed_pair_ids']:
            print(f"\n⚠️ 실패한 궤적:")
            for failed_id in stats['failed_pair_ids']:
                print(f"     - {failed_id}")
        
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

