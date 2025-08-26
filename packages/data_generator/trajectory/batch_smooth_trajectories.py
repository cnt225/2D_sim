#!/usr/bin/env python3
"""
Trajectory Smoothing Batch Processor
기존 HDF5 파일의 원본 궤적들을 B-spline으로 스무딩 처리

사용법:
    python batch_smooth_trajectories.py --env-name circle_env_000000 --pair-ids raw_pair_001,raw_pair_002
    python batch_smooth_trajectories.py --env-name circle_env_000000 --all-pairs
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 로컬 모듈 import
from trajectory_data_manager import TrajectoryDataManager, PosePairMetadata
from utils.trajectory_smoother import BSplineTrajectoryProcessor

class TrajectorySmootherBatch:
    """배치 궤적 스무딩 처리기"""
    
    def __init__(self, 
                 env_name: str,
                 bspline_config: Dict[str, Any] = None):
        """
        초기화
        
        Args:
            env_name: 환경 이름
            bspline_config: B-spline 설정
        """
        self.env_name = env_name
        
        # B-spline 설정
        self.bspline_config = bspline_config or {
            'degree': 3,
            'smoothing_factor': 0.0,
            'density_multiplier': 2
        }
        
        print(f"🌊 TrajectorySmootherBatch 초기화:")
        print(f"   환경: {env_name}")
        print(f"   B-spline degree: {self.bspline_config['degree']}")
        print(f"   Smoothing factor: {self.bspline_config['smoothing_factor']}")
        print(f"   Density multiplier: {self.bspline_config['density_multiplier']}")
        
        # 데이터 매니저 초기화
        self.data_manager = TrajectoryDataManager(env_name)
        print(f"✅ HDF5 파일: {self.data_manager.h5_file_path}")
        
        # B-spline 프로세서 초기화
        try:
            self.bspline_processor = BSplineTrajectoryProcessor(
                degree=self.bspline_config['degree'],
                smoothing_factor=self.bspline_config['smoothing_factor']
            )
            print("✅ B-spline 프로세서 초기화 완료")
        except Exception as e:
            raise RuntimeError(f"B-spline 프로세서 초기화 실패: {e}")
        
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
            
            # 2. B-spline 스무딩 실행
            smooth_start_time = time.time()
            
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
            else:
                self.stats['successful_smooth'] += 1
                print(f"✅ 스무딩 성공: {len(raw_trajectory)} → {len(smooth_trajectory)}개 점 ({smooth_time:.3f}초)")
                smoothing_method = "bspline"
            
            # 3. 메타데이터 업데이트
            updated_metadata = PosePairMetadata(
                start_pose=metadata['start_pose'],
                end_pose=metadata['end_pose'],
                generation_method=metadata.get('generation_method', 'unknown'),
                smoothing_method=smoothing_method,
                collision_free=metadata.get('collision_free', True),
                path_length=float(np.sum(np.linalg.norm(
                    np.diff(smooth_trajectory[:, :2], axis=0), axis=1
                ))),
                generation_time=metadata.get('generation_time', 0.0),
                smoothing_time=smooth_time,
                validation_time=metadata.get('validation_time', 0.0)
            )
            
            # 4. HDF5 업데이트 (smooth_trajectory 필드만)
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
    
    # B-spline 설정
    parser.add_argument('--bspline-degree', type=int, default=3,
                       help='B-spline degree (default: 3)')
    parser.add_argument('--smoothing-factor', type=float, default=0.0,
                       help='B-spline smoothing factor (default: 0.0)')
    parser.add_argument('--density-multiplier', type=float, default=2.0,
                       help='Point density multiplier (default: 2.0)')
    
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
        # B-spline 설정
        bspline_config = {
            'degree': args.bspline_degree,
            'smoothing_factor': args.smoothing_factor,
            'density_multiplier': args.density_multiplier
        }
        
        # 스무딩 처리기 초기화
        smoother = TrajectorySmootherBatch(
            env_name=args.env_name,
            bspline_config=bspline_config
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
        print(f"   B-spline 성공률: {stats['bspline_success_rate']:.1f}%")
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

