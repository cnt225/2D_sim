#!/usr/bin/env python3
"""
🚀 통합 Trajectory Smoothing System
기존 raw trajectory HDF5 파일을 읽어서 스무딩된 새 HDF5 파일 생성
- SE(3) B-spline + SLERP 스무딩 지원
- 메모리 효율적 청크 처리
- 안전한 파일 관리 (입력 → 새 출력)

사용법:
    # 전체 파일 스무딩 (권장)
    python batch_smooth_trajectories.py --input circles_only_trajs.h5 --output circles_only_smooth.h5
    
    # 메모리 제한 환경에서
    python batch_smooth_trajectories.py --input large_file.h5 --output smooth_file.h5 --chunk-size 10
"""

import os
import sys
import argparse
import time
import gc
import psutil
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'packages'))

# SE(3) 함수 import
from packages.utils.SE3_functions import (
    traj_smooth_se3_bspline_slerp,
    traj_resample_by_arclength
)


class IntegratedTrajectorySmoother:
    """
    통합 궤적 스무딩 시스템
    - 메모리 효율적 청크 처리
    - SE(3) B-spline + SLERP 스무딩
    - 안전한 파일 관리
    """
    
    def __init__(self, input_file: str, output_file: str, chunk_size: int = 20, 
                 memory_threshold: float = 75.0, verbose: bool = False, output_format: str = 'se3_6d'):
        """
        Args:
            input_file: 입력 HDF5 파일 경로
            output_file: 출력 HDF5 파일 경로
            chunk_size: 환경 청크 크기
            memory_threshold: 메모리 경고 임계값 (%)
            verbose: 상세 출력
            output_format: 궤적 출력 형식 ('se2', 'se3', 'se3_6d', 'quaternion_7d')
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold
        self.verbose = verbose
        
        # 출력 형식 검증
        valid_formats = ['se2', 'se3', 'se3_6d', 'quaternion_7d']
        if output_format not in valid_formats:
            raise ValueError(f"❌ 지원하지 않는 출력 형식: {output_format}. 지원 형식: {valid_formats}")
        self.output_format = output_format
        
        # 출력 디렉토리 생성
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 통계
        self.stats = {
            'total_environments': 0,
            'processed_environments': 0,
            'total_trajectories': 0,
            'successful_smoothing': 0,
            'failed_smoothing': 0,
            'processing_time': 0.0,
            'memory_warnings': 0
        }
        
        print(f"🚀 통합 궤적 스무딩 시스템 초기화")
        print(f"   입력: {self.input_file}")
        print(f"   출력: {self.output_file}")
        print(f"   청크 크기: {chunk_size}")
        print(f"   메모리 임계값: {memory_threshold}%")
    
    def _get_memory_usage(self) -> Tuple[float, float, float]:
        """현재 메모리 사용량 반환 (MB, %, GPU MB)"""
        process = psutil.Process()
        cpu_mb = process.memory_info().rss / 1024 / 1024
        system_percent = psutil.virtual_memory().percent
        
        gpu_mb = 0.0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return cpu_mb, system_percent, gpu_mb
    
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

    def _monitor_memory(self, context: str = "") -> None:
        """메모리 모니터링 및 경고"""
        cpu_mb, system_percent, gpu_mb = self._get_memory_usage()
        
        if self.verbose:
            print(f"   💾 메모리 {context}: CPU {cpu_mb:.1f}MB, 시스템 {system_percent:.1f}%, GPU {gpu_mb:.1f}MB")
        
        if system_percent > self.memory_threshold:
            print(f"⚠️ 메모리 사용률 높음: {system_percent:.1f}%")
            self.stats['memory_warnings'] += 1
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _se3_smooth_trajectory(self, raw_traj: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        SE(3) 방식으로 궤적 스무딩
        
        Args:
            raw_traj: SE(2) 궤적 (N, 3) [x, y, yaw]
            
        Returns:
            (smoothed_traj, stats): 스무딩된 궤적과 통계
        """
        try:
            if self.verbose:
                print(f"      🔧 입력 궤적: {raw_traj.shape}, 범위: x[{raw_traj[:, 0].min():.2f}-{raw_traj[:, 0].max():.2f}], y[{raw_traj[:, 1].min():.2f}-{raw_traj[:, 1].max():.2f}], yaw[{raw_traj[:, 2].min():.2f}-{raw_traj[:, 2].max():.2f}]")
            
            # 입력 검증
            if len(raw_traj) < 2:
                raise ValueError(f"궤적 포인트 부족: {len(raw_traj)}개")
            
            if raw_traj.shape[1] != 3:
                raise ValueError(f"잘못된 형태: {raw_traj.shape}, 3열 필요")
            
            if np.any(np.isnan(raw_traj)) or np.any(np.isinf(raw_traj)):
                raise ValueError("NaN 또는 Inf 값 발견")
            
            # SE(2) → SE(3) 변환
            se3_matrices = []
            for i, pose in enumerate(raw_traj):
                x, y, yaw = pose
                se3_matrix = torch.eye(4, dtype=torch.float64)
                se3_matrix[0, 3] = x
                se3_matrix[1, 3] = y
                se3_matrix[0, 0] = torch.cos(torch.tensor(yaw, dtype=torch.float64))
                se3_matrix[0, 1] = -torch.sin(torch.tensor(yaw, dtype=torch.float64))
                se3_matrix[1, 0] = torch.sin(torch.tensor(yaw, dtype=torch.float64))
                se3_matrix[1, 1] = torch.cos(torch.tensor(yaw, dtype=torch.float64))
                se3_matrices.append(se3_matrix)
            
            se3_trajectory = torch.stack(se3_matrices)
            
            if self.verbose:
                print(f"      🔧 SE(3) 변환 완료: {se3_trajectory.shape}")
            
            # SE(3) B-spline + SLERP 스무딩
            smoothed_se3 = traj_smooth_se3_bspline_slerp(
                se3_trajectory,
                pos_method="bspline_scipy",
                degree=3,
                smooth=0.01
            )
            
            if self.verbose:
                print(f"      🔧 SE(3) 스무딩 완료: {smoothed_se3.shape}")
            
            # Arc-length 재샘플링 (옵션) - 일단 비활성화
            # if len(smoothed_se3) > 10:
            #     try:
            #         resampled_se3 = traj_resample_by_arclength(
            #             smoothed_se3, 
            #             num_samples=len(raw_traj)
            #         )
            #         smoothed_se3 = resampled_se3
            #     except Exception as e:
            #         if self.verbose:
            #             print(f"      ⚠️ 재샘플링 실패, 원본 사용: {e}")
            
                        # 지정된 형식으로 변환
            smoothed_traj = self._convert_trajectory_format(smoothed_se3, self.output_format)
            
            if self.verbose:
                print(f"      🔧 {self.output_format} 변환 완료: {smoothed_traj.shape}")
            
            # 통계 계산
            stats = {
                'method': 'SE3_bspline_slerp',
                'original_points': len(raw_traj),
                'smoothed_points': len(smoothed_traj),
                'success': True
            }
            
            return smoothed_traj, stats
            
        except Exception as e:
            error_msg = f"SE(3) 스무딩 실패: {e}"
            if self.verbose:
                print(f"      ❌ {error_msg}")
                import traceback
                traceback.print_exc()
            
            return None, {
                'method': 'SE3_bspline_slerp',
                'original_points': len(raw_traj) if raw_traj is not None else 0,
                'smoothed_points': 0,
                'success': False,
                'error': error_msg
            }
    
    def _process_single_environment(self, env_group_in: h5py.Group, f_out: h5py.File, env_name: str):
        """단일 환경 처리"""
        self._monitor_memory(f"{env_name} 시작 전")
        
        # 출력 환경 그룹 생성
        env_group_out = f_out.create_group(env_name)
        
        try:
            pair_ids = list(env_group_in.keys())
        except Exception as e:
            print(f"  ❌ {env_name}: 페어 목록 로드 실패 - {e}")
            return
        
        successful_pairs = 0
        failed_pairs = 0
        
        for pair_id in pair_ids:
            try:
                pair_group_in = env_group_in[pair_id]
            except Exception as e:
                print(f"  ❌ {env_name}/{pair_id}: 페어 그룹 접근 실패 - {e}")
                failed_pairs += 1
                continue
            
            if 'raw_trajectory' not in pair_group_in:
                print(f"  ⚠️ {env_name}/{pair_id}: raw_trajectory 없음")
                failed_pairs += 1
                continue
            
            try:
                # Raw trajectory 로드
                raw_trajectory = pair_group_in['raw_trajectory'][:]
                
                # 출력 페어 그룹 생성
                pair_group_out = env_group_out.create_group(pair_id)
                
                # Raw trajectory 복사
                pair_group_out.create_dataset('raw_trajectory', data=raw_trajectory, compression='gzip')
                
                # 기존 속성 복사
                for key, value in pair_group_in.attrs.items():
                    pair_group_out.attrs[key] = value
                
                # SE(3) 스무딩 수행
                smooth_trajectory, smooth_stats = self._se3_smooth_trajectory(raw_trajectory)
                
                if smooth_trajectory is not None:
                    # 스무딩된 궤적 저장
                    pair_group_out.create_dataset('smooth_trajectory', data=smooth_trajectory, compression='gzip')
                    
                    # 스무딩 메타데이터 추가
                    pair_group_out.attrs['smooth_method'] = smooth_stats['method']
                    pair_group_out.attrs['smooth_success'] = smooth_stats['success']
                    pair_group_out.attrs['smooth_points'] = smooth_stats['smoothed_points']
                    
                    successful_pairs += 1
                    if self.verbose:
                        print(f"  ✅ {env_name}/{pair_id}: 스무딩 완료 ({len(raw_trajectory)} → {len(smooth_trajectory)} 포인트)")
                else:
                    # 스무딩 실패 시에도 메타데이터 저장
                    pair_group_out.attrs['smooth_success'] = False
                    pair_group_out.attrs['smooth_error'] = smooth_stats.get('error', 'Unknown error')
                    failed_pairs += 1
                
                # 메모리 정리
                del raw_trajectory
                if smooth_trajectory is not None:
                    del smooth_trajectory
                
            except Exception as e:
                print(f"  ❌ {env_name}/{pair_id}: 처리 실패 - {e}")
                failed_pairs += 1
        
        # 환경 레벨 통계 업데이트
        self.stats['total_trajectories'] += (successful_pairs + failed_pairs)
        self.stats['successful_smoothing'] += successful_pairs
        self.stats['failed_smoothing'] += failed_pairs
        
        print(f"  📊 {env_name}: {successful_pairs}/{successful_pairs + failed_pairs} 성공")
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_file(self) -> Dict[str, Any]:
        """전체 파일 처리"""
        start_time = time.time()
        
        try:
            with h5py.File(self.input_file, 'r') as f_in:
                with h5py.File(self.output_file, 'w') as f_out:
                    
                    # 메타데이터 복사
                    if 'metadata' in f_in:
                        f_in.copy('metadata', f_out)
                        print("✅ 메타데이터 복사 완료")
                    
                    if 'global_stats' in f_in:
                        f_in.copy('global_stats', f_out)
                        print("✅ 전역 통계 복사 완료")
                    
                    # 환경 목록 로드
                    env_names = [name for name in f_in.keys() 
                                if name not in ['metadata', 'global_stats']]
                    self.stats['total_environments'] = len(env_names)
                    
                    print(f"\n🎯 총 {len(env_names)}개 환경 처리 시작")
                    
                    # 청크 단위로 처리
                    for i in range(0, len(env_names), self.chunk_size):
                        env_chunk = env_names[i:i+self.chunk_size]
                        
                        print(f"\n📦 청크 {i//self.chunk_size + 1}/{(len(env_names)-1)//self.chunk_size + 1}: {len(env_chunk)}개 환경")
                        
                        for env_name in env_chunk:
                            try:
                                env_group_in = f_in[env_name]
                                self._process_single_environment(env_group_in, f_out, env_name)
                                self.stats['processed_environments'] += 1
                            except Exception as e:
                                print(f"  ❌ {env_name}: 환경 처리 실패 - {e}")
                        
                        # 청크 완료 후 메모리 정리
                        self._monitor_memory("청크 완료 후")
                        
                        # 진행률 출력
                        progress = min(i + self.chunk_size, len(env_names))
                        print(f"📊 진행률: {progress}/{len(env_names)} ({progress/len(env_names)*100:.1f}%)")
            
            # 처리 완료
            self.stats['processing_time'] = time.time() - start_time
            
            print(f"\n🎉 파일 처리 완료!")
            print(f"   입력 파일: {self.input_file}")
            print(f"   출력 파일: {self.output_file}")
            print(f"   처리된 환경: {self.stats['processed_environments']}/{self.stats['total_environments']}")
            print(f"   총 궤적: {self.stats['total_trajectories']}")
            print(f"   스무딩 성공: {self.stats['successful_smoothing']}")
            print(f"   스무딩 실패: {self.stats['failed_smoothing']}")
            
            if self.stats['total_trajectories'] > 0:
                success_rate = (self.stats['successful_smoothing'] / self.stats['total_trajectories']) * 100
                print(f"   성공률: {success_rate:.1f}%")
            
            print(f"   처리 시간: {self.stats['processing_time']:.2f}초")
            print(f"   메모리 경고: {self.stats['memory_warnings']}회")
            
            return self.stats
            
        except Exception as e:
            print(f"❌ 파일 처리 실패: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description='통합 궤적 스무딩 시스템')
    
    # 필수 인자
    parser.add_argument('--input', required=True, help='입력 HDF5 파일')
    parser.add_argument('--output', required=True, help='출력 HDF5 파일')
    
    # 옵션 인자
    parser.add_argument('--chunk-size', type=int, default=20, 
                       help='환경 청크 크기 (기본: 20)')
    parser.add_argument('--memory-threshold', type=float, default=75.0,
                       help='메모리 경고 임계값 %% (기본: 75.0)')
    
    # 출력 형식
    parser.add_argument('--output-format', type=str, default='se3_6d',
                       choices=['se2', 'se3', 'se3_6d', 'quaternion_7d'],
                       help='궤적 출력 형식 (기본: se3_6d)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='상세 출력')
    
    args = parser.parse_args()
    
    # 입력 파일 존재 확인
    if not Path(args.input).exists():
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return 1
    
    # 출력 파일 덮어쓰기 확인
    if Path(args.output).exists():
        response = input(f"⚠️ 출력 파일이 이미 존재합니다: {args.output}\n   덮어쓰시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("🚫 작업 취소됨")
            return 1
    
    try:
            # 스무더 생성 및 실행
    smoother = IntegratedTrajectorySmoother(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size,
        memory_threshold=args.memory_threshold,
        verbose=args.verbose,
        output_format=args.output_format
    )
        
        stats = smoother.process_file()
        
        if not stats:
            print("❌ 처리 실패")
            return 1
        
        print("\n✅ 스무딩 완료!")
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)