#!/usr/bin/env python3
"""
🚀 Tdot Trajectory Generation Pipeline
스무딩된 SE(3) 궤적에서 속도(Tdot) 궤적을 생성하는 파이프라인
- 균등 시간 할당 (추후 곡률 기반 할당 추가 가능)
- 각 waypoint에 대응되는 Tdot 계산
- HDF5 형식으로 저장

사용법:
    python generate_tdot_trajectories.py --input circles_only_integrated_trajs.h5 --dt 0.01
    python generate_tdot_trajectories.py --input circles_only_integrated_trajs.h5 --time-policy uniform --dt 0.01
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
from dataclasses import dataclass

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'packages'))

# SE(3) 함수 import
from packages.utils.SE3_functions import (
    log_SO3,
    Tdot_to_Vb, Vb_to_Tdot,
    traj_dt_from_length,
    _se3_exp, _se3_log, _so3_exp, _so3_hat
)


@dataclass
class TdotConfig:
    """Tdot 생성 설정"""
    time_policy: str = 'uniform'  # 'uniform' or 'curvature'
    dt: float = 0.01  # 기본 시간 간격 (uniform policy)
    v_ref: float = 0.4  # 참조 속도 (m/s)
    v_cap: float = 0.5  # 최대 속도 (m/s)
    a_lat_max: float = 1.0  # 최대 횡가속도 (m/s²)
    save_as_4x4: bool = True  # True: [N,4,4], False: [N,6]


class TdotTrajectoryGenerator:
    """
    Tdot 궤적 생성 시스템
    - 스무딩된 SE(3) 궤적에서 속도 계산
    - 균등/곡률 기반 시간 할당
    - HDF5 형식으로 저장
    """
    
    def __init__(self, input_file: str, config: TdotConfig = None, 
                 chunk_size: int = 20, memory_threshold: float = 75.0, verbose: bool = False):
        """
        Args:
            input_file: 스무딩된 궤적이 포함된 입력 HDF5 파일
            config: Tdot 생성 설정
            chunk_size: 환경 청크 크기
            memory_threshold: 메모리 경고 임계값 (%)
            verbose: 상세 출력
        """
        self.input_file = Path(input_file)
        
        # 출력 파일 경로 설정 (root/data/Tdot/)
        input_name = self.input_file.stem  # 확장자 제외 파일명
        tdot_dir = project_root / 'data' / 'Tdot'
        tdot_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = tdot_dir / f"{input_name}_Tdot.h5"
        
        self.config = config or TdotConfig()
        self.chunk_size = chunk_size
        self.memory_threshold = memory_threshold
        self.verbose = verbose
        
        # 통계
        self.stats = {
            'total_environments': 0,
            'processed_environments': 0,
            'total_trajectories': 0,
            'successful_tdot': 0,
            'failed_tdot': 0,
            'processing_time': 0.0,
            'memory_warnings': 0
        }
        
        print(f"🚀 Tdot 궤적 생성 시스템 초기화")
        print(f"   입력: {self.input_file}")
        print(f"   출력: {self.output_file}")
        print(f"   시간 정책: {self.config.time_policy}")
        print(f"   dt: {self.config.dt}s")
        print(f"   저장 형식: {'[N,4,4]' if self.config.save_as_4x4 else '[N,6]'}")
    
    def _get_memory_usage(self) -> Tuple[float, float, float]:
        """현재 메모리 사용량 반환 (MB, %, GPU MB)"""
        process = psutil.Process()
        cpu_mb = process.memory_info().rss / 1024 / 1024
        system_percent = psutil.virtual_memory().percent
        
        gpu_mb = 0.0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        return cpu_mb, system_percent, gpu_mb
    
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
    
    def _se3_6d_to_matrix(self, pose_6d: np.ndarray) -> torch.Tensor:
        """SE(3) 6D 표현을 4x4 행렬로 변환"""
        if len(pose_6d.shape) == 1:
            pose_6d = pose_6d.reshape(1, -1)
        
        N = pose_6d.shape[0]
        T_matrices = torch.zeros(N, 4, 4, dtype=torch.float32)
        
        for i in range(N):
            x, y, z, rx, ry, rz = pose_6d[i]
            
            # 회전 행렬 생성 (오일러각 → 회전행렬)
            w = torch.tensor([rx, ry, rz], dtype=torch.float32)
            R = _so3_exp(w)
            
            # SE(3) 행렬 구성
            T_matrices[i, :3, :3] = R
            T_matrices[i, 0, 3] = x
            T_matrices[i, 1, 3] = y
            T_matrices[i, 2, 3] = z
            T_matrices[i, 3, 3] = 1.0
        
        return T_matrices
    
    def _compute_tdot_from_trajectory(self, smooth_traj: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        스무딩된 궤적에서 Tdot 계산
        
        Args:
            smooth_traj: 스무딩된 궤적 [N, 6] (x,y,z,rx,ry,rz)
            
        Returns:
            (Tdot_traj, stats): Tdot 궤적과 통계
        """
        try:
            if len(smooth_traj) < 2:
                raise ValueError(f"궤적 포인트 부족: {len(smooth_traj)}개")
            
            # SE(3) 6D → 4x4 행렬 변환
            T_matrices = self._se3_6d_to_matrix(smooth_traj)
            
            if self.verbose:
                print(f"      🔧 SE(3) 행렬 변환 완료: {T_matrices.shape}")
            
            N = len(T_matrices)
            
            if self.config.time_policy == 'uniform':
                # 균등 시간 할당
                dt_seq = torch.full((N-1,), self.config.dt, dtype=torch.float32)
            elif self.config.time_policy == 'curvature':
                # 곡률 기반 시간 할당
                dt_seq = traj_dt_from_length(
                    T_matrices,
                    policy='curvature',
                    v_ref=self.config.v_ref,
                    v_cap=self.config.v_cap,
                    a_lat_max=self.config.a_lat_max
                )
            else:
                raise ValueError(f"Unknown time policy: {self.config.time_policy}")
            
            # Tdot 계산: T_dot = T * (T^{-1} * T_next)
            Tdot_list = []
            
            for i in range(N-1):
                T_curr = T_matrices[i]
                T_next = T_matrices[i+1]
                dt = dt_seq[i].item()
                
                # Relative transformation
                T_rel = torch.linalg.inv(T_curr) @ T_next
                
                # Log mapping to get body twist
                xi = _se3_log(T_rel.unsqueeze(0)).squeeze(0)  # [6]
                
                # Scale by time to get velocity
                xi_vel = xi / dt
                
                # Convert to Tdot = T * skew(xi)
                xi_skew = torch.zeros(4, 4, dtype=torch.float32)
                xi_skew[:3, :3] = _so3_hat(xi_vel[:3])
                xi_skew[:3, 3] = xi_vel[3:]
                
                Tdot = T_curr @ xi_skew
                
                if self.config.save_as_4x4:
                    Tdot_list.append(Tdot.numpy())
                else:
                    # Extract only the velocity part [wx,wy,wz,vx,vy,vz]
                    Vb = Tdot_to_Vb(Tdot.unsqueeze(0), T_curr.unsqueeze(0)).squeeze(0)
                    Vb_vec = torch.cat([
                        torch.tensor([Vb[2,1] - Vb[1,2], Vb[0,2] - Vb[2,0], Vb[1,0] - Vb[0,1]]) * 0.5,
                        Vb[:3, 3]
                    ])
                    Tdot_list.append(Vb_vec.numpy())
            
            # 마지막 waypoint는 속도 0
            if self.config.save_as_4x4:
                Tdot_list.append(np.zeros((4, 4), dtype=np.float32))
            else:
                Tdot_list.append(np.zeros(6, dtype=np.float32))
            
            Tdot_traj = np.stack(Tdot_list)
            
            if self.verbose:
                print(f"      ✅ Tdot 계산 완료: {Tdot_traj.shape}")
            
            stats = {
                'method': self.config.time_policy,
                'dt_mean': float(dt_seq.mean().item()) if len(dt_seq) > 0 else self.config.dt,
                'dt_std': float(dt_seq.std().item()) if len(dt_seq) > 0 else 0.0,
                'original_points': len(smooth_traj),
                'tdot_points': len(Tdot_traj),
                'success': True
            }
            
            return Tdot_traj, stats
            
        except Exception as e:
            error_msg = f"Tdot 계산 실패: {e}"
            if self.verbose:
                print(f"      ❌ {error_msg}")
                import traceback
                traceback.print_exc()
            
            return None, {
                'method': self.config.time_policy,
                'original_points': len(smooth_traj) if smooth_traj is not None else 0,
                'tdot_points': 0,
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
            
            # smooth_trajectory 우선, 없으면 raw_trajectory 사용
            traj_key = 'smooth_trajectory' if 'smooth_trajectory' in pair_group_in else 'raw_trajectory'
            
            if traj_key not in pair_group_in:
                print(f"  ⚠️ {env_name}/{pair_id}: 궤적 데이터 없음")
                failed_pairs += 1
                continue
            
            try:
                # 궤적 로드
                trajectory = pair_group_in[traj_key][:]
                
                # 출력 페어 그룹 생성
                pair_group_out = env_group_out.create_group(pair_id)
                
                # 기존 데이터 복사
                for key in pair_group_in.keys():
                    if key != 'Tdot_trajectory':  # Tdot는 새로 생성
                        pair_group_out.create_dataset(key, data=pair_group_in[key][:], compression='gzip')
                
                # 기존 속성 복사
                for key, value in pair_group_in.attrs.items():
                    pair_group_out.attrs[key] = value
                
                # Tdot 계산
                Tdot_traj, tdot_stats = self._compute_tdot_from_trajectory(trajectory)
                
                if Tdot_traj is not None:
                    # Tdot 궤적 저장
                    pair_group_out.create_dataset('Tdot_trajectory', data=Tdot_traj, compression='gzip')
                    
                    # Tdot 메타데이터 추가
                    pair_group_out.attrs['tdot_method'] = tdot_stats['method']
                    pair_group_out.attrs['tdot_success'] = tdot_stats['success']
                    pair_group_out.attrs['tdot_points'] = tdot_stats['tdot_points']
                    pair_group_out.attrs['tdot_dt_mean'] = tdot_stats['dt_mean']
                    pair_group_out.attrs['tdot_dt_std'] = tdot_stats['dt_std']
                    pair_group_out.attrs['tdot_format'] = '4x4' if self.config.save_as_4x4 else '6d'
                    
                    successful_pairs += 1
                    if self.verbose:
                        print(f"  ✅ {env_name}/{pair_id}: Tdot 생성 완료 ({len(trajectory)} → {len(Tdot_traj)} 포인트)")
                else:
                    # Tdot 실패 시 메타데이터
                    pair_group_out.attrs['tdot_success'] = False
                    pair_group_out.attrs['tdot_error'] = tdot_stats.get('error', 'Unknown error')
                    failed_pairs += 1
                
                # 메모리 정리
                del trajectory
                if Tdot_traj is not None:
                    del Tdot_traj
                
            except Exception as e:
                print(f"  ❌ {env_name}/{pair_id}: 처리 실패 - {e}")
                failed_pairs += 1
        
        # 환경 레벨 통계 업데이트
        self.stats['total_trajectories'] += (successful_pairs + failed_pairs)
        self.stats['successful_tdot'] += successful_pairs
        self.stats['failed_tdot'] += failed_pairs
        
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
                    
                    # 메타데이터 복사 및 업데이트
                    if 'metadata' in f_in:
                        metadata_group = f_out.create_group('metadata')
                        for key, value in f_in['metadata'].attrs.items():
                            metadata_group.attrs[key] = value
                        
                        # Tdot 관련 메타데이터 추가
                        metadata_group.attrs['tdot_generation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                        metadata_group.attrs['tdot_time_policy'] = self.config.time_policy
                        metadata_group.attrs['tdot_dt'] = self.config.dt
                        metadata_group.attrs['tdot_format'] = '4x4' if self.config.save_as_4x4 else '6d'
                        
                        print("✅ 메타데이터 복사 및 업데이트 완료")
                    
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
            
            print(f"\n🎉 Tdot 생성 완료!")
            print(f"   입력 파일: {self.input_file}")
            print(f"   출력 파일: {self.output_file}")
            print(f"   처리된 환경: {self.stats['processed_environments']}/{self.stats['total_environments']}")
            print(f"   총 궤적: {self.stats['total_trajectories']}")
            print(f"   Tdot 성공: {self.stats['successful_tdot']}")
            print(f"   Tdot 실패: {self.stats['failed_tdot']}")
            
            if self.stats['total_trajectories'] > 0:
                success_rate = (self.stats['successful_tdot'] / self.stats['total_trajectories']) * 100
                print(f"   성공률: {success_rate:.1f}%")
            
            print(f"   처리 시간: {self.stats['processing_time']:.2f}초")
            print(f"   메모리 경고: {self.stats['memory_warnings']}회")
            
            return self.stats
            
        except Exception as e:
            print(f"❌ 파일 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    parser = argparse.ArgumentParser(description='Tdot 궤적 생성 시스템')
    
    # 필수 인자
    parser.add_argument('--input', required=True, help='입력 HDF5 파일 (스무딩된 궤적)')
    
    # 시간 정책 설정
    parser.add_argument('--time-policy', type=str, default='uniform',
                       choices=['uniform', 'curvature'],
                       help='시간 할당 정책 (기본: uniform)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='시간 간격 (uniform policy, 기본: 0.01s)')
    
    # 곡률 기반 정책 파라미터
    parser.add_argument('--v-ref', type=float, default=0.4,
                       help='참조 속도 m/s (curvature policy, 기본: 0.4)')
    parser.add_argument('--v-cap', type=float, default=0.5,
                       help='최대 속도 m/s (curvature policy, 기본: 0.5)')
    parser.add_argument('--a-lat-max', type=float, default=1.0,
                       help='최대 횡가속도 m/s² (curvature policy, 기본: 1.0)')
    
    # 저장 형식
    parser.add_argument('--save-format', type=str, default='4x4',
                       choices=['4x4', '6d'],
                       help='Tdot 저장 형식 (기본: 4x4)')
    
    # 처리 옵션
    parser.add_argument('--chunk-size', type=int, default=20,
                       help='환경 청크 크기 (기본: 20)')
    parser.add_argument('--memory-threshold', type=float, default=75.0,
                       help='메모리 경고 임계값 %% (기본: 75.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='상세 출력')
    
    args = parser.parse_args()
    
    # 입력 파일 존재 확인
    if not Path(args.input).exists():
        # data/trajectory 디렉토리에서 찾기
        alt_path = project_root / 'data' / 'trajectory' / args.input
        if alt_path.exists():
            args.input = str(alt_path)
        else:
            print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
            return 1
    
    try:
        # 설정 생성
        config = TdotConfig(
            time_policy=args.time_policy,
            dt=args.dt,
            v_ref=args.v_ref,
            v_cap=args.v_cap,
            a_lat_max=args.a_lat_max,
            save_as_4x4=(args.save_format == '4x4')
        )
        
        # 생성기 생성 및 실행
        generator = TdotTrajectoryGenerator(
            input_file=args.input,
            config=config,
            chunk_size=args.chunk_size,
            memory_threshold=args.memory_threshold,
            verbose=args.verbose
        )
        
        stats = generator.process_file()
        
        if not stats:
            print("❌ 처리 실패")
            return 1
        
        print("\n✅ Tdot 생성 완료!")
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)