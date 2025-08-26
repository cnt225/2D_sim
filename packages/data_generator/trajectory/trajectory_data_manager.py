#!/usr/bin/env python3
"""
궤적 데이터 HDF5 관리 클래스
환경별 궤적 데이터를 체계적으로 관리하는 시스템

HDF5 구조:
{env_name}_trajs.h5
├── metadata/
│   ├── environment_info          # 환경 기본 정보
│   ├── generation_config         # 생성 설정
│   └── summary_stats             # 통계 정보  
├── pose_pairs/
│   ├── pair_000001/
│   │   ├── metadata              # pose pair 정보
│   │   ├── raw_trajectory        # RRT 원본 궤적
│   │   ├── smooth_trajectory     # B-spline 스무딩 궤적
│   │   └── validation_results    # 충돌 검증 결과
│   └── ...
└── global_stats/                 # 전체 통계
"""

import h5py
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class PosePairMetadata:
    """Pose pair 메타데이터"""
    start_pose: List[float]  # [x, y, theta]
    end_pose: List[float]    # [x, y, theta]
    generation_method: str   # 'rrt_connect'
    smoothing_method: str    # 'bspline', 'sperl', 'none'
    collision_free: bool
    path_length: float
    generation_time: float
    smoothing_time: float = 0.0
    validation_time: float = 0.0


@dataclass 
class EnvironmentInfo:
    """환경 정보"""
    name: str
    type: str               # 'circle', 'random', etc.
    pointcloud_file: str
    workspace_bounds: List[float]  # [x_min, x_max, y_min, y_max]
    creation_timestamp: str


@dataclass
class GenerationConfig:
    """궤적 생성 설정"""
    rigid_body_id: int
    safety_margin: float
    rrt_range: float
    rrt_max_time: float
    bspline_degree: int
    bspline_smoothing_factor: float
    validation_enabled: bool


class TrajectoryDataManager:
    """궤적 데이터 HDF5 관리 클래스"""
    
    def __init__(self, env_name: str, base_dir: str = "/home/dhkang225/2D_sim/data/trajectories"):
        """
        Args:
            env_name: 환경 이름 (예: 'circle_env_000001')
            base_dir: 기본 저장 디렉토리
        """
        self.env_name = env_name
        self.base_dir = Path(base_dir)
        
        # 환경별 디렉토리 생성
        self.env_dir = self.base_dir / env_name
        self.env_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 파일 경로
        self.h5_file_path = self.env_dir / f"{env_name}_trajs.h5"
        
        # 내부 상태
        self._env_info: Optional[EnvironmentInfo] = None
        self._gen_config: Optional[GenerationConfig] = None
        self._pose_pair_count = 0
        
        print(f"✅ TrajectoryDataManager 초기화 완료")
        print(f"   환경: {env_name}")
        print(f"   HDF5 파일: {self.h5_file_path}")
    
    def initialize_h5_file(self, env_info: EnvironmentInfo, gen_config: GenerationConfig) -> bool:
        """HDF5 파일 초기화"""
        try:
            with h5py.File(self.h5_file_path, 'w') as f:
                # 메타데이터 그룹 생성
                metadata_group = f.create_group('metadata')
                
                # 환경 정보 저장
                env_group = metadata_group.create_group('environment_info')
                for key, value in asdict(env_info).items():
                    if isinstance(value, (str, int, float)):
                        env_group.attrs[key] = value
                    elif isinstance(value, list):
                        env_group.create_dataset(key, data=np.array(value))
                
                # 생성 설정 저장
                config_group = metadata_group.create_group('generation_config')
                for key, value in asdict(gen_config).items():
                    config_group.attrs[key] = value
                
                # 요약 통계 초기화
                stats_group = metadata_group.create_group('summary_stats')
                stats_group.attrs['total_pairs'] = 0
                stats_group.attrs['successful_pairs'] = 0
                stats_group.attrs['collision_free_pairs'] = 0
                stats_group.attrs['avg_path_length'] = 0.0
                stats_group.attrs['avg_generation_time'] = 0.0
                stats_group.attrs['last_updated'] = datetime.now().isoformat()
                
                # pose_pairs 그룹 생성
                f.create_group('pose_pairs')
                
                # global_stats 그룹 생성
                f.create_group('global_stats')
            
            self._env_info = env_info
            self._gen_config = gen_config
            
            print(f"✅ HDF5 파일 초기화 완료: {self.h5_file_path}")
            return True
            
        except Exception as e:
            print(f"❌ HDF5 파일 초기화 실패: {e}")
            return False
    
    def add_pose_pair(self, 
                     pair_id: str,
                     metadata: PosePairMetadata,
                     raw_trajectory: np.ndarray,
                     smooth_trajectory: Optional[np.ndarray] = None,
                     validation_results: Optional[Dict[str, Any]] = None) -> bool:
        """새로운 pose pair 추가"""
        try:
            with h5py.File(self.h5_file_path, 'a') as f:
                pose_pairs_group = f['pose_pairs']
                
                # pair 그룹 생성
                pair_group = pose_pairs_group.create_group(pair_id)
                
                # 메타데이터 저장
                meta_group = pair_group.create_group('metadata')
                for key, value in asdict(metadata).items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, list):
                        meta_group.create_dataset(key, data=np.array(value))
                
                # Raw 궤적 저장
                pair_group.create_dataset('raw_trajectory', data=raw_trajectory, 
                                        compression='gzip', compression_opts=9)
                
                # 스무딩된 궤적 저장 (있는 경우)
                if smooth_trajectory is not None:
                    pair_group.create_dataset('smooth_trajectory', data=smooth_trajectory,
                                            compression='gzip', compression_opts=9)
                
                # 검증 결과 저장 (있는 경우)
                if validation_results is not None:
                    val_group = pair_group.create_group('validation_results')
                    for key, value in validation_results.items():
                        if isinstance(value, (str, int, float, bool)):
                            val_group.attrs[key] = value
                        elif isinstance(value, (list, np.ndarray)):
                            val_group.create_dataset(key, data=np.array(value))
                
                # 요약 통계 업데이트
                self._update_summary_stats(f, metadata)
            
            self._pose_pair_count += 1
            print(f"✅ Pose pair 추가 완료: {pair_id}")
            return True
            
        except Exception as e:
            print(f"❌ Pose pair 추가 실패 ({pair_id}): {e}")
            return False
    
    def get_pose_pair(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """특정 pose pair 데이터 조회"""
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                if pair_id not in f['pose_pairs']:
                    return None
                
                pair_group = f['pose_pairs'][pair_id]
                
                # 메타데이터 로드
                meta_group = pair_group['metadata']
                metadata = {}
                for key in meta_group.attrs:
                    metadata[key] = meta_group.attrs[key]
                for key in meta_group.keys():
                    metadata[key] = meta_group[key][:]
                
                # 궤적 데이터 로드
                result = {
                    'metadata': metadata,
                    'raw_trajectory': pair_group['raw_trajectory'][:]
                }
                
                if 'smooth_trajectory' in pair_group:
                    result['smooth_trajectory'] = pair_group['smooth_trajectory'][:]
                
                if 'validation_results' in pair_group:
                    val_group = pair_group['validation_results']
                    validation = {}
                    for key in val_group.attrs:
                        validation[key] = val_group.attrs[key]
                    for key in val_group.keys():
                        validation[key] = val_group[key][:]
                    result['validation_results'] = validation
                
                return result
                
        except Exception as e:
            print(f"❌ Pose pair 조회 실패 ({pair_id}): {e}")
            return None
    
    def get_all_pair_ids(self) -> List[str]:
        """모든 pose pair ID 목록 조회"""
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                if 'pose_pairs' not in f:
                    return []
                return list(f['pose_pairs'].keys())
                
        except Exception as e:
            print(f"❌ Pose pair 목록 조회 실패: {e}")
            return []
    
    def update_pose_pair_smooth_trajectory(self, 
                                         pair_id: str,
                                         smooth_trajectory: np.ndarray,
                                         updated_metadata: PosePairMetadata) -> bool:
        """
        기존 pose pair의 smooth_trajectory와 메타데이터 업데이트
        
        Args:
            pair_id: 업데이트할 pair ID
            smooth_trajectory: 새로운 스무딩된 궤적
            updated_metadata: 업데이트된 메타데이터
            
        Returns:
            success: 업데이트 성공 여부
        """
        try:
            with h5py.File(self.h5_file_path, 'a') as f:
                if pair_id not in f['pose_pairs']:
                    print(f"❌ Pose pair 없음: {pair_id}")
                    return False
                
                pair_group = f['pose_pairs'][pair_id]
                
                # 1. smooth_trajectory 업데이트/추가
                if 'smooth_trajectory' in pair_group:
                    del pair_group['smooth_trajectory']
                
                pair_group.create_dataset('smooth_trajectory', data=smooth_trajectory,
                                        compression='gzip', compression_opts=6)
                
                # 2. 메타데이터 업데이트
                meta_group = pair_group['metadata']
                
                # 기존 메타데이터 유지하고 업데이트된 것만 변경
                meta_dict = asdict(updated_metadata)
                
                for key, value in meta_dict.items():
                    if isinstance(value, (list, np.ndarray)):
                        # 배열 데이터는 dataset으로 저장
                        if key in meta_group:
                            del meta_group[key]
                        meta_group.create_dataset(key, data=value)
                    else:
                        # 스칼라 데이터는 attribute로 저장
                        meta_group.attrs[key] = value
                
                # 3. 업데이트 시간 기록
                meta_group.attrs['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"✅ Smooth trajectory 업데이트 완료: {pair_id}")
            return True
            
        except Exception as e:
            print(f"❌ Smooth trajectory 업데이트 실패 ({pair_id}): {e}")
            return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """요약 통계 조회"""
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                if 'metadata/summary_stats' not in f:
                    return {}
                
                stats_group = f['metadata/summary_stats']
                stats = {}
                for key in stats_group.attrs:
                    stats[key] = stats_group.attrs[key]
                
                return stats
                
        except Exception as e:
            print(f"❌ 요약 통계 조회 실패: {e}")
            return {}
    
    def validate_trajectory(self, pair_id: str, collision_checker) -> bool:
        """궤적 충돌 검증 (나중에 구현)"""
        # TODO: 충돌 체커와 연동하여 궤적 검증
        return True
    
    def _update_summary_stats(self, h5_file, metadata: PosePairMetadata):
        """요약 통계 업데이트"""
        stats_group = h5_file['metadata/summary_stats']
        
        # 현재 통계 로드
        total_pairs = stats_group.attrs.get('total_pairs', 0)
        successful_pairs = stats_group.attrs.get('successful_pairs', 0)
        collision_free_pairs = stats_group.attrs.get('collision_free_pairs', 0)
        avg_path_length = stats_group.attrs.get('avg_path_length', 0.0)
        avg_generation_time = stats_group.attrs.get('avg_generation_time', 0.0)
        
        # 새로운 통계 계산
        total_pairs += 1
        successful_pairs += 1  # 추가된 것은 성공한 것
        if metadata.collision_free:
            collision_free_pairs += 1
        
        # 평균 계산 (온라인 알고리즘)
        avg_path_length = ((avg_path_length * (total_pairs - 1)) + metadata.path_length) / total_pairs
        avg_generation_time = ((avg_generation_time * (total_pairs - 1)) + metadata.generation_time) / total_pairs
        
        # 업데이트
        stats_group.attrs['total_pairs'] = total_pairs
        stats_group.attrs['successful_pairs'] = successful_pairs
        stats_group.attrs['collision_free_pairs'] = collision_free_pairs
        stats_group.attrs['avg_path_length'] = avg_path_length
        stats_group.attrs['avg_generation_time'] = avg_generation_time
        stats_group.attrs['last_updated'] = datetime.now().isoformat()
    
    def export_to_json(self, output_dir: Optional[Path] = None) -> bool:
        """HDF5 데이터를 JSON 형태로 내보내기 (호환성)"""
        if output_dir is None:
            output_dir = self.env_dir / "exported_json"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            pair_ids = self.get_all_pair_ids()
            
            for pair_id in pair_ids:
                pair_data = self.get_pose_pair(pair_id)
                if pair_data is None:
                    continue
                
                # JSON 호환 형태로 변환
                def convert_numpy_types(obj):
                    """numpy 타입을 JSON 호환 타입으로 변환"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    elif isinstance(obj, (np.integer, int)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, float)):
                        return float(obj)
                    else:
                        return obj
                
                json_data = {
                    'pair_id': pair_id,
                    'environment': {'name': self.env_name},
                    'metadata': {k: convert_numpy_types(v) for k, v in pair_data['metadata'].items()},
                    'path': {
                        'data': pair_data['raw_trajectory'].tolist(),
                        'timestamps': list(range(len(pair_data['raw_trajectory'])))
                    }
                }
                
                if 'smooth_trajectory' in pair_data:
                    json_data['smooth_path'] = {
                        'data': pair_data['smooth_trajectory'].tolist(),
                        'timestamps': list(range(len(pair_data['smooth_trajectory'])))
                    }
                
                # JSON 파일 저장
                json_file = output_dir / f"{pair_id}.json"
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            print(f"✅ JSON 내보내기 완료: {output_dir} ({len(pair_ids)}개 파일)")
            return True
            
        except Exception as e:
            print(f"❌ JSON 내보내기 실패: {e}")
            return False
    
    def print_info(self):
        """데이터 매니저 정보 출력"""
        print(f"\n📊 TrajectoryDataManager 정보")
        print(f"   환경: {self.env_name}")
        print(f"   HDF5 파일: {self.h5_file_path}")
        print(f"   파일 존재: {self.h5_file_path.exists()}")
        
        if self.h5_file_path.exists():
            stats = self.get_summary_stats()
            print(f"   총 pose pair 수: {stats.get('total_pairs', 0)}")
            print(f"   성공한 궤적 수: {stats.get('successful_pairs', 0)}")
            print(f"   충돌 없는 궤적 수: {stats.get('collision_free_pairs', 0)}")
            print(f"   평균 경로 길이: {stats.get('avg_path_length', 0.0):.3f}")
            print(f"   평균 생성 시간: {stats.get('avg_generation_time', 0.0):.3f}초")


# 유틸리티 함수들
def create_environment_info(env_name: str, env_type: str, pointcloud_file: str, 
                          workspace_bounds: List[float]) -> EnvironmentInfo:
    """환경 정보 생성 헬퍼"""
    return EnvironmentInfo(
        name=env_name,
        type=env_type,
        pointcloud_file=pointcloud_file,
        workspace_bounds=workspace_bounds,
        creation_timestamp=datetime.now().isoformat()
    )


def create_generation_config(rigid_body_id: int = 3, safety_margin: float = 0.05,
                           rrt_range: float = 0.5, rrt_max_time: float = 5.0,
                           bspline_degree: int = 3, bspline_smoothing: float = 0.0,
                           validation_enabled: bool = True) -> GenerationConfig:
    """생성 설정 생성 헬퍼"""
    return GenerationConfig(
        rigid_body_id=rigid_body_id,
        safety_margin=safety_margin,
        rrt_range=rrt_range,
        rrt_max_time=rrt_max_time,
        bspline_degree=bspline_degree,
        bspline_smoothing_factor=bspline_smoothing,
        validation_enabled=validation_enabled
    )


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 TrajectoryDataManager 테스트")
    
    # 테스트 환경 설정
    env_info = create_environment_info(
        env_name="test_circle_env_000001",
        env_type="circle",
        pointcloud_file="test_env.ply",
        workspace_bounds=[-5.0, 5.0, -5.0, 5.0]
    )
    
    gen_config = create_generation_config()
    
    # 데이터 매니저 생성
    manager = TrajectoryDataManager("test_circle_env_000001")
    
    # HDF5 파일 초기화
    success = manager.initialize_h5_file(env_info, gen_config)
    if success:
        print("✅ 테스트 성공: HDF5 파일 초기화")
    else:
        print("❌ 테스트 실패: HDF5 파일 초기화")
    
    # 정보 출력
    manager.print_info()
