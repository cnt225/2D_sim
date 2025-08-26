#!/usr/bin/env python3
"""
통합 메타데이터 관리자 (HDF5 기반)
환경별 메타데이터를 하나의 HDF5 파일로 통합 관리
"""

import h5py
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
import os
from pathlib import Path
import time
from datetime import datetime


class UnifiedMetadataManager:
    """
    HDF5 기반 통합 메타데이터 관리자
    
    구조:
    unified_metadata.h5
    ├── environments/
    │   ├── env_000000/
    │   │   ├── attributes (스칼라 메타데이터)
    │   │   ├── obstacles (장애물 정보)
    │   │   └── generation_info (생성 정보)
    │   └── env_000001/
    │       └── ...
    ├── generation_history/
    │   ├── batch_000 (배치별 생성 기록)
    │   └── batch_001
    └── summary/
        ├── total_environments
        ├── generation_stats
        └── last_updated
    """
    
    def __init__(self, h5_path: str):
        """
        Args:
            h5_path: HDF5 파일 경로
        """
        self.h5_path = Path(h5_path)
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 초기화
        self._initialize_file()
    
    def _initialize_file(self):
        """HDF5 파일 초기화"""
        with h5py.File(self.h5_path, 'a') as f:
            # 기본 그룹 생성
            if 'environments' not in f:
                f.create_group('environments')
            if 'generation_history' not in f:
                f.create_group('generation_history')
            if 'summary' not in f:
                summary = f.create_group('summary')
                summary.attrs['total_environments'] = 0
                summary.attrs['creation_time'] = datetime.now().isoformat()
                summary.attrs['last_updated'] = datetime.now().isoformat()
    
    def add_environment(self, env_id: str, metadata: Dict[str, Any]) -> bool:
        """
        환경 메타데이터 추가
        
        Args:
            env_id: 환경 ID (예: "000000")
            metadata: 환경 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            with h5py.File(self.h5_path, 'a') as f:
                env_group_name = f'environments/env_{env_id}'
                
                # 기존 환경이 있으면 삭제
                if env_group_name in f:
                    del f[env_group_name]
                
                # 새 환경 그룹 생성
                env_group = f.create_group(env_group_name)
                
                # 스칼라 속성 저장
                scalar_keys = ['env_type', 'resolution', 'noise_level', 'clustering_eps', 
                              'min_samples', 'obstacle_type', 'num_points', 'num_obstacles']
                
                for key in scalar_keys:
                    if key in metadata:
                        value = metadata[key]
                        if isinstance(value, (str, int, float, bool)):
                            env_group.attrs[key] = value
                
                # 워크스페이스 경계 저장 (배열)
                if 'workspace_bounds' in metadata:
                    env_group.create_dataset('workspace_bounds', 
                                           data=np.array(metadata['workspace_bounds']))
                
                # 장애물 정보 저장
                if 'environment_details' in metadata and 'obstacles' in metadata['environment_details']:
                    obstacles = metadata['environment_details']['obstacles']
                    
                    # 장애물별로 서브그룹 생성
                    obstacles_group = env_group.create_group('obstacles')
                    
                    for i, obstacle in enumerate(obstacles):
                        obs_group = obstacles_group.create_group(f'obstacle_{i:03d}')
                        
                        # 장애물 속성
                        for key, value in obstacle.items():
                            if key == 'position':
                                obs_group.create_dataset('position', data=np.array(value))
                            elif isinstance(value, (str, int, float, bool)):
                                obs_group.attrs[key] = value
                
                # 생성 정보 저장
                generation_info = env_group.create_group('generation_info')
                generation_info.attrs['creation_time'] = datetime.now().isoformat()
                generation_info.attrs['generator_version'] = '2.0_improved'
                
                if 'difficulty' in metadata:
                    generation_info.attrs['difficulty'] = metadata['difficulty']
                if 'seed' in metadata:
                    generation_info.attrs['seed'] = metadata['seed']
                
                # 전체 환경 수 업데이트
                f['summary'].attrs['total_environments'] = len(f['environments'].keys())
                f['summary'].attrs['last_updated'] = datetime.now().isoformat()
                
                print(f"✅ Environment {env_id} added to unified metadata")
                return True
                
        except Exception as e:
            print(f"❌ Failed to add environment {env_id}: {e}")
            return False
    
    def get_environment(self, env_id: str) -> Optional[Dict[str, Any]]:
        """
        환경 메타데이터 조회
        
        Args:
            env_id: 환경 ID
            
        Returns:
            Dict: 환경 메타데이터 또는 None
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                env_group_name = f'environments/env_{env_id}'
                
                if env_group_name not in f:
                    return None
                
                env_group = f[env_group_name]
                metadata = {}
                
                # 스칼라 속성 읽기
                for key, value in env_group.attrs.items():
                    metadata[key] = value
                
                # 배열 데이터 읽기
                if 'workspace_bounds' in env_group:
                    metadata['workspace_bounds'] = env_group['workspace_bounds'][:].tolist()
                
                # 장애물 정보 읽기
                if 'obstacles' in env_group:
                    obstacles = []
                    obstacles_group = env_group['obstacles']
                    
                    for obs_name in sorted(obstacles_group.keys()):
                        obs_group = obstacles_group[obs_name]
                        obstacle = {}
                        
                        # 장애물 속성
                        for key, value in obs_group.attrs.items():
                            obstacle[key] = value
                        
                        # 위치 정보
                        if 'position' in obs_group:
                            obstacle['position'] = obs_group['position'][:].tolist()
                        
                        obstacles.append(obstacle)
                    
                    metadata['environment_details'] = {'obstacles': obstacles}
                
                # 생성 정보 읽기
                if 'generation_info' in env_group:
                    gen_info = {}
                    gen_group = env_group['generation_info']
                    for key, value in gen_group.attrs.items():
                        gen_info[key] = value
                    metadata['generation_info'] = gen_info
                
                return metadata
                
        except Exception as e:
            print(f"❌ Failed to get environment {env_id}: {e}")
            return None
    
    def list_environments(self) -> List[str]:
        """환경 ID 목록 반환"""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'environments' not in f:
                    return []
                
                env_ids = []
                for env_name in f['environments'].keys():
                    if env_name.startswith('env_'):
                        env_id = env_name[4:]  # 'env_' 제거
                        env_ids.append(env_id)
                
                return sorted(env_ids)
                
        except Exception as e:
            print(f"❌ Failed to list environments: {e}")
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 요약 정보 반환"""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'summary' not in f:
                    return {}
                
                summary = {}
                for key, value in f['summary'].attrs.items():
                    summary[key] = value
                
                # 환경별 통계
                environments = self.list_environments()
                summary['environment_ids'] = environments
                summary['actual_count'] = len(environments)
                
                return summary
                
        except Exception as e:
            print(f"❌ Failed to get summary: {e}")
            return {}
    
    def add_batch_record(self, batch_info: Dict[str, Any]) -> bool:
        """배치 생성 기록 추가"""
        try:
            with h5py.File(self.h5_path, 'a') as f:
                history_group = f['generation_history']
                
                # 배치 번호 계산
                existing_batches = [key for key in history_group.keys() if key.startswith('batch_')]
                batch_num = len(existing_batches)
                
                batch_group = history_group.create_group(f'batch_{batch_num:03d}')
                
                # 배치 정보 저장
                for key, value in batch_info.items():
                    if isinstance(value, (str, int, float, bool)):
                        batch_group.attrs[key] = value
                    elif isinstance(value, list) and all(isinstance(x, (str, int, float)) for x in value):
                        batch_group.create_dataset(key, data=np.array(value))
                
                batch_group.attrs['timestamp'] = datetime.now().isoformat()
                
                print(f"✅ Batch {batch_num} record added")
                return True
                
        except Exception as e:
            print(f"❌ Failed to add batch record: {e}")
            return False
    
    def migrate_from_json(self, json_dir: str) -> int:
        """기존 JSON 메타데이터를 HDF5로 마이그레이션"""
        json_path = Path(json_dir)
        migrated_count = 0
        
        print(f"🔄 Migrating JSON metadata from {json_path}")
        
        for json_file in json_path.glob("*_meta.json"):
            try:
                # 환경 ID 추출
                env_id = json_file.stem.replace('circle_env_', '').replace('_meta', '')
                
                # JSON 데이터 로드
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # HDF5에 추가
                if self.add_environment(env_id, metadata):
                    migrated_count += 1
                    
            except Exception as e:
                print(f"❌ Failed to migrate {json_file}: {e}")
        
        print(f"✅ Migrated {migrated_count} environments from JSON to HDF5")
        return migrated_count


def test_unified_metadata():
    """통합 메타데이터 시스템 테스트"""
    print("🧪 Testing Unified Metadata Manager")
    
    # 테스트 HDF5 파일 생성
    test_path = "/tmp/test_unified_metadata.h5"
    if os.path.exists(test_path):
        os.remove(test_path)
    
    manager = UnifiedMetadataManager(test_path)
    
    # 테스트 데이터
    test_metadata = {
        "env_type": "circles",
        "resolution": 0.05,
        "noise_level": 0.01,
        "workspace_bounds": [-1, 11, -1, 11],
        "clustering_eps": 0.3,
        "min_samples": 5,
        "obstacle_type": "auto",
        "num_points": 1000,
        "num_obstacles": 5,
        "difficulty": "medium",
        "seed": 12345,
        "environment_details": {
            "obstacles": [
                {"id": 0, "position": [2.5, 3.0], "radius": 0.5, "type": "circle"},
                {"id": 1, "position": [7.0, 8.5], "radius": 0.3, "type": "circle"}
            ]
        }
    }
    
    # 환경 추가 테스트
    success = manager.add_environment("000001", test_metadata)
    print(f"Environment addition: {'✅' if success else '❌'}")
    
    # 환경 조회 테스트
    retrieved = manager.get_environment("000001")
    print(f"Environment retrieval: {'✅' if retrieved else '❌'}")
    
    # 요약 정보 테스트
    summary = manager.get_summary()
    print(f"Summary: {summary}")
    
    # 정리
    os.remove(test_path)
    print("🎉 Test completed")


if __name__ == "__main__":
    test_unified_metadata()
