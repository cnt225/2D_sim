#!/usr/bin/env python3
"""
통합 Pose 관리자 (HDF5 기반)
환경별 pose와 pose_pair 데이터를 하나의 HDF5 파일로 통합 관리

구조:
unified_poses.h5
├── environments/
│   └── {env_name}/                    # e.g., "circle_env_000000"
│       ├── poses/
│       │   └── rb_{id}/               # e.g., "rb_0", "rb_1", "rb_2"  
│       │       ├── data               # Dataset: (N, 6) [x,y,z,roll,pitch,yaw]
│       │       ├── attributes         # 메타데이터 (생성시간, 개수, 설정 등)
│       │       └── validation_info    # 충돌검사 결과, 성공률 등
│       └── pose_pairs/
│           └── rb_{id}/
│               ├── data               # Dataset: (M, 12) [init_pose, target_pose]
│               ├── attributes         # 쌍 생성 메타데이터
│               └── pair_metadata      # 각 쌍별 거리, 난이도 등
└── global_metadata/
    ├── generation_config              # 전역 생성 설정
    ├── rigid_body_configs            # RB별 설정 정보
    └── statistics                    # 전체 통계 정보
"""

import h5py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import time
from datetime import datetime


class UnifiedPoseManager:
    """
    HDF5 기반 통합 Pose 관리자
    
    환경별로 pose와 pose_pair 데이터를 효율적으로 관리
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
            if 'global_metadata' not in f:
                metadata_group = f.create_group('global_metadata')
                metadata_group.attrs['creation_time'] = datetime.now().isoformat()
                metadata_group.attrs['total_environments'] = 0
                metadata_group.attrs['last_updated'] = datetime.now().isoformat()
    
    def add_poses(self, env_name: str, rb_id: int, poses: np.ndarray, metadata: dict) -> bool:
        """
        환경-RB별 pose 데이터 추가
        
        Args:
            env_name: 환경 이름 (예: "circle_env_000000")
            rb_id: Rigid body ID (0, 1, 2, ...)
            poses: pose 배열 (N, 6) [x, y, z, roll, pitch, yaw]
            metadata: 생성 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            with h5py.File(self.h5_path, 'a') as f:
                # 환경 그룹 생성
                env_group_path = f'environments/{env_name}'
                if env_group_path not in f:
                    env_group = f.create_group(env_group_path)
                else:
                    env_group = f[env_group_path]
                
                # poses 그룹 생성
                if 'poses' not in env_group:
                    poses_group = env_group.create_group('poses')
                else:
                    poses_group = env_group['poses']
                
                # rigid body별 그룹 생성
                rb_group_name = f'rb_{rb_id}'
                if rb_group_name in poses_group:
                    # 기존 데이터 삭제
                    del poses_group[rb_group_name]
                
                rb_group = poses_group.create_group(rb_group_name)
                
                # pose 데이터 저장
                rb_group.create_dataset('data', data=poses, compression='gzip')
                
                # 메타데이터 저장
                rb_group.attrs['rb_id'] = rb_id
                rb_group.attrs['pose_count'] = len(poses)
                rb_group.attrs['creation_time'] = datetime.now().isoformat()
                rb_group.attrs['pose_format'] = 'se3_poses'
                rb_group.attrs['coordinate_system'] = 'world_frame'
                
                # 생성 정보 저장
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        rb_group.attrs[key] = value
                
                # 전체 환경 수 업데이트
                total_envs = len(f['environments'].keys())
                f['global_metadata'].attrs['total_environments'] = total_envs
                f['global_metadata'].attrs['last_updated'] = datetime.now().isoformat()
                
                print(f"✅ Added {len(poses)} poses for {env_name}/rb_{rb_id}")
                return True
                
        except Exception as e:
            print(f"❌ Failed to add poses for {env_name}/rb_{rb_id}: {e}")
            return False
    
    def add_pose_pairs(self, env_name: str, rb_id: int, pairs: np.ndarray, metadata: dict) -> bool:
        """
        환경-RB별 pose_pair 데이터 추가
        
        Args:
            env_name: 환경 이름
            rb_id: Rigid body ID
            pairs: pose_pair 배열 (M, 12) [init_pose + target_pose]
            metadata: 생성 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        try:
            with h5py.File(self.h5_path, 'a') as f:
                # 환경 그룹 확인
                env_group_path = f'environments/{env_name}'
                if env_group_path not in f:
                    print(f"Warning: Environment {env_name} not found, creating...")
                    env_group = f.create_group(env_group_path)
                else:
                    env_group = f[env_group_path]
                
                # pose_pairs 그룹 생성
                if 'pose_pairs' not in env_group:
                    pairs_group = env_group.create_group('pose_pairs')
                else:
                    pairs_group = env_group['pose_pairs']
                
                # rigid body별 그룹 생성
                rb_group_name = f'rb_{rb_id}'
                if rb_group_name in pairs_group:
                    # 기존 데이터 삭제
                    del pairs_group[rb_group_name]
                
                rb_group = pairs_group.create_group(rb_group_name)
                
                # pose_pair 데이터 저장
                rb_group.create_dataset('data', data=pairs, compression='gzip')
                
                # 메타데이터 저장
                rb_group.attrs['rb_id'] = rb_id
                rb_group.attrs['pair_count'] = len(pairs)
                rb_group.attrs['creation_time'] = datetime.now().isoformat()
                rb_group.attrs['pair_format'] = 'se3_pose_pairs'
                rb_group.attrs['description'] = 'Init-target SE(3) pose pairs for path planning'
                
                # 생성 정보 저장
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        rb_group.attrs[key] = value
                
                # 전체 환경 수 업데이트
                f['global_metadata'].attrs['last_updated'] = datetime.now().isoformat()
                
                print(f"✅ Added {len(pairs)} pose pairs for {env_name}/rb_{rb_id}")
                return True
                
        except Exception as e:
            print(f"❌ Failed to add pose pairs for {env_name}/rb_{rb_id}: {e}")
            return False
    
    def get_poses(self, env_name: str, rb_id: int) -> Tuple[Optional[np.ndarray], dict]:
        """
        pose 데이터 조회
        
        Args:
            env_name: 환경 이름
            rb_id: Rigid body ID
            
        Returns:
            (poses, metadata): pose 배열과 메타데이터
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                rb_group_path = f'environments/{env_name}/poses/rb_{rb_id}'
                
                if rb_group_path not in f:
                    return None, {}
                
                rb_group = f[rb_group_path]
                poses = rb_group['data'][:]
                
                # 메타데이터 수집
                metadata = {}
                for key, value in rb_group.attrs.items():
                    metadata[key] = value
                
                return poses, metadata
                
        except Exception as e:
            print(f"❌ Failed to get poses for {env_name}/rb_{rb_id}: {e}")
            return None, {}
    
    def get_pose_pairs(self, env_name: str, rb_id: int) -> Tuple[Optional[np.ndarray], dict]:
        """
        pose_pair 데이터 조회
        
        Args:
            env_name: 환경 이름
            rb_id: Rigid body ID
            
        Returns:
            (pairs, metadata): pose_pair 배열과 메타데이터
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                rb_group_path = f'environments/{env_name}/pose_pairs/rb_{rb_id}'
                
                if rb_group_path not in f:
                    return None, {}
                
                rb_group = f[rb_group_path]
                pairs = rb_group['data'][:]
                
                # 메타데이터 수집
                metadata = {}
                for key, value in rb_group.attrs.items():
                    metadata[key] = value
                
                return pairs, metadata
                
        except Exception as e:
            print(f"❌ Failed to get pose pairs for {env_name}/rb_{rb_id}: {e}")
            return None, {}
    
    def list_environments(self) -> List[str]:
        """사용 가능한 환경 목록 반환"""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if 'environments' not in f:
                    return []
                
                return sorted(list(f['environments'].keys()))
                
        except Exception as e:
            print(f"❌ Failed to list environments: {e}")
            return []
    
    def list_rigid_bodies(self, env_name: str) -> List[int]:
        """특정 환경의 사용 가능한 rigid body ID 목록"""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                poses_path = f'environments/{env_name}/poses'
                
                if poses_path not in f:
                    return []
                
                rb_ids = []
                for rb_name in f[poses_path].keys():
                    if rb_name.startswith('rb_'):
                        rb_id = int(rb_name.split('_')[1])
                        rb_ids.append(rb_id)
                
                return sorted(rb_ids)
                
        except Exception as e:
            print(f"❌ Failed to list rigid bodies for {env_name}: {e}")
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 요약 정보 반환"""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                summary = {}
                
                # 전역 메타데이터
                if 'global_metadata' in f:
                    for key, value in f['global_metadata'].attrs.items():
                        summary[key] = value
                
                # 환경별 통계
                environments = self.list_environments()
                summary['environment_count'] = len(environments)
                summary['environments'] = environments
                
                # 각 환경별 pose/pair 개수
                env_stats = {}
                for env_name in environments:
                    rb_ids = self.list_rigid_bodies(env_name)
                    env_stats[env_name] = {
                        'rigid_bodies': rb_ids,
                        'total_poses': 0,
                        'total_pairs': 0
                    }
                    
                    for rb_id in rb_ids:
                        poses, _ = self.get_poses(env_name, rb_id)
                        pairs, _ = self.get_pose_pairs(env_name, rb_id)
                        
                        if poses is not None:
                            env_stats[env_name]['total_poses'] += len(poses)
                        if pairs is not None:
                            env_stats[env_name]['total_pairs'] += len(pairs)
                
                summary['environment_stats'] = env_stats
                
                return summary
                
        except Exception as e:
            print(f"❌ Failed to get summary: {e}")
            return {}


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Testing UnifiedPoseManager...")
    
    # 테스트 HDF5 파일 생성
    test_path = "/tmp/test_unified_poses.h5"
    if os.path.exists(test_path):
        os.remove(test_path)
    
    manager = UnifiedPoseManager(test_path)
    
    # 테스트 데이터
    test_poses = np.array([
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.5],
        [3.0, 4.0, 0.0, 0.0, 0.0, 1.0],
        [5.0, 6.0, 0.0, 0.0, 0.0, 1.5]
    ])
    
    test_pairs = np.array([
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.5, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0],
        [3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.5]
    ])
    
    # pose 추가 테스트
    pose_metadata = {
        'safety_margin': 0.05,
        'max_attempts': 1000,
        'success_rate': 85.5
    }
    
    success = manager.add_poses("circle_env_000000", 0, test_poses, pose_metadata)
    print(f"Pose addition: {'✅' if success else '❌'}")
    
    # pose_pair 추가 테스트
    pair_metadata = {
        'generation_method': 'random_sampling',
        'min_distance': 1.0
    }
    
    success = manager.add_pose_pairs("circle_env_000000", 0, test_pairs, pair_metadata)
    print(f"Pose pair addition: {'✅' if success else '❌'}")
    
    # 조회 테스트
    poses, pose_meta = manager.get_poses("circle_env_000000", 0)
    pairs, pair_meta = manager.get_pose_pairs("circle_env_000000", 0)
    
    print(f"Pose retrieval: {'✅' if poses is not None else '❌'}")
    print(f"Pose pair retrieval: {'✅' if pairs is not None else '❌'}")
    
    # 요약 정보 테스트
    summary = manager.get_summary()
    print(f"Summary: {summary}")
    
    # 정리
    os.remove(test_path)
    print("🎉 Test completed")
