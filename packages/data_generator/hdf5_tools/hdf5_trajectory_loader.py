#!/usr/bin/env python3
"""
HDF5 Trajectory Loader
HDF5 형태로 저장된 궤적 데이터를 로드하는 도구
"""

import h5py
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
sys.path.append('/Users/a123/Documents/Projects/2D_sim/packages/utils')
from SE3_functions import trajectory_quaternion_to_euler, quaternion_7d_to_euler_6d


class HDF5TrajectoryLoader:
    """HDF5 궤적 데이터 로더"""
    
    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: HDF5 파일 경로
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        self.hdf5_file = None
        self._open_file()
        
        print(f"✅ HDF5 Trajectory Loader initialized")
        print(f"   File: {self.hdf5_path}")
        self._print_summary()
    
    def _open_file(self):
        """HDF5 파일 열기"""
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
    
    def close(self):
        """HDF5 파일 닫기"""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _print_summary(self):
        """HDF5 파일 요약 정보 출력"""
        try:
            # Environment count
            env_count = len(list(self.hdf5_file['pose_pairs'].keys()))
            
            # Rigid body count  
            rb_count = len(list(self.hdf5_file['metadata/rigid_bodies'].keys()))
            
            # Schema info
            schema_info = self.hdf5_file['metadata/schema_info']
            version = schema_info.attrs.get('version', 'unknown')
            format_type = schema_info.attrs.get('format', 'unknown')
            
            print(f"   Schema version: {version}")
            print(f"   Data format: {format_type}")
            print(f"   Environments: {env_count}")
            print(f"   Rigid bodies: {rb_count}")
            
        except Exception as e:
            print(f"   ⚠️ Could not read summary: {e}")
    
    def list_environments(self) -> List[str]:
        """사용 가능한 환경 ID 목록 반환"""
        return list(self.hdf5_file['pose_pairs'].keys())
    
    def list_rigid_bodies(self) -> List[int]:
        """사용 가능한 로봇 ID 목록 반환"""
        rb_ids = []
        for rb_key in self.hdf5_file['metadata/rigid_bodies'].keys():
            if rb_key.startswith('rb_'):
                rb_id = int(rb_key.split('_')[1])
                rb_ids.append(rb_id)
        return sorted(rb_ids)
    
    def get_environment_metadata(self, env_id: str) -> Dict[str, Any]:
        """환경 메타데이터 반환"""
        if env_id not in self.hdf5_file['metadata/environments']:
            raise ValueError(f"Environment {env_id} not found")
        
        env_group = self.hdf5_file['metadata/environments'][env_id]
        metadata = {}
        
        # Attributes
        for key, value in env_group.attrs.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value
        
        # Datasets
        for key in env_group.keys():
            metadata[key] = env_group[key][...]
        
        return metadata
    
    def get_rigid_body_metadata(self, rigid_body_id: int) -> Dict[str, Any]:
        """로봇 메타데이터 반환"""
        rb_key = f"rb_{rigid_body_id}"
        if rb_key not in self.hdf5_file['metadata/rigid_bodies']:
            raise ValueError(f"Rigid body {rigid_body_id} not found")
        
        rb_group = self.hdf5_file['metadata/rigid_bodies'][rb_key]
        metadata = {}
        
        # Attributes
        for key, value in rb_group.attrs.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value
        
        # Datasets
        for key in rb_group.keys():
            metadata[key] = rb_group[key][...]
        
        return metadata
    
    def load_pose_pairs(self, env_id: str, output_format: str = '7d') -> np.ndarray:
        """
        환경의 pose pairs 로드
        
        Args:
            env_id: 환경 ID
            output_format: '7d' (quaternion) 또는 '6d' (euler)
        
        Returns:
            [N, 2, 7] 또는 [N, 2, 6] pose pairs
        """
        if env_id not in self.hdf5_file['pose_pairs']:
            raise ValueError(f"Environment {env_id} not found in pose_pairs")
        
        pairs_data = self.hdf5_file['pose_pairs'][env_id]['pairs'][...]
        
        if output_format == '6d':
            # Convert 7D quaternion to 6D euler
            N = pairs_data.shape[0]
            pairs_6d = np.zeros((N, 2, 6))
            
            for i in range(N):
                for j in range(2):  # init, target
                    pose_7d = pairs_data[i, j, :]
                    pose_6d = quaternion_7d_to_euler_6d(pose_7d)
                    pairs_6d[i, j, :] = pose_6d
            
            return pairs_6d
        
        return pairs_data
    
    def load_trajectory(self, env_id: str, rb_id: int, pair_index: int, 
                       trajectory_type: str = 'raw', output_format: str = '7d') -> Optional[np.ndarray]:
        """
        특정 궤적 로드
        
        Args:
            env_id: 환경 ID
            rb_id: 로봇 ID  
            pair_index: pose pair 인덱스
            trajectory_type: 'raw' 또는 'bsplined'
            output_format: '7d' (quaternion) 또는 '6d' (euler)
        
        Returns:
            [N, 7] 또는 [N, 6] 궤적 데이터, 없으면 None
        """
        # 궤적 경로 구성
        traj_path = f"trajectories/{trajectory_type}/{env_id}/rb_{rb_id}/traj_{pair_index:06d}"
        
        if traj_path not in self.hdf5_file:
            return None
        
        trajectory_data = self.hdf5_file[traj_path][...]
        
        if output_format == '6d' and trajectory_data.shape[1] == 7:
            # Convert 7D quaternion to 6D euler
            trajectory_6d = trajectory_quaternion_to_euler(trajectory_data)
            return trajectory_6d
        
        return trajectory_data
    
    def load_trajectories_by_environment(self, env_id: str, rb_id: int = None, 
                                       trajectory_type: str = 'raw',
                                       output_format: str = '7d') -> Dict[int, np.ndarray]:
        """
        환경별 모든 궤적 로드
        
        Args:
            env_id: 환경 ID
            rb_id: 로봇 ID (None이면 모든 로봇)
            trajectory_type: 'raw' 또는 'bsplined'  
            output_format: '7d' 또는 '6d'
        
        Returns:
            {pair_index: trajectory_data} 딕셔너리
        """
        trajectories = {}
        
        traj_base_path = f"trajectories/{trajectory_type}/{env_id}"
        if traj_base_path not in self.hdf5_file:
            print(f"⚠️ No trajectories found for {env_id} in {trajectory_type}")
            return trajectories
        
        env_group = self.hdf5_file[traj_base_path]
        
        # Rigid body 필터링
        rb_keys = list(env_group.keys())
        if rb_id is not None:
            rb_keys = [key for key in rb_keys if key == f"rb_{rb_id}"]
        
        for rb_key in rb_keys:
            rb_group = env_group[rb_key]
            
            for traj_key in rb_group.keys():
                if traj_key.startswith('traj_'):
                    pair_index = int(traj_key.split('_')[1])
                    trajectory_data = rb_group[traj_key][...]
                    
                    if output_format == '6d' and trajectory_data.shape[1] == 7:
                        trajectory_data = trajectory_quaternion_to_euler(trajectory_data)
                    
                    trajectories[pair_index] = trajectory_data
        
        return trajectories
    
    def load_trajectories_by_success_status(self, success_only: bool = True,
                                          trajectory_type: str = 'raw',
                                          output_format: str = '7d') -> List[Dict[str, Any]]:
        """
        성공/실패 상태별 궤적 로드
        
        Args:
            success_only: True이면 성공한 궤적만
            trajectory_type: 'raw' 또는 'bsplined'
            output_format: '7d' 또는 '6d'
        
        Returns:
            궤적 정보 딕셔너리 리스트
        """
        trajectories = []
        
        # Success index가 있는 경우에만 사용
        if 'success_index' in self.hdf5_file['indices']:
            success_data = self.hdf5_file['indices/success_index'][...]
            # TODO: success index 기반 필터링 구현
        
        # 현재는 모든 궤적을 반환 (success index 없는 경우)
        for env_id in self.list_environments():
            env_trajectories = self.load_trajectories_by_environment(
                env_id, trajectory_type=trajectory_type, output_format=output_format
            )
            
            for pair_index, traj_data in env_trajectories.items():
                traj_info = {
                    'env_id': env_id,
                    'pair_index': pair_index,
                    'trajectory': traj_data,
                    'success': True  # 현재는 모두 성공으로 가정
                }
                trajectories.append(traj_info)
        
        return trajectories
    
    def get_trajectory_count(self, env_id: str = None, rb_id: int = None, 
                           trajectory_type: str = 'raw') -> int:
        """궤적 개수 반환"""
        count = 0
        
        if env_id:
            env_list = [env_id]
        else:
            env_list = self.list_environments()
        
        for env in env_list:
            trajectories = self.load_trajectories_by_environment(
                env, rb_id=rb_id, trajectory_type=trajectory_type
            )
            count += len(trajectories)
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """HDF5 데이터셋 통계 반환"""
        stats = {
            'environments': len(self.list_environments()),
            'rigid_bodies': len(self.list_rigid_bodies()),
            'trajectory_counts': {
                'raw': self.get_trajectory_count(trajectory_type='raw'),
                'bsplined': self.get_trajectory_count(trajectory_type='bsplined')
            }
        }
        
        return stats


if __name__ == "__main__":
    """테스트 실행"""
    print("🧪 Testing HDF5 Trajectory Loader")
    
    test_hdf5_path = "test_trajectory_dataset.h5"
    
    try:
        with HDF5TrajectoryLoader(test_hdf5_path) as loader:
            # Print statistics
            stats = loader.get_statistics()
            print(f"\n📊 Dataset Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # List environments and rigid bodies
            envs = loader.list_environments()
            rbs = loader.list_rigid_bodies()
            print(f"\n🌍 Environments: {envs}")
            print(f"🤖 Rigid Bodies: {rbs}")
            
    except FileNotFoundError:
        print(f"⚠️ Test file {test_hdf5_path} not found")
        print("   Run hdf5_schema_creator.py first to create test data")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()