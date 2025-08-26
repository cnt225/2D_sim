#!/usr/bin/env python3
"""
HDF5 Schema Creator
Plan.md에 따른 HDF5 스키마 구조를 생성하는 도구

HDF5 구조:
trajectory_dataset.h5
├── metadata/                 # 메타데이터
│   ├── schema_info/         # 스키마 정보
│   ├── environments/        # 환경 정보
│   ├── rigid_bodies/        # 로봇 정보
│   └── generation_settings/ # 생성 설정
├── pose_pairs/              # 환경별 pose pair 데이터
│   ├── circle_env_000000/
│   │   └── pairs: [N, 2, 7] # init/target pose [x,y,z,qw,qx,qy,qz]
│   └── ...
├── trajectories/            # 궤적 데이터
│   ├── raw/                 # 원본 RRT 궤적
│   ├── bsplined/           # B-spline 스무딩된 궤적
│   └── derivatives/        # 속도, 가속도 등 파생 데이터
└── indices/                # 빠른 검색을 위한 인덱스
"""

import h5py
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def create_hdf5_schema(hdf5_path: str, overwrite: bool = False) -> h5py.File:
    """
    빈 HDF5 스키마 생성
    
    Args:
        hdf5_path: HDF5 파일 경로
        overwrite: 기존 파일 덮어쓰기 여부
    
    Returns:
        h5py.File: 열린 HDF5 파일 객체
    """
    hdf5_path = Path(hdf5_path)
    
    # 기존 파일 체크
    if hdf5_path.exists() and not overwrite:
        print(f"⚠️ HDF5 file already exists: {hdf5_path}")
        print("   Use overwrite=True to recreate")
        return h5py.File(hdf5_path, 'a')  # append mode
    
    # Create directory if needed
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Creating HDF5 schema: {hdf5_path}")
    
    # Create HDF5 file
    hdf5_file = h5py.File(hdf5_path, 'w')
    
    # === 1. Metadata 그룹 생성 ===
    metadata_group = hdf5_file.create_group('metadata')
    
    # Schema info
    schema_group = metadata_group.create_group('schema_info')
    schema_group.attrs['version'] = '1.0.0'
    schema_group.attrs['format'] = 'quaternion_7d'
    schema_group.attrs['created'] = datetime.now().isoformat()
    schema_group.attrs['description'] = 'HDF5-based trajectory dataset with quaternion representation'
    
    # Environments
    env_group = metadata_group.create_group('environments')
    env_group.attrs['description'] = 'Environment metadata and configurations'
    
    # Rigid bodies
    rb_group = metadata_group.create_group('rigid_bodies')
    rb_group.attrs['description'] = 'Rigid body configurations and parameters'
    
    # Generation settings
    gen_group = metadata_group.create_group('generation_settings')
    gen_group.attrs['description'] = 'Trajectory generation parameters and settings'
    
    # === 2. Pose pairs 그룹 생성 ===
    pairs_group = hdf5_file.create_group('pose_pairs')
    pairs_group.attrs['format'] = '[N, 2, 7]'
    pairs_group.attrs['description'] = 'Pose pairs in 7D quaternion format [x,y,z,qw,qx,qy,qz]'
    
    # === 3. Trajectories 그룹 생성 ===
    traj_group = hdf5_file.create_group('trajectories')
    
    # Raw trajectories (original RRT output)
    raw_group = traj_group.create_group('raw')
    raw_group.attrs['format'] = '[N, 7]'
    raw_group.attrs['description'] = 'Raw RRT trajectories in 7D quaternion format'
    
    # B-splined trajectories
    bsplined_group = traj_group.create_group('bsplined')
    bsplined_group.attrs['format'] = '[N, 7]'
    bsplined_group.attrs['description'] = 'B-spline smoothed trajectories with quaternion SLERP'
    
    # Derivative data
    deriv_group = traj_group.create_group('derivatives')
    deriv_group.attrs['description'] = 'Velocity, acceleration and other derivative data'
    
    # === 4. Indices 그룹 생성 ===
    index_group = hdf5_file.create_group('indices')
    index_group.attrs['description'] = 'Indexing data for fast trajectory lookup'
    
    # Environment index
    env_idx = index_group.create_dataset('environment_index', (0,), maxshape=(None,), 
                                        dtype=h5py.string_dtype(encoding='utf-8'))
    env_idx.attrs['description'] = 'Environment ID to HDF5 path mapping'
    
    # Success/failure index
    success_idx = index_group.create_dataset('success_index', (0, 2), maxshape=(None, 2),
                                           dtype='bool')
    success_idx.attrs['description'] = 'Success status [environment_idx, trajectory_idx]'
    
    print(f"✅ HDF5 schema created successfully")
    print(f"   Structure: metadata/, pose_pairs/, trajectories/, indices/")
    
    return hdf5_file


def add_environment_metadata(hdf5_file: h5py.File, env_data: Dict[str, Any]) -> None:
    """
    환경 메타데이터 추가
    
    Args:
        hdf5_file: HDF5 파일 객체
        env_data: 환경 데이터 딕셔너리
            {
                'env_id': 'circle_env_000000',
                'name': 'Circle Environment',
                'description': 'Circular obstacles',
                'ply_file': 'path/to/env.ply',
                'bounds': [x_min, x_max, y_min, y_max],
                'obstacles': [...],
                'difficulty': 'medium'
            }
    """
    required_fields = ['env_id', 'name']
    if not all(field in env_data for field in required_fields):
        raise ValueError(f"Environment data must contain: {required_fields}")
    
    env_id = env_data['env_id']
    env_group = hdf5_file['metadata/environments']
    
    # Create environment subgroup
    if env_id in env_group:
        print(f"⚠️ Environment {env_id} already exists, updating...")
        del env_group[env_id]
    
    env_subgroup = env_group.create_group(env_id)
    
    # Add metadata as attributes
    for key, value in env_data.items():
        if isinstance(value, (str, int, float, bool)):
            env_subgroup.attrs[key] = value
        elif isinstance(value, (list, np.ndarray)):
            # Store arrays as datasets
            env_subgroup.create_dataset(key, data=np.array(value))
        else:
            # Convert complex types to JSON strings
            env_subgroup.attrs[key] = json.dumps(value)
    
    env_subgroup.attrs['added_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Environment metadata added: {env_id}")


def add_rigid_body_metadata(hdf5_file: h5py.File, rb_data: Dict[str, Any]) -> None:
    """
    로봇 메타데이터 추가
    
    Args:
        hdf5_file: HDF5 파일 객체
        rb_data: 로봇 데이터 딕셔너리
            {
                'rigid_body_id': 3,
                'name': 'EndEffector',
                'type': 'elongated_ellipse',
                'dimensions': [length, width, height],
                'mass': 1.0,
                'collision_model': 'ellipsoid',
                'dof': 3,
                'description': 'SE(3) rigid body with elongated ellipse shape'
            }
    """
    required_fields = ['rigid_body_id', 'name', 'type']
    if not all(field in rb_data for field in required_fields):
        raise ValueError(f"Rigid body data must contain: {required_fields}")
    
    rb_id = f"rb_{rb_data['rigid_body_id']}"
    rb_group = hdf5_file['metadata/rigid_bodies']
    
    # Create rigid body subgroup
    if rb_id in rb_group:
        print(f"⚠️ Rigid body {rb_id} already exists, updating...")
        del rb_group[rb_id]
    
    rb_subgroup = rb_group.create_group(rb_id)
    
    # Add metadata as attributes
    for key, value in rb_data.items():
        if isinstance(value, (str, int, float, bool)):
            rb_subgroup.attrs[key] = value
        elif isinstance(value, (list, np.ndarray)):
            # Store arrays as datasets
            rb_subgroup.create_dataset(key, data=np.array(value))
        else:
            # Convert complex types to JSON strings
            rb_subgroup.attrs[key] = json.dumps(value)
    
    rb_subgroup.attrs['added_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Rigid body metadata added: {rb_id}")


def add_generation_settings(hdf5_file: h5py.File, settings: Dict[str, Any]) -> None:
    """
    궤적 생성 설정 추가
    
    Args:
        hdf5_file: HDF5 파일 객체
        settings: 생성 설정
            {
                'planner': 'RRT-Connect',
                'max_planning_time': 5.0,
                'range': 0.1,
                'goal_bias': 0.05,
                'bspline_degree': 3,
                'conversion': '6d_to_7d'
            }
    """
    gen_group = hdf5_file['metadata/generation_settings']
    
    # Add current timestamp
    settings['generation_timestamp'] = datetime.now().isoformat()
    
    # Store settings
    for key, value in settings.items():
        if isinstance(value, (str, int, float, bool)):
            gen_group.attrs[key] = value
        elif isinstance(value, (list, np.ndarray)):
            gen_group.create_dataset(key, data=np.array(value))
        else:
            gen_group.attrs[key] = json.dumps(value)
    
    print(f"✅ Generation settings added")


def create_pose_pair_group(hdf5_file: h5py.File, env_id: str, pose_pairs_data: np.ndarray) -> None:
    """
    환경별 pose pair 그룹 생성
    
    Args:
        hdf5_file: HDF5 파일 객체
        env_id: 환경 ID (예: 'circle_env_000000')
        pose_pairs_data: [N, 2, 7] 형태의 pose pairs 데이터
    """
    pairs_group = hdf5_file['pose_pairs']
    
    if env_id in pairs_group:
        print(f"⚠️ Pose pairs for {env_id} already exist, updating...")
        del pairs_group[env_id]
    
    env_pair_group = pairs_group.create_group(env_id)
    
    # Validate data format
    if len(pose_pairs_data.shape) != 3 or pose_pairs_data.shape[1:] != (2, 7):
        raise ValueError(f"Expected [N, 2, 7] pose pairs, got shape {pose_pairs_data.shape}")
    
    # Store pose pairs
    pairs_dataset = env_pair_group.create_dataset('pairs', data=pose_pairs_data, 
                                                 compression='gzip', compression_opts=6)
    pairs_dataset.attrs['format'] = '[x, y, z, qw, qx, qy, qz]'
    pairs_dataset.attrs['description'] = 'Initial and target poses in 7D quaternion format'
    pairs_dataset.attrs['count'] = pose_pairs_data.shape[0]
    pairs_dataset.attrs['added_timestamp'] = datetime.now().isoformat()
    
    print(f"✅ Pose pairs added for {env_id}: {pose_pairs_data.shape[0]} pairs")


def validate_hdf5_schema(hdf5_path: str) -> bool:
    """
    HDF5 스키마 유효성 검증
    
    Args:
        hdf5_path: HDF5 파일 경로
    
    Returns:
        bool: 스키마가 유효한지 여부
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Required groups
            required_groups = [
                'metadata',
                'metadata/schema_info',
                'metadata/environments', 
                'metadata/rigid_bodies',
                'metadata/generation_settings',
                'pose_pairs',
                'trajectories',
                'trajectories/raw',
                'trajectories/bsplined',
                'trajectories/derivatives',
                'indices'
            ]
            
            for group_path in required_groups:
                if group_path not in f:
                    print(f"❌ Missing required group: {group_path}")
                    return False
            
            # Check schema version
            if 'version' not in f['metadata/schema_info'].attrs:
                print(f"❌ Missing schema version")
                return False
            
            print(f"✅ HDF5 schema validation passed")
            return True
            
    except Exception as e:
        print(f"❌ HDF5 schema validation failed: {e}")
        return False


if __name__ == "__main__":
    """테스트 실행"""
    print("🧪 Testing HDF5 Schema Creator")
    
    # Test schema creation
    test_hdf5_path = "test_trajectory_dataset.h5"
    
    try:
        # Create schema
        hdf5_file = create_hdf5_schema(test_hdf5_path, overwrite=True)
        
        # Add sample environment metadata
        env_data = {
            'env_id': 'circle_env_000000',
            'name': 'Circle Test Environment',
            'description': 'Test environment with circular obstacles',
            'ply_file': 'data/environments/circles.ply',
            'bounds': [-5.0, 5.0, -5.0, 5.0],
            'difficulty': 'medium'
        }
        add_environment_metadata(hdf5_file, env_data)
        
        # Add sample rigid body metadata
        rb_data = {
            'rigid_body_id': 3,
            'name': 'TestEndEffector',
            'type': 'elongated_ellipse',
            'dimensions': [0.2, 0.1, 0.05],
            'mass': 1.0,
            'collision_model': 'ellipsoid',
            'dof': 3
        }
        add_rigid_body_metadata(hdf5_file, rb_data)
        
        # Add generation settings
        settings = {
            'planner': 'RRT-Connect',
            'max_planning_time': 5.0,
            'conversion': '6d_to_7d'
        }
        add_generation_settings(hdf5_file, settings)
        
        # Close file
        hdf5_file.close()
        
        # Validate schema
        if validate_hdf5_schema(test_hdf5_path):
            print(f"✅ Test completed successfully: {test_hdf5_path}")
        else:
            print(f"❌ Schema validation failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()