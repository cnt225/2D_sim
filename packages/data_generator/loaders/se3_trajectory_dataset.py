#!/usr/bin/env python3
"""
SE(3) Trajectory Dataset for PyTorch
HDF5 기반 SE(3) 궤적 데이터를 위한 PyTorch Dataset 클래스
fm-main 패턴을 참조하여 구현
"""

import torch
import torch.utils.data
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from copy import deepcopy
import sys

# Import SE(3) functions and HDF5 loader
sys.path.append('/Users/a123/Documents/Projects/2D_sim/packages/utils')
sys.path.append('/Users/a123/Documents/Projects/2D_sim/packages/data_generator/hdf5_tools')

from SE3_functions import (
    traj_smooth_se3_bspline_slerp,
    traj_process_se3_pipeline,
    traj_build_labels_with_policy,
    traj_integrate_by_twist,
    _se3_exp,
    _se3_log,
    trajectory_quaternion_to_euler,
    euler_6d_to_quaternion_7d
)
from hdf5_trajectory_loader import HDF5TrajectoryLoader


class SE3TrajectoryDataset(torch.utils.data.Dataset):
    """
    SE(3) 궤적 데이터셋 클래스 (fm-main 패턴 참조)
    
    Features:
    - HDF5 기반 데이터 로딩
    - SE(3) 궤적 스무딩 및 처리 파이프라인
    - Arc-length 리샘플링 및 시간 정책
    - Body twist 라벨 생성
    - 데이터 증강 (회전, 노이즈 등)
    - 배치 처리 및 텐서 변환
    """
    
    def __init__(self, 
                 hdf5_path: str,
                 split: str = 'train',
                 env_ids: Optional[List[str]] = None,
                 rigid_body_ids: Optional[List[int]] = None,
                 trajectory_type: str = 'raw',
                 
                 # SE(3) 처리 파라미터
                 use_smoothing: bool = True,
                 smooth_strength: float = 0.1,
                 num_samples: int = 200,
                 lambda_rot: float = 0.0,
                 time_policy: str = "curvature",
                 v_ref: float = 0.4,
                 v_cap: float = 0.5,
                 a_lat_max: float = 1.0,
                 
                 # 데이터 증강
                 augmentation: bool = True,
                 rotation_noise_std: float = 0.1,
                 position_noise_std: float = 0.05,
                 
                 # 배치 설정
                 max_trajectories: Optional[int] = None,
                 **kwargs):
        """
        Args:
            hdf5_path: HDF5 데이터 파일 경로
            split: 데이터 분할 ('train', 'valid', 'test')
            env_ids: 사용할 환경 ID 목록 (None이면 모든 환경)
            rigid_body_ids: 사용할 로봇 ID 목록 (None이면 모든 로봇)
            trajectory_type: 궤적 타입 ('raw' | 'bsplined')
            
            # SE(3) 처리
            use_smoothing: SE(3) 스무딩 사용 여부
            smooth_strength: 스무딩 강도 (0.0=보간, >0=스무딩)
            num_samples: 리샘플링 포인트 수
            lambda_rot: 회전 가중치 (arc-length 계산 시)
            time_policy: 시간 정책 ("uniform" | "curvature")
            v_ref: 기준 속도
            v_cap: 최대 속도
            a_lat_max: 최대 횡가속도
            
            # 증강
            augmentation: 데이터 증강 사용 여부
            rotation_noise_std: 회전 노이즈 표준편차
            position_noise_std: 위치 노이즈 표준편차
            
            # 기타
            max_trajectories: 최대 궤적 수 제한
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.trajectory_type = trajectory_type
        
        # SE(3) 처리 파라미터
        self.use_smoothing = use_smoothing
        self.smooth_strength = smooth_strength
        self.num_samples = num_samples
        self.lambda_rot = lambda_rot
        self.time_policy = time_policy
        self.v_ref = v_ref
        self.v_cap = v_cap
        self.a_lat_max = a_lat_max
        
        # 증강 파라미터
        self.augmentation = augmentation and (split == 'train')
        self.rotation_noise_std = rotation_noise_std
        self.position_noise_std = position_noise_std
        
        # HDF5 로더 초기화
        self.hdf5_loader = HDF5TrajectoryLoader(str(self.hdf5_path))
        
        # 데이터 수집
        self._collect_trajectory_data(env_ids, rigid_body_ids, max_trajectories)
        
        print(f"✅ SE3TrajectoryDataset initialized")
        print(f"   Split: {split}")
        print(f"   Trajectories: {len(self.trajectory_list)}")
        print(f"   Environments: {len(set([t['env_id'] for t in self.trajectory_list]))}")
        print(f"   Augmentation: {self.augmentation}")
    
    def _collect_trajectory_data(self, 
                               env_ids: Optional[List[str]], 
                               rigid_body_ids: Optional[List[int]],
                               max_trajectories: Optional[int]):
        """궤적 데이터 수집"""
        self.trajectory_list = []
        
        # 환경 목록 결정
        available_envs = self.hdf5_loader.list_environments()
        if env_ids is None:
            env_ids = available_envs
        else:
            env_ids = [env for env in env_ids if env in available_envs]
        
        # 로봇 목록 결정
        available_rbs = self.hdf5_loader.list_rigid_bodies()
        if rigid_body_ids is None:
            rigid_body_ids = available_rbs
        else:
            rigid_body_ids = [rb for rb in rigid_body_ids if rb in available_rbs]
        
        print(f"📊 Collecting trajectories from:")
        print(f"   Environments: {len(env_ids)}")
        print(f"   Rigid bodies: {rigid_body_ids}")
        
        # 데이터 분할 (fm-main 패턴 참조)
        total_collected = 0
        
        for env_id in env_ids:
            try:
                # 환경별 모든 궤적 로드
                env_trajectories = self.hdf5_loader.load_trajectories_by_environment(
                    env_id, rb_id=None, trajectory_type=self.trajectory_type, output_format='7d'
                )
                
                if not env_trajectories:
                    continue
                
                # 궤적 인덱스 정렬
                pair_indices = sorted(env_trajectories.keys())
                
                # 데이터 분할 (fm-main 스타일)
                num_trajectories = len(pair_indices)
                num_val_data = num_test_data = num_trajectories // 5
                num_train_data = num_trajectories - num_val_data - num_test_data
                
                if self.split == 'train':
                    split_indices = pair_indices[:num_train_data]
                elif self.split == 'valid':
                    split_indices = pair_indices[num_train_data:num_train_data+num_val_data]
                elif self.split == 'test':
                    split_indices = pair_indices[num_train_data+num_val_data:]
                else:
                    raise ValueError(f"Unknown split: {self.split}")
                
                # 각 로봇에 대해 궤적 수집
                for rb_id in rigid_body_ids:
                    for pair_index in split_indices:
                        traj_data = self.hdf5_loader.load_trajectory(
                            env_id, rb_id, pair_index, self.trajectory_type, '7d'
                        )
                        
                        if traj_data is not None and len(traj_data) > 5:  # 최소 길이 체크
                            trajectory_info = {
                                'env_id': env_id,
                                'rb_id': rb_id,
                                'pair_index': pair_index,
                                'raw_trajectory': traj_data  # [N, 7] numpy array
                            }
                            self.trajectory_list.append(trajectory_info)
                            total_collected += 1
                            
                            # 최대 궤적 수 제한
                            if max_trajectories and total_collected >= max_trajectories:
                                break
                    
                    if max_trajectories and total_collected >= max_trajectories:
                        break
                
                if max_trajectories and total_collected >= max_trajectories:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error loading environment {env_id}: {e}")
                continue
        
        print(f"📋 Collected {len(self.trajectory_list)} trajectories for {self.split}")
        
        if len(self.trajectory_list) == 0:
            raise RuntimeError("No valid trajectories found")
    
    def _process_trajectory(self, raw_traj_7d: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        SE(3) 궤적 처리 파이프라인
        7D quaternion → SE(3) matrix → 스무딩 → 리샘플링 → 라벨 생성
        """
        # 1) 7D quaternion → SE(3) matrices
        N = raw_traj_7d.shape[0]
        T_raw = torch.zeros(N, 4, 4, dtype=torch.float32)
        
        for i in range(N):
            pose_7d = raw_traj_7d[i]  # [x, y, z, qw, qx, qy, qz]
            
            # Position
            T_raw[i, :3, 3] = torch.tensor(pose_7d[:3])
            
            # Quaternion → Rotation matrix
            qw, qx, qy, qz = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
            
            # Quaternion to rotation matrix conversion
            R = torch.zeros(3, 3)
            R[0, 0] = 1 - 2*(qy*qy + qz*qz)
            R[0, 1] = 2*(qx*qy - qw*qz)
            R[0, 2] = 2*(qx*qz + qw*qy)
            R[1, 0] = 2*(qx*qy + qw*qz)
            R[1, 1] = 1 - 2*(qx*qx + qz*qz)
            R[1, 2] = 2*(qy*qz - qw*qx)
            R[2, 0] = 2*(qx*qz - qw*qy)
            R[2, 1] = 2*(qy*qz + qw*qx)
            R[2, 2] = 1 - 2*(qx*qx + qy*qy)
            
            T_raw[i, :3, :3] = R
            T_raw[i, 3, 3] = 1.0
        
        # 2) SE(3) 처리 파이프라인 적용
        T_processed, dt_seq, xi_labels, T_smooth = traj_process_se3_pipeline(
            T_raw,
            smooth_first=self.use_smoothing,
            smooth=self.smooth_strength,
            num_samples=self.num_samples,
            lambda_rot=self.lambda_rot,
            policy=self.time_policy,
            v_ref=self.v_ref,
            v_cap=self.v_cap,
            a_lat_max=self.a_lat_max
        )
        
        return {
            'T_raw': T_raw,                    # [N, 4, 4]
            'T_processed': T_processed,        # [M, 4, 4] 
            'dt_seq': dt_seq,                  # [M-1]
            'xi_labels': xi_labels,            # [M, 6]
            'T_smooth': T_smooth if T_smooth is not None else T_raw  # [N, 4, 4]
        }
    
    def _apply_augmentation(self, T: torch.Tensor) -> torch.Tensor:
        """데이터 증강 적용 (fm-main 패턴 참조)"""
        if not self.augmentation:
            return T
        
        T_aug = T.clone()
        
        # 1) Random rotation around Z-axis (fm-main 스타일)
        if torch.rand(1) > 0.5:
            theta = torch.rand(1) * 2 * np.pi
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            
            R_z = torch.eye(4)
            R_z[0, 0] = cos_theta
            R_z[0, 1] = -sin_theta
            R_z[1, 0] = sin_theta
            R_z[1, 1] = cos_theta
            
            # Apply rotation to all poses
            T_aug = R_z @ T_aug
        
        # 2) Add noise to positions
        if self.position_noise_std > 0:
            pos_noise = torch.randn_like(T_aug[:, :3, 3]) * self.position_noise_std
            T_aug[:, :3, 3] += pos_noise
        
        # 3) Add rotation noise (small random rotations)
        if self.rotation_noise_std > 0:
            for i in range(T_aug.shape[0]):
                # Small random rotation around random axis
                axis = torch.randn(3)
                axis = axis / torch.norm(axis)
                angle = torch.randn(1) * self.rotation_noise_std
                
                # Rodrigues formula for rotation
                K = torch.zeros(3, 3)
                K[0, 1] = -axis[2]
                K[0, 2] = axis[1]
                K[1, 0] = axis[2]
                K[1, 2] = -axis[0]
                K[2, 0] = -axis[1]
                K[2, 1] = axis[0]
                
                R_noise = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
                T_aug[i, :3, :3] = T_aug[i, :3, :3] @ R_noise
        
        return T_aug
    
    def __len__(self) -> int:
        """데이터셋 크기"""
        return len(self.trajectory_list)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        단일 궤적 데이터 반환
        
        Returns:
            Dict containing:
            - 'T_processed': [M, 4, 4] 처리된 SE(3) 궤적
            - 'xi_labels': [M, 6] body twist 라벨
            - 'dt_seq': [M-1] 시간 간격
            - 'T_raw': [N, 4, 4] 원시 궤적 (참조용)
            - 'T_smooth': [N, 4, 4] 스무딩된 궤적
            - 'metadata': 메타데이터
        """
        traj_info = self.trajectory_list[index]
        
        # 궤적 처리
        processed_data = self._process_trajectory(traj_info['raw_trajectory'])
        
        # 증강 적용 (훈련 시에만)
        T_processed = self._apply_augmentation(processed_data['T_processed'])
        T_raw = self._apply_augmentation(processed_data['T_raw'])
        T_smooth = self._apply_augmentation(processed_data['T_smooth'])
        
        # 메타데이터
        metadata = {
            'env_id': traj_info['env_id'],
            'rb_id': traj_info['rb_id'], 
            'pair_index': traj_info['pair_index'],
            'split': self.split,
            'original_length': len(traj_info['raw_trajectory']),
            'processed_length': len(T_processed)
        }
        
        return {
            'T_processed': T_processed,           # [M, 4, 4]
            'xi_labels': processed_data['xi_labels'],  # [M, 6]
            'dt_seq': processed_data['dt_seq'],        # [M-1]
            'T_raw': T_raw,                       # [N, 4, 4]
            'T_smooth': T_smooth,                 # [N, 4, 4]
            'metadata': metadata
        }
    
    def get_dataloader(self, batch_size: int = 32, shuffle: Optional[bool] = None, 
                      num_workers: int = 4, **kwargs) -> torch.utils.data.DataLoader:
        """
        DataLoader 생성 (fm-main 패턴 참조)
        
        Args:
            batch_size: 배치 크기
            shuffle: 셔플 여부 (None이면 split에 따라 자동 결정)
            num_workers: worker 수
            **kwargs: DataLoader 추가 인수
        
        Returns:
            torch.utils.data.DataLoader
        """
        if shuffle is None:
            shuffle = (self.split == 'train')
        
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        배치 생성을 위한 collate function
        각 궤적의 길이가 다를 수 있으므로 패딩 처리
        """
        # 최대 길이 찾기
        max_len_processed = max(len(item['T_processed']) for item in batch)
        max_len_raw = max(len(item['T_raw']) for item in batch)
        batch_size = len(batch)
        
        # 패딩된 텐서 생성
        T_processed_batch = torch.zeros(batch_size, max_len_processed, 4, 4)
        xi_labels_batch = torch.zeros(batch_size, max_len_processed, 6)
        dt_seq_batch = torch.zeros(batch_size, max_len_processed - 1)
        T_raw_batch = torch.zeros(batch_size, max_len_raw, 4, 4)
        T_smooth_batch = torch.zeros(batch_size, max_len_raw, 4, 4)
        
        # 길이 마스크
        lengths_processed = torch.zeros(batch_size, dtype=torch.long)
        lengths_raw = torch.zeros(batch_size, dtype=torch.long)
        
        # 메타데이터
        metadata_batch = []
        
        for i, item in enumerate(batch):
            len_processed = len(item['T_processed'])
            len_raw = len(item['T_raw'])
            
            T_processed_batch[i, :len_processed] = item['T_processed']
            xi_labels_batch[i, :len_processed] = item['xi_labels']
            dt_seq_batch[i, :len_processed-1] = item['dt_seq']
            T_raw_batch[i, :len_raw] = item['T_raw']
            T_smooth_batch[i, :len_raw] = item['T_smooth']
            
            lengths_processed[i] = len_processed
            lengths_raw[i] = len_raw
            metadata_batch.append(item['metadata'])
        
        return {
            'T_processed': T_processed_batch,    # [B, M, 4, 4]
            'xi_labels': xi_labels_batch,        # [B, M, 6]
            'dt_seq': dt_seq_batch,              # [B, M-1]
            'T_raw': T_raw_batch,                # [B, N, 4, 4]
            'T_smooth': T_smooth_batch,          # [B, N, 4, 4]
            'lengths_processed': lengths_processed,  # [B]
            'lengths_raw': lengths_raw,          # [B]
            'metadata': metadata_batch           # List[Dict]
        }
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'hdf5_loader'):
            self.hdf5_loader.close()
    
    def __del__(self):
        """소멸자"""
        self.close()


def get_se3_dataloaders(hdf5_path: str, 
                       train_config: Dict[str, Any],
                       val_config: Dict[str, Any],
                       test_config: Dict[str, Any]) -> Tuple[torch.utils.data.DataLoader, 
                                                           torch.utils.data.DataLoader,
                                                           torch.utils.data.DataLoader]:
    """
    SE(3) 데이터로더 팩토리 함수 (fm-main 스타일)
    
    Args:
        hdf5_path: HDF5 파일 경로
        train_config: 훈련 설정
        val_config: 검증 설정  
        test_config: 테스트 설정
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 데이터셋 생성
    train_dataset = SE3TrajectoryDataset(hdf5_path, split='train', **train_config)
    val_dataset = SE3TrajectoryDataset(hdf5_path, split='valid', **val_config)
    test_dataset = SE3TrajectoryDataset(hdf5_path, split='test', **test_config)
    
    # DataLoader 생성
    train_loader = train_dataset.get_dataloader(shuffle=True)
    val_loader = val_dataset.get_dataloader(shuffle=False)
    test_loader = test_dataset.get_dataloader(shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """테스트 실행"""
    print("🧪 Testing SE3TrajectoryDataset")
    
    test_hdf5_path = "/Users/a123/Documents/Projects/2D_sim/packages/data_generator/test_trajectory_dataset.h5"
    
    try:
        # 간단한 설정으로 데이터셋 테스트
        config = {
            'use_smoothing': True,
            'smooth_strength': 0.1,
            'num_samples': 50,
            'augmentation': True,
            'max_trajectories': 5
        }
        
        # 데이터셋 생성
        dataset = SE3TrajectoryDataset(test_hdf5_path, split='train', **config)
        
        print(f"\n📊 Dataset Info:")
        print(f"   Length: {len(dataset)}")
        
        # 샘플 데이터 확인
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n🔍 Sample Data:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {value}")
            
            # DataLoader 테스트
            dataloader = dataset.get_dataloader(batch_size=2)
            batch = next(iter(dataloader))
            print(f"\n📦 Batch Data:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {len(value) if isinstance(value, list) else value}")
        
        dataset.close()
        print("\n✅ Test completed successfully!")
        
    except FileNotFoundError:
        print(f"⚠️ Test file not found: {test_hdf5_path}")
        print("   Create HDF5 data first using hdf5_schema_creator.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()