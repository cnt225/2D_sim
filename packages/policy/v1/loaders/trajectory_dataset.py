import torch
import os
import numpy as np
import json
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import glob
import sys
from pathlib import Path

# 정규화 유틸리티 import
sys.path.append(str(Path(__file__).parent.parent))
from utils.normalization import TwistNormalizer


class TrajectoryDataset(Dataset):
    """
    궤적 데이터에서 motion planning을 위한 데이터셋
    
    각 waypoint에서:
    - current_T: 현재 SE(3) pose
    - target_T: 궤적의 최종 목표 pose (고정)
    - pointcloud: 환경 포인트클라우드
    - time_t: 정규화된 시간 [0,1]
    - T_dot: velocity (역산)
    """
    
    def __init__(self, 
                 trajectory_root,
                 pointcloud_root,
                 split='train',
                 max_trajectories=None,
                 use_bsplined=True,
                 augment_data=True,
                 num_points=1000,
                 normalize_twist=True,
                 normalization_stats_path="configs/normalization_stats.json",
                 **kwargs):
        
        self.trajectory_root = trajectory_root
        self.pointcloud_root = pointcloud_root
        self.split = split
        self.max_trajectories = max_trajectories
        self.use_bsplined = use_bsplined
        self.augment_data = augment_data
        self.num_points = num_points
        self.normalize_twist = normalize_twist
        
        # 정규화 설정
        if normalize_twist:
            self.twist_normalizer = TwistNormalizer(stats_path=normalization_stats_path)
            print(f"✅ Twist 정규화 활성화: {normalization_stats_path}")
        else:
            self.twist_normalizer = None
            print("⚠️ Twist 정규화 비활성화")
        
        # 궤적 파일들 수집
        self.trajectory_files = self._collect_trajectory_files()
        
        # 데이터 샘플들 생성
        self.samples = self._create_samples()
        
        print(f"TrajectoryDataset {split}: {len(self.samples)} samples from {len(self.trajectory_files)} trajectories")
    
    def _collect_trajectory_files(self):
        """궤적 파일들을 수집"""
        if self.use_bsplined:
            pattern = os.path.join(self.trajectory_root, "*_bsplined.json")
        else:
            pattern = os.path.join(self.trajectory_root, "*_traj_rb3.json")
        
        files = sorted(glob.glob(pattern))
        
        if self.max_trajectories:
            files = files[:self.max_trajectories]
        
        return files
    
    def _create_samples(self):
        """각 궤적에서 waypoint별 샘플 생성"""
        samples = []
        
        for traj_file in self.trajectory_files:
            # 궤적 데이터 로드
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)
            
            # 환경 ID 추출
            env_name = traj_data.get('environment', {}).get('name', 'unknown')
            env_id = env_name  # circle_env_000000 형태
            pair_id = traj_data.get('pair_id', 0)
            
            # 포인트클라우드 파일 경로 (.ply 파일)
            pc_file = os.path.join(self.pointcloud_root, f"{env_id}.ply")
            if not os.path.exists(pc_file):
                print(f"Warning: Pointcloud file not found: {pc_file}")
                continue
            
            # 궤적 waypoints 추출
            path_data = traj_data.get('path', {})
            timestamps = path_data.get('timestamps', [])
            poses_flat = path_data.get('data', [])
            
            # 6D poses (x,y,z,rx,ry,rz)를 4x4 SE(3) 행렬로 변환
            waypoints = []
            if len(poses_flat) > 0:  # poses_flat이 비어있지 않으면
                # timestamps 개수만큼만 처리
                num_waypoints = min(len(timestamps), len(poses_flat))
                for i in range(num_waypoints):
                    timestamp = timestamps[i] if i < len(timestamps) else 0.0
                    pose_6d = poses_flat[i]  # [x, y, z, rx, ry, rz]
                    if len(pose_6d) >= 6:
                        # SE(3) 변환행렬 생성
                        x, y, z, rx, ry, rz = pose_6d[:6]
                        
                        # Rotation matrix from Euler angles (assuming ZYX order)
                        from scipy.spatial.transform import Rotation
                        R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
                        
                        # SE(3) matrix
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = R
                        pose_matrix[:3, 3] = [x, y, z]
                        
                        waypoints.append({
                            'pose': pose_matrix.tolist(),
                            'timestamp': timestamp
                        })
            
            if len(waypoints) < 2:
                continue
            
            # 각 waypoint에서 샘플 생성
            for i, waypoint in enumerate(waypoints):
                # 시간 정규화
                time_t = i / (len(waypoints) - 1)
                
                # current pose
                current_T = np.array(waypoint['pose'])
                
                # target pose (마지막 waypoint)
                target_T = np.array(waypoints[-1]['pose'])
                
                # 개선된 SE3 velocity 계산
                if i < len(waypoints) - 1:
                    next_T = np.array(waypoints[i + 1]['pose'])
                    dt = waypoints[i + 1].get('timestamp', 1.0) - waypoint.get('timestamp', 0.0)
                    if dt <= 0:
                        dt = 0.1  # 기본값
                    
                    # 개선된 SE3 velocity 계산
                    T_dot = self._compute_velocity_improved(current_T, next_T, dt)
                else:
                    # 마지막 점에서는 정지
                    T_dot = np.zeros(6)
                
                sample = {
                    'traj_file': traj_file,
                    'pc_file': pc_file,
                    'waypoint_idx': i,
                    'current_T': current_T,
                    'target_T': target_T,
                    'time_t': time_t,
                    'T_dot': T_dot,
                    'env_id': env_id,
                    'pair_id': pair_id
                }
                
                samples.append(sample)
        
        return samples
    
    def _compute_velocity(self, T1, T2, dt):
        """두 SE(3) pose 간의 velocity 계산 (SE(3) Lie algebra 사용)"""
        
        # Translation velocity
        v_trans = (T2[:3, 3] - T1[:3, 3]) / dt
        
        # Rotation velocity (proper SE(3) computation)
        R1 = T1[:3, :3]
        R2 = T2[:3, :3]
        
        # Relative rotation: R_rel = R2 * R1^T
        R_rel = R2 @ R1.T
        
        # Extract angular velocity from rotation matrix
        # For small rotations: R ≈ I + [ω]_× * dt
        # where [ω]_× is the skew-symmetric matrix of angular velocity
        
        # Use Rodrigues' formula to extract axis-angle representation
        from scipy.spatial.transform import Rotation as R_scipy
        r_rel = R_scipy.from_matrix(R_rel)
        axis_angle = r_rel.as_rotvec()
        
        # Angular velocity: ω = axis_angle / dt
        v_rot = axis_angle / dt
        
        return np.concatenate([v_rot, v_trans])  # [ωx, ωy, ωz, vx, vy, vz] 형태
    
    def _compute_velocity_improved(self, T1, T2, dt):
        """개선된 SE(3) velocity 계산 (body frame 기준)"""
        # Translation velocity (body frame)
        R1 = T1[:3, :3]
        p1 = T1[:3, 3]
        p2 = T2[:3, 3]
        # Body frame에서의 선속도: R1^T * (p2 - p1) / dt
        v_body = R1.T @ (p2 - p1) / dt
        
        # Angular velocity (body frame)
        R2 = T2[:3, :3]
        R_rel = R1.T @ R2  # Body frame에서의 상대 회전
        
        from scipy.spatial.transform import Rotation as R_scipy
        r_rel = R_scipy.from_matrix(R_rel)
        axis_angle = r_rel.as_rotvec()
        w_body = axis_angle / dt
        
        # Twist vector: [ωx, ωy, ωz, vx, vy, vz] (body frame)
        return np.concatenate([w_body, v_body])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 포인트클라우드 로드 (.ply 파일)
        import open3d as o3d
        # 경고 메시지 완전 억제
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        import os
        import warnings
        warnings.filterwarnings('ignore')
        os.environ['PYTHONWARNINGS'] = 'ignore'
        
        try:
            mesh = o3d.io.read_triangle_mesh(sample['pc_file'])
            points = np.asarray(mesh.vertices)
            
            # 빈 파일이나 손상된 파일 처리
            if len(points) == 0:
                # 대체 포인트클라우드 생성
                points = self._generate_fallback_pointcloud()
        except Exception as e:
            # 완전히 손상된 파일의 경우 대체 포인트클라우드 생성
            points = self._generate_fallback_pointcloud()
        
        pointcloud = points
        
        # 포인트 수 조정
        if len(pointcloud) > self.num_points:
            # 랜덤 샘플링
            indices = np.random.choice(len(pointcloud), self.num_points, replace=False)
            pointcloud = pointcloud[indices]
        elif len(pointcloud) < self.num_points:
            # 중복 샘플링으로 채우기
            indices = np.random.choice(len(pointcloud), self.num_points, replace=True)
            pointcloud = pointcloud[indices]
        
        # 데이터 증강 (훈련 시에만)
        if self.split == 'train' and self.augment_data:
            pointcloud = self._augment_pointcloud(pointcloud)
        
        # Twist 정규화 (학습 시)
        T_dot = sample['T_dot']
        if self.twist_normalizer is not None:
            T_dot = self.twist_normalizer.normalize_twist(T_dot)
        
        return {
            'current_T': torch.FloatTensor(sample['current_T']),
            'target_T': torch.FloatTensor(sample['target_T']),
            'time_t': torch.FloatTensor([sample['time_t']]),
            'pointcloud': torch.FloatTensor(pointcloud).transpose(0, 1),  # [N, 3] -> [3, N] for DGCNN
            'T_dot': torch.FloatTensor(T_dot),
            'env_id': sample['env_id'],
            'pair_id': sample['pair_id']
        }
    
    def _augment_pointcloud(self, pointcloud):
        """포인트클라우드 증강"""
        # Z축 중심 회전 (2D 환경이므로)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0],
                     [sin_a, cos_a, 0],
                     [0, 0, 1]])
        
        pointcloud = (R @ pointcloud.T).T
        
        # 약간의 노이즈 추가
        noise = np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud += noise
        
        return pointcloud
    
    def _generate_fallback_pointcloud(self):
        """대체 포인트클라우드 생성 (원형 환경)"""
        # 원형 환경에 맞는 간단한 포인트클라우드 생성
        angles = np.linspace(0, 2*np.pi, self.num_points//2)
        # 외곽 원
        outer_x = 3.0 * np.cos(angles)
        outer_y = 3.0 * np.sin(angles)
        outer_z = np.zeros_like(outer_x)
        # 내부 장애물
        inner_x = np.random.uniform(-2, 2, self.num_points//2)
        inner_y = np.random.uniform(-2, 2, self.num_points//2)
        inner_z = np.random.uniform(0, 1, self.num_points//2)
        
        points = np.vstack([
            np.column_stack([outer_x, outer_y, outer_z]),
            np.column_stack([inner_x, inner_y, inner_z])
        ])
        return points.astype(np.float32)
    