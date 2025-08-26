import torch
import os
import numpy as np
import json
import h5py
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import glob
import open3d as o3d
from copy import deepcopy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.motion_utils import twist_to_se3_matrix
from utils.normalization import TwistNormalizer


class MotionDataset4RCFM(torch.utils.data.Dataset):
    """
    Dual-format Motion Planning Dataset for RCFM
    - Trajectories: JSON + H5 support (auto-detection)
    - Point clouds: PLY + OBJ support (auto-detection)
    - fm-main style structure with motion planning data
    """
    
    def __init__(self,
                 trajectory_root,
                 pointcloud_root,
                 split='train',
                 max_trajectories=None,
                 use_bsplined=True,
                 num_point_cloud=2000,
                 num_twists=1000,  # fm-main의 num_grasps와 동일한 역할
                 scale=1.0,
                 augmentation=True,
                 normalize_twist=True,
                 normalization_stats_path="configs/normalization_stats.json",
                 **kwargs):
        
        self.trajectory_root = trajectory_root
        self.pointcloud_root = pointcloud_root
        self.split = split
        self.max_trajectories = max_trajectories
        self.use_bsplined = use_bsplined
        self.num_point_cloud = num_point_cloud
        self.num_twists = num_twists
        self.scale = scale
        self.augmentation = augmentation and (split == 'train')
        
        # 정규화 설정
        self.normalize_twist = normalize_twist
        if normalize_twist:
            try:
                self.normalizer = TwistNormalizer(normalization_stats_path)
                print(f"✅ 정규화 활성화: {normalization_stats_path}")
            except:
                print(f"⚠️ 정규화 통계 파일 없음, 정규화 비활성화")
                self.normalize_twist = False
                self.normalizer = None
        else:
            self.normalizer = None
            print("📝 정규화 비활성화")
        
        print(f"🚀 MotionDataset4RCFM init - split: {split}")
        
        # 궤적 데이터 수집
        self.trajectory_files = self._collect_trajectory_files()
        print(f"📁 Found {len(self.trajectory_files)} trajectory files")
        
        # 샘플 생성
        self.samples = self._create_samples()
        print(f"📊 Created {len(self.samples)} samples")
        
        # CFM을 위한 그룹핑 (환경별)
        self._group_samples_by_env()
        print(f"🗂️ Grouped into {len(self.env_groups)} environments")
    
    def _collect_trajectory_files(self):
        """궤적 파일 수집 (JSON + H5 자동 감지)"""
        files = []
        
        # JSON 파일 패턴
        if self.use_bsplined:
            json_pattern = os.path.join(self.trajectory_root, "*_bsplined.json")
        else:
            json_pattern = os.path.join(self.trajectory_root, "*_traj_rb3.json")
        
        json_files = sorted(glob.glob(json_pattern))
        
        # H5 파일 패턴 (fm-main 호환)
        h5_pattern = os.path.join(self.trajectory_root, "*.h5")
        h5_files = sorted(glob.glob(h5_pattern))
        
        # 파일 형식 우선순위: JSON > H5
        files = json_files + h5_files
        
        if self.max_trajectories:
            files = files[:self.max_trajectories]
        
        print(f"📁 Found {len(json_files)} JSON + {len(h5_files)} H5 trajectory files")
        return files
    
    def _load_h5_trajectory(self, h5_file):
        """H5 궤적 파일 로딩 (fm-main 호환)"""
        try:
            with h5py.File(h5_file, 'r') as f:
                # fm-main H5 구조 추정
                if 'poses' in f and 'timestamps' in f:
                    poses = f['poses'][:]  # [N, 4, 4] SE(3) matrices
                    timestamps = f['timestamps'][:]  # [N] timestamps
                elif 'trajectory' in f:
                    traj_group = f['trajectory']
                    poses = traj_group['poses'][:]
                    timestamps = traj_group['timestamps'][:]
                else:
                    # 기본 구조 시도
                    keys = list(f.keys())
                    print(f"⚠️ Unknown H5 structure in {h5_file}, keys: {keys}")
                    return None
                
                # 환경 ID 추출 (파일명에서)
                env_id = os.path.basename(h5_file).replace('.h5', '')
                
                return {
                    'poses': poses,
                    'timestamps': timestamps,
                    'env_id': env_id
                }
                
        except Exception as e:
            print(f"❌ Error loading H5 file {h5_file}: {e}")
            return None
    
    def _detect_pointcloud_file(self, env_id):
        """포인트클라우드 파일 자동 감지 (PLY -> OBJ 순서)"""
        # PLY 우선 시도
        ply_file = os.path.join(self.pointcloud_root, f"{env_id}.ply")
        if os.path.exists(ply_file):
            return ply_file, 'ply'
        
        # OBJ 시도
        obj_file = os.path.join(self.pointcloud_root, f"{env_id}.obj")
        if os.path.exists(obj_file):
            return obj_file, 'obj'
        
        # 둘 다 없음
        print(f"⚠️ No pointcloud found for {env_id} (checked .ply and .obj)")
        return None, None
    
    def _create_samples(self):
        """CFM용 twist vector 샘플 생성 (JSON + H5 지원)"""
        samples = []
        
        for traj_file in self.trajectory_files:
            try:
                # 파일 형식 감지
                if traj_file.endswith('.json'):
                    # JSON 파일 처리
                    with open(traj_file, 'r') as f:
                        traj_data = json.load(f)
                    
                    # 환경 정보
                    env_name = traj_data.get('environment', {}).get('name', 'unknown')
                    env_id = env_name
                    pair_id = traj_data.get('pair_id', 0)
                    
                    # 궤적 데이터 파싱
                    waypoints = self._parse_trajectory(traj_data)
                    
                elif traj_file.endswith('.h5'):
                    # H5 파일 처리
                    h5_data = self._load_h5_trajectory(traj_file)
                    if h5_data is None:
                        continue
                    
                    env_id = h5_data['env_id']
                    pair_id = 0
                    
                    # H5에서 waypoint 변환
                    waypoints = []
                    for i, (pose, timestamp) in enumerate(zip(h5_data['poses'], h5_data['timestamps'])):
                        waypoints.append({
                            'pose': pose,
                            'timestamp': timestamp
                        })
                
                else:
                    print(f"⚠️ Unsupported trajectory format: {traj_file}")
                    continue
                
                if len(waypoints) < 2:
                    continue
                
                # 포인트클라우드 파일 자동 감지
                pc_file, pc_format = self._detect_pointcloud_file(env_id)
                if pc_file is None:
                    continue
                
                # 각 waypoint에서 twist vector 추출
                twist_vectors = []
                target_poses = []
                
                for i, waypoint in enumerate(waypoints):
                    # 목표 twist vector (CFM의 x1 역할)
                    if i < len(waypoints) - 1:
                        next_wp = waypoints[i + 1]
                        dt = next_wp['timestamp'] - waypoint['timestamp']
                        if dt <= 0:
                            dt = 0.1
                        
                        twist = self._compute_twist_vector(
                            waypoint['pose'], next_wp['pose'], dt
                        )
                    else:
                        twist = np.zeros(6)  # 정지
                    
                    twist_vectors.append(twist)
                    target_poses.append(waypoints[-1]['pose'])  # 최종 목표 pose
                
                # 샘플 저장 (fm-main 스타일)
                sample = {
                    'env_id': env_id,
                    'pc_file': pc_file,
                    'pc_format': pc_format,  # 포인트클라우드 형식 정보
                    'twist_vectors': np.array(twist_vectors),  # [N, 6] - CFM 타겟
                    'target_poses': np.array(target_poses),    # [N, 4, 4] - 조건
                    'traj_file': traj_file,
                    'traj_format': 'json' if traj_file.endswith('.json') else 'h5'
                }
                
                samples.append(sample)
                
            except Exception as e:
                print(f"❌ Error processing {traj_file}: {e}")
                continue
        
        return samples
    
    def _parse_trajectory(self, traj_data):
        """JSON 궤적 → waypoint 리스트"""
        path_data = traj_data.get('path', {})
        timestamps = path_data.get('timestamps', [])
        poses_flat = path_data.get('data', [])
        
        waypoints = []
        num_waypoints = min(len(timestamps), len(poses_flat))
        
        for i in range(num_waypoints):
            timestamp = timestamps[i] if i < len(timestamps) else 0.0
            pose_6d = poses_flat[i]
            
            if len(pose_6d) >= 6:
                x, y, z, rx, ry, rz = pose_6d[:6]
                
                # SE(3) 행렬 생성
                R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = R
                pose_matrix[:3, 3] = [x, y, z]
                
                waypoints.append({
                    'pose': pose_matrix,
                    'timestamp': timestamp
                })
        
        return waypoints
    
    def _compute_twist_vector(self, T1, T2, dt):
        """SE(3) pose 간 twist vector 계산 (body frame)"""
        # Translation (body frame)
        R1 = T1[:3, :3]
        p1, p2 = T1[:3, 3], T2[:3, 3]
        v_body = R1.T @ (p2 - p1) / dt
        
        # Rotation (body frame) 
        R2 = T2[:3, :3]
        R_rel = R1.T @ R2
        r_rel = Rotation.from_matrix(R_rel)
        w_body = r_rel.as_rotvec() / dt
        
        return np.concatenate([w_body, v_body])  # [ω, v]
    
    def _group_samples_by_env(self):
        """환경별 샘플 그룹핑 (CFM을 위한 배치 구성)"""
        self.env_groups = {}
        for i, sample in enumerate(self.samples):
            env_id = sample['env_id']
            if env_id not in self.env_groups:
                self.env_groups[env_id] = []
            self.env_groups[env_id].append(i)
        
        self.env_list = list(self.env_groups.keys())
    
    def __len__(self):
        return len(self.env_list)  # 환경 단위로 iterate
    
    def __getitem__(self, index):
        """fm-main 스타일 배치 반환"""
        env_id = self.env_list[index]
        sample_indices = self.env_groups[env_id]
        
        # 환경의 대표 샘플 선택
        main_sample = self.samples[sample_indices[0]]
        
        # 포인트클라우드 로딩 (안전, 자동 형식 감지)
        pc = self._load_pointcloud_safe(main_sample['pc_file'], pc_format='auto')
        
        # twist vectors 샘플링 (CFM x1 역할)
        twist_vectors = main_sample['twist_vectors']
        target_poses = main_sample['target_poses']
        
        # 샘플 개수 조정
        if len(twist_vectors) > self.num_twists:
            indices = np.random.choice(len(twist_vectors), self.num_twists, replace=False)
        else:
            indices = np.random.choice(len(twist_vectors), self.num_twists, replace=True)
        
        selected_twists = twist_vectors[indices]
        selected_targets = target_poses[indices]
        
        # Twist 정규화 적용 (CFM 훈련용)
        if self.normalize_twist and self.normalizer is not None:
            selected_twists = self.normalizer.normalize_twist(selected_twists)
        
        # Twist vectors를 SE(3) 매트릭스로 변환 (CFM 호환)
        selected_twists_torch = torch.FloatTensor(selected_twists)
        Ts_twist = twist_to_se3_matrix(selected_twists_torch).numpy()
        
        # fm-main 호환: [num_twists, 4, 4] 형태로 유지, 패딩으로 num_twists 개 맞춤
        # 유효하지 않은 grasp는 마지막 행이 [0,0,0,0]이 되도록 패딩
        Ts_grasp_padded = np.zeros((self.num_twists, 4, 4))
        num_valid = min(len(Ts_twist), self.num_twists)
        
        if num_valid > 0:
            Ts_grasp_padded[:num_valid] = Ts_twist[:num_valid]
            # 유효한 SE(3) 행렬은 마지막 행이 [0,0,0,1]
            Ts_grasp_padded[:num_valid, 3, 3] = 1.0
        
        # 디버깅: 실제로는 모든 twist를 사용하지 말고 일부만 사용하도록 제한
        # CFM 훈련에서는 너무 많은 target이 있으면 문제가 될 수 있음
        max_valid_twists = min(100, num_valid)  # 최대 100개만 유효하게 설정
        if max_valid_twists < self.num_twists:
            # 100개 이후는 무효한 SE(3)로 설정 (마지막 행이 [0,0,0,0])
            Ts_grasp_padded[max_valid_twists:, 3, 3] = 0.0
        
        return {
            'pc': torch.FloatTensor(pc),
            'Ts_grasp': torch.FloatTensor(Ts_grasp_padded),  # [num_twists, 4, 4] fm-main 호환
            'target_poses': torch.FloatTensor(selected_targets),
            'env_id': env_id
        }
    
    def _load_obj_pointcloud(self, obj_file):
        """OBJ 파일에서 포인트클라우드 로딩 (fm-main 호환)"""
        vertices = []
        try:
            with open(obj_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):  # vertex line
                        parts = line.split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append([x, y, z])
            
            if len(vertices) == 0:
                print(f"⚠️ No vertices found in OBJ file: {obj_file}")
                return self._generate_fallback_pointcloud()
            
            return np.array(vertices, dtype=np.float32)
            
        except Exception as e:
            print(f"❌ OBJ loading failed: {obj_file}, error: {e}")
            return self._generate_fallback_pointcloud()
    
    def _load_pointcloud_safe(self, pc_file, pc_format='auto'):
        """안전한 포인트클라우드 로딩 (PLY + OBJ 지원)"""
        try:
            # 형식 자동 감지
            if pc_format == 'auto':
                if pc_file.endswith('.ply'):
                    pc_format = 'ply'
                elif pc_file.endswith('.obj'):
                    pc_format = 'obj'
                else:
                    print(f"⚠️ Unknown pointcloud format: {pc_file}")
                    return self._generate_fallback_pointcloud()
            
            # 형식별 로딩
            if pc_format == 'ply':
                o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
                mesh = o3d.io.read_triangle_mesh(pc_file)
                points = np.asarray(mesh.vertices)
            elif pc_format == 'obj':
                points = self._load_obj_pointcloud(pc_file)
            else:
                raise ValueError(f"Unsupported format: {pc_format}")
            
            if len(points) == 0:
                points = self._generate_fallback_pointcloud()
            
            # 스케일링
            if self.scale != 1.0:
                points *= self.scale
            
            # 포인트 수 조정
            if len(points) > self.num_point_cloud:
                indices = np.random.choice(len(points), self.num_point_cloud, replace=False)
                points = points[indices]
            elif len(points) < self.num_point_cloud:
                indices = np.random.choice(len(points), self.num_point_cloud, replace=True)
                points = points[indices]
            
            # 데이터 증강
            if self.augmentation:
                points = self._augment_pointcloud(points)
            
            return points.astype(np.float32)
            
        except Exception as e:
            print(f"❌ PC loading failed: {pc_file} ({pc_format}), error: {e}")
            return self._generate_fallback_pointcloud()
    
    def _generate_fallback_pointcloud(self):
        """대체 포인트클라우드 생성"""
        # 원형 환경 시뮬레이션
        angles = np.linspace(0, 2*np.pi, self.num_point_cloud//2)
        outer_x = 3.0 * np.cos(angles)
        outer_y = 3.0 * np.sin(angles)
        outer_z = np.zeros_like(outer_x)
        
        # 내부 장애물
        inner_x = np.random.uniform(-2, 2, self.num_point_cloud//2)
        inner_y = np.random.uniform(-2, 2, self.num_point_cloud//2)
        inner_z = np.random.uniform(0, 1, self.num_point_cloud//2)
        
        points = np.vstack([
            np.column_stack([outer_x, outer_y, outer_z]),
            np.column_stack([inner_x, inner_y, inner_z])
        ])
        
        return points.astype(np.float32)
    
    def _augment_pointcloud(self, pointcloud):
        """포인트클라우드 증강"""
        # Z축 회전
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0],
                     [sin_a, cos_a, 0],
                     [0, 0, 1]])
        
        pointcloud = (R @ pointcloud.T).T
        
        # 노이즈
        noise = np.random.normal(0, 0.01, pointcloud.shape)
        pointcloud += noise
        
        return pointcloud