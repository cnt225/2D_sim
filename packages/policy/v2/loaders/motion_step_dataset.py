import os, json, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import sys
sys.path.append('/home/dhkang225/2D_sim/packages/policy/v2')
from utils.normalization import TwistNormalizer
from utils.motion_utils import se3_matrix_to_twist

class MotionStepDataset(Dataset):
    """Per-step motion dataset: current_T, target_T, time_t, pc, g(3), T_dot(6), delta_T(4x4)
    Expects paths constructed from env_id and pair index.
    """
    def __init__(self,
                 trajectory_root,
                 pointcloud_root,
                 split='train',
                 max_trajectories=None,
                 num_point_cloud=2000,
                 normalize_twist=False,
                 normalization_stats_path=None,
                 exclude_corrupted_files=False,
                 **kwargs):
        self.pointcloud_root = pointcloud_root
        self.tdelta_root = trajectory_root  # SE(3) delta_T files
        self.traj_root = trajectory_root.replace('_tdelta_se3_velocity', '').replace('_tdelta_se3', '')  # Original trajectory files
        self.pairs_root = self.traj_root  # pairs are in same directory as trajectories
        self.split = split
        self.max_trajectories = max_trajectories
        self.num_point_cloud = num_point_cloud
        self.radii = torch.tensor([0.05, 0.05, 0.10], dtype=torch.float32)  # Default robot geometry
        
        # Load corrupted environment IDs to exclude
        self.exclude_corrupted_files = exclude_corrupted_files
        self.corrupted_env_ids = set()
        if exclude_corrupted_files:
            corrupted_list_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'corrupted_env_ids.json')
            if os.path.exists(corrupted_list_path):
                with open(corrupted_list_path, 'r') as f:
                    self.corrupted_env_ids = set(json.load(f))
                print(f"🚫 Excluding {len(self.corrupted_env_ids)} corrupted environments from training")
            else:
                print(f"⚠️ Corrupted env list not found: {corrupted_list_path}")
        
        # Initialize normalizer
        self.normalize_twist = normalize_twist
        self.normalizer = None
        if normalize_twist and normalization_stats_path:
            self.normalizer = TwistNormalizer(stats_path=normalization_stats_path)

        self.samples = self._index_all()

    def _index_all(self):
        files = sorted(glob.glob(os.path.join(self.tdelta_root, '*.json')))
        if self.max_trajectories:
            files = files[:self.max_trajectories]
        samples = []
        excluded_count = 0
        for fp in files:
            base = os.path.basename(fp)
            # parse env id and pair
            # circle_env_000001_pair_1_traj_rb3_bsplined_tdot.json -> env=...000001, pair=1
            try:
                parts = base.split('_')
                env_id = '_'.join(parts[:3])  # circle_env_000001
                pair_idx = int(parts[4])
                
                # Check if this environment should be excluded
                if self.exclude_corrupted_files:
                    env_number = parts[2]  # 000001 from circle_env_000001
                    if env_number in self.corrupted_env_ids:
                        excluded_count += 1
                        continue
                        
            except Exception:
                continue
            samples.append((env_id, pair_idx))
        
        if self.exclude_corrupted_files and excluded_count > 0:
            print(f"🚫 Excluded {excluded_count} samples from corrupted environments")
            print(f"✅ Using {len(samples)} clean samples for {self.split}")
            
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_pointcloud(self, env_id):
        import open3d as o3d
        ply = os.path.join(self.pointcloud_root, f"{env_id}.ply")
        if not os.path.exists(ply):
            # minimal fallback
            pts = np.random.randn(self.num_point_cloud, 3).astype(np.float32)
            return torch.from_numpy(pts)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        mesh = o3d.io.read_triangle_mesh(ply)
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            pts = np.random.randn(self.num_point_cloud, 3)
        if len(pts) > self.num_point_cloud:
            idx = np.random.choice(len(pts), self.num_point_cloud, replace=False)
            pts = pts[idx]
        elif len(pts) < self.num_point_cloud:
            idx = np.random.choice(len(pts), self.num_point_cloud, replace=True)
            pts = pts[idx]
        return torch.from_numpy(pts.astype(np.float32))

    def _load_pairs(self, env_id):
        pairs_path = os.path.join(self.pairs_root, f"{env_id}_rb_3_pairs.json")
        with open(pairs_path,'r') as f:
            return json.load(f)

    def _rotvec_to_R(self, rv):
        return Rotation.from_rotvec(rv).as_matrix()

    def _se3_from_6d(self, pose6):
        x,y,z, rx,ry,rz = pose6[:6]
        R = self._rotvec_to_R([rx,ry,rz])
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = R
        T[:3,3] = [x,y,z]
        return torch.from_numpy(T)

    def __getitem__(self, idx):
        env_id, pair_idx = self.samples[idx]
        # paths
        traj = os.path.join(self.traj_root, f"{env_id}_pair_{pair_idx}_traj_rb3_bsplined.json")
        tdelta = os.path.join(self.tdelta_root, f"{env_id}_pair_{pair_idx}_traj_rb3_bsplined_tdot.json")
        with open(traj,'r') as f:
            td = json.load(f)
        with open(tdelta,'r') as f:
            dd = json.load(f)
        timestamps = td['path']['timestamps']
        poses6 = td['path']['data']
        # Check if converted data has delta_T (it's in path subdictionary)
        if 'path' in dd and 'delta_T' in dd['path'] and 'dt' in dd:
            # New converted SE3 format
            delta_T = dd['path']['delta_T']  # [N,4,4]
            dt_val = dd['dt']
            delta = torch.tensor(delta_T[0], dtype=torch.float32)
            T_dot = se3_matrix_to_twist(delta.unsqueeze(0)).squeeze(0)  # Keep as displacement for stable learning
        else:
            # Calculate T_dot from consecutive poses (fallback)
            if len(poses6) > 1 and len(timestamps) > 1:
                dt = timestamps[1] - timestamps[0]
                current_6d = poses6[0]
                next_6d = poses6[1]
                
                T0 = self._se3_from_6d(current_6d)
                T1 = self._se3_from_6d(next_6d)
                
                # Calculate displacement  
                delta = T1 @ torch.inverse(T0)
                T_dot = se3_matrix_to_twist(delta.unsqueeze(0)).squeeze(0)  # Keep as displacement
            else:
                T_dot = torch.zeros(6, dtype=torch.float32)
        
        # current/target
        start6 = td['start_pose']
        goal6 = td['goal_pose']
        current_T = self._se3_from_6d(poses6[0])
        target_T = self._se3_from_6d(goal6)
        
        # time scalar (normalize to [0,1] based on trajectory progress)
        if len(timestamps) > 1:
            total_time = timestamps[-1] - timestamps[0]
            # For first step, time = 0
            tnorm = torch.tensor([0.0], dtype=torch.float32)
        else:
            tnorm = torch.tensor([0.0], dtype=torch.float32)
        
        # Apply normalization if enabled
        if self.normalize_twist and self.normalizer:
            T_dot = self.normalizer.normalize_twist(T_dot.unsqueeze(0)).squeeze(0)
        
        # pc and geometry radii
        pc = self._load_pointcloud(env_id)
        g = self.radii.clone()
        return {
            'pc': pc,                        # [N,3]
            'current_T': current_T,          # [4,4]
            'target_T': target_T,            # [4,4]
            'time_t': tnorm,                 # [1]
            'g': g,                          # [3]
            'T_dot': T_dot                   # [6] twist vector (normalized if enabled)
        }


