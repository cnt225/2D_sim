from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch
import wandb

from utils.Lie import *
from utils.ode_solver import get_ode_solver
from utils.utils import init_parameter
from utils.motion_utils import se3_matrix_to_twist, interpolate_se3_poses

class MotionRCFM(torch.nn.Module): 
    """
    Motion Planning Rectified Conditional Flow Model
    Based on GraspRCFM but adapted for trajectory generation
    """
    def __init__(self, velocity_field, latent_feature, prob_path='OT', init_dist={'arch':'uniform'}, ode_solver={'arch':'RK4','n_steps':20}, **kwargs):
        super().__init__()

        self.velocity_field = velocity_field
        self.latent_feature = latent_feature
        self.prob_path = prob_path
        self.init_dist = init_dist
        self.ode_steps = ode_solver['n_steps']
        self.ode_solver = get_ode_solver(ode_solver['arch'])

        if self.prob_path == 'OT' or self.prob_path == 'OT_CFM':
            pass
        else:
            print(f"Prob Path: {self.prob_path} has not been implemented!")
            raise NotImplementedError

        self.count = 0

    def count_nfe(self, init=False, display=False):
        if init:
            self.count = 0
        elif display:
            return self.count
        else:
            self.count += 1

    def forward(self, x_t, target_T, t, v, g=None):
        """
        Forward pass through velocity field (fm-main compatible)
        
        Args:
            x_t: [B, 4, 4] current SE(3) pose
            t: [B, 1] time parameter
            v: [B, lat_dim] point cloud features
        """
        self.count_nfe()
        return self.velocity_field(x_t, target_T, t, v, g)

    def init_distribution(self, num_samples, pc=None, target_poses=None):
        """Generate initial distribution for motion planning"""
        if self.init_dist['arch'] == 'uniform':
            # Random initial poses
            R0 = Rotation.random(num_samples).as_matrix()
            p0 = torch.randn(num_samples, 3) * 0.5  # Smaller initial spread
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            T0[:, :3, :3] = torch.Tensor(R0)
            T0[:, :3, 3] = p0
            return T0

        elif self.init_dist['arch'] == 'gaussian':
            # Gaussian distribution around identity
            w_std = 0.3  # Reduced angular spread
            p_std = 0.5  # Reduced position spread
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            w_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * w_std)
            p_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
            Gaussian_w = w_distribution.sample((num_samples,))
            Gaussian_p = p_distribution.sample((num_samples,))
            T0[:, :3, :3] = T0[:, :3, :3] @ exp_so3(bracket_so3(Gaussian_w))
            T0[:, :3, 3] = T0[:, :3, 3] + Gaussian_p
            return T0

        elif self.init_dist['arch'] == 'near_start':
            # Initialize near the first waypoint if available
            if target_poses is not None and len(target_poses) > 0:
                # Use target_poses[0] as reference start
                ref_pose = target_poses[0] if len(target_poses.shape) > 2 else target_poses
                T0 = ref_pose.unsqueeze(0).repeat(num_samples, 1, 1)
                
                # Add small random perturbation
                w_noise = torch.randn(num_samples, 3) * 0.1
                p_noise = torch.randn(num_samples, 3) * 0.1
                
                T0[:, :3, :3] = T0[:, :3, :3] @ exp_so3(bracket_so3(w_noise))
                T0[:, :3, 3] = T0[:, :3, 3] + p_noise
                return T0
            else:
                # Fallback to gaussian
                return self.init_distribution(num_samples, pc, None)

        else:
            # Default identity
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            return T0

    def x_1(self, t, x0, x1):
        """Target endpoint (twist vectors converted to SE(3))"""
        return x1

    def x_t(self, t, x0, x1):
        """SE(3) interpolation from x0 to x1 at time t (fm-main compatible)"""
        R0 = x0[:, :3, :3]
        P0 = x0[:, :3, 3]
        R1 = x1[:, :3, :3]
        P1 = x1[:, :3, 3]
        
        xt = torch.eye(4).unsqueeze(0).repeat(len(x0), 1, 1).to(x1)
        
        # Rotation interpolation: R0 @ exp_SO3(t * log_SO3(R0^-1 @ R1))
        xt[:, :3, :3] = torch.einsum('nij,njk->nik', R0, 
                                   exp_so3(t.reshape(-1, 1, 1) * 
                                          log_SO3(torch.einsum('nij,njk->nik', inv_SO3(R0), R1))))
        
        # Translation interpolation: P0 + t * (P1 - P0)
        xt[:, :3, 3] = P0 + t.reshape(-1, 1) * (P1 - P0)
        
        return xt

    def u_t(self, t, x0, x1):
        """Velocity field target (6D twist vectors)"""
        R0 = x0[:, :3, :3]
        P0 = x0[:, :3, 3]
        R1 = x1[:, :3, :3]
        P1 = x1[:, :3, 3]
        ut = torch.zeros((len(x0), 6)).to(x1)
        
        # Angular velocity (body frame)
        ut[:, 0:3] = bracket_so3(log_SO3(torch.einsum('nij,njk->nik', inv_SO3(R0), R1)))
        ut[:, 0:3] = torch.einsum('nij, nj -> ni', R0, ut[:, 0:3])  # w_b to w_s
        
        # Linear velocity (world frame)
        ut[:, 3:6] = P1 - P0
        
        return ut

    def get_latent_vector(self, y):
        """Extract point cloud features"""
        # y는 이미 [B, 3, N] 형태여야 함
        if y.shape[1] != 3:  # [B, N, 3] 형태라면 transpose 필요
            y = y.transpose(2, 1)
        return self.latent_feature.forward(y)

    @torch.no_grad()
    def sample(self, num_samples, pc, target_poses=None):
        """Generate motion trajectory samples"""
        # Initial poses
        T0 = self.init_distribution(num_samples, pc, target_poses).to(pc)
        
        # Dummy target poses if not provided
        if target_poses is None:
            target_poses = T0.clone()
            target_poses[:, :3, 3] += torch.randn_like(target_poses[:, :3, 3]) * 0.5
        
        # Generate trajectory
        NFE = self.ode_steps
        self.count_nfe(init=True)
        
        # Extract point cloud features
        v = self.get_latent_vector(pc)
        
        # Custom ODE function for motion planning
        def motion_velocity_func(x_t, t, v_features):
            # v_features는 ODE solver에서 전달되는 점클라우드 특징
            # target_poses 반복 (ODE 중 target은 고정)
            if target_poses.shape[0] == 1 and x_t.shape[0] > 1:
                target_batch = target_poses.repeat(x_t.shape[0], 1, 1)
            else:
                target_batch = target_poses
            
            # 더미 robot geometry 생성 (추론 시 필요)
            g_dummy = torch.zeros(x_t.shape[0], 3).to(x_t.device)
            
            return self.velocity_field(x_t, target_batch, t.view(-1, 1), v_features, g_dummy)
        
        trajs = self.ode_solver(
            func=motion_velocity_func,
            x0=T0,
            t=torch.linspace(0, 1, NFE).to(pc),
            v=v,
            device=pc.device
        )
        
        generated_samples = deepcopy(trajs[-1].detach())
        counts = self.count_nfe(display=True)
        return generated_samples, counts

    def train_step(self, data, losses, optimizer):
        """Training step for motion planning (per-step supervised): predict T_dot"""
        pc = data['pc']                 # [B, N, 3]
        current_T = data['current_T']   # [B, 4, 4]
        target_T = data['target_T']     # [B, 4, 4]
        time_t = data['time_t']         # [B, 1]
        g = data.get('g', None)         # [B, 3]
        # label: if provided as delta_T, convert to 6D twist per dt=1 (assume normalized)
        if 'T_dot' in data:
            T_dot_gt = data['T_dot']    # [B, 6]
        else:
            delta_T = data['delta_T']   # [B, 4, 4]
            # log map to twist assuming dt normalized to 1
            T_dot_gt = se3_matrix_to_twist(delta_T)

        optimizer.zero_grad()

        v = self.get_latent_vector(pc)
        T_dot_pred = self.forward(current_T, target_T, time_t, v, g)

        loss = ((T_dot_pred - T_dot_gt) ** 2).mean()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()

        return {
            'loss': loss.item(),
            'train/loss_': loss.item(),
        }

    def val_step(self, data, losses):
        """Validation step (per-step supervised)"""
        pc = data['pc']
        current_T = data['current_T']
        target_T = data['target_T']
        time_t = data['time_t']
        g = data.get('g', None)
        if 'T_dot' in data:
            T_dot_gt = data['T_dot']
        else:
            delta_T = data['delta_T']
            T_dot_gt = se3_matrix_to_twist(delta_T)

        with torch.no_grad():
            v = self.get_latent_vector(pc)
            T_dot_pred = self.forward(current_T, target_T, time_t, v, g)
            loss = ((T_dot_pred - T_dot_gt) ** 2).mean()

        return {
            'loss': loss.item(),
            'valid/loss_': loss.item(),
        }

    def eval_step(self, ys, num_trajectories, *args, **kwargs):
        """Evaluation step - generate trajectories"""
        trajectory_list = []

        for y in ys:
            y = y.unsqueeze(0)

            # Generate trajectory
            trajectories, _ = self.sample(num_trajectories, y)
            trajectory_list.append(trajectories)

        return trajectory_list

    def vis_step(self, ys, num_trajectories=5, *args, **kwargs):
        """Visualization step"""
        trajectory_list = []
        img_dict = {}

        for idx, y in enumerate(ys):
            y = y.unsqueeze(0)

            # Generate trajectories
            trajectories, _ = self.sample(num_trajectories, y)
            trajectory_list.append(trajectories.cpu().numpy())

            # Simple trajectory visualization could be added here
            # For now, just return the trajectories
            img_dict[f'trajectory_{idx+1}$'] = f"Generated {num_trajectories} trajectories"

        return trajectory_list, img_dict