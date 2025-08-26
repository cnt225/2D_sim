from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch
import wandb

from utils.plot import plot_grasping, plot_grasping_process
from utils.Lie import *
from utils.ode_solver import get_ode_solver
from utils.utils import init_parameter
from utils.utils_plotly import PlotlyPlot

class GraspRCFM(torch.nn.Module): 
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
        self.num_samples_at_a_time = None

    def count_nfe(self, init=False, display=False):
        if init:
            self.count = 0
        elif display:
            return self.count
        else:
            self.count += 1

    def forward(self, x, t, v):
        self.count_nfe()
        return self.velocity_field(x, t, v)

    def init_distribution(self, num_samples, pc=None, num_repeats=None):
        if self.init_dist['arch'] == 'uniform':
            R0 = Rotation.random(num_samples).as_matrix()
            p0 = torch.randn(num_samples, 3)
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            T0[:, :3, :3] = torch.Tensor(R0)
            T0[:, :3, 3] = p0
            return T0

        elif self.init_dist['arch'] == 'gaussian':
            w_std = 0.1
            p_std = 0.1
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            w_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * w_std)
            p_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
            Gaussian_w = w_distribution.sample((num_samples,))
            Gaussian_p = p_distribution.sample((num_samples,))
            T0[:, :3, :3] = T0[:, :3, :3] @ exp_so3(bracket_so3(Gaussian_w))
            T0[:, :3, 3] = T0[:, :3, 3] + Gaussian_p
            return T0

        elif self.init_dist['arch'] == 'canonical_fixed':
            assert num_samples >= 2, "pick more samples"
            num1 = torch.randint(1, num_samples-1, (1,))
            num2 = num_samples - num1
            theta1 = torch.rand(num1) * torch.pi * 2
            c1 = torch.cos(theta1).unsqueeze(1)
            s1 = torch.sin(theta1).unsqueeze(1)
            z = 1.106 * torch.ones(len(c1)).unsqueeze(1)
            p1 = torch.cat([0.23 * c1, 0.23 * s1, z], axis=1)
            T1 = torch.eye(4).unsqueeze(0).repeat(num1, 1, 1)
            T1[:, 0, 0] = c1.squeeze(1)
            T1[:, 1, 1] = -c1.squeeze(1)
            T1[:, 1, 0] = T1[:, 0, 1] = s1.squeeze(1)
            T1[:, 2, 2] = -1
            T1[:, :3, 3] = p1

            theta2 = torch.rand(num2) * torch.pi
            c2 = torch.cos(theta2).unsqueeze(1)
            s2 = torch.sin(theta2).unsqueeze(1)
            p2 = torch.cat([torch.zeros((len(c2),1)), -0.2 -1.006 * s2, 0.025 + 1.006 * c2], axis=1)
            T2 = torch.eye(4).unsqueeze(0).repeat(num2, 1, 1)
            T2[:, 1, 2] = s2.squeeze(1)
            T2[:, 2, 1] = -s2.squeeze(1)
            T2[:, 1, 1] = T2[:, 2, 2] = -c2.squeeze(1)
            T2[:, :3, 3] = p2

            T0 = torch.cat([T1, T2], axis=0)
            T0 = T0[torch.randperm(len(T0))]
            return T0
        
        elif self.init_dist['arch'] == 'canonical':
            assert num_samples >= 2, "pick more samples"
            num1 = torch.randint(1, num_samples-1, (1,))
            num2 = num_samples - num1
            theta1 = torch.rand(num1) * torch.pi * 2
            c1 = torch.cos(theta1).unsqueeze(1)
            s1 = torch.sin(theta1).unsqueeze(1)
            z = 1.106 * torch.ones(len(c1)).unsqueeze(1)
            p1 = torch.cat([0.23 * c1, 0.23 * s1, z], axis=1)
            T1 = torch.eye(4).unsqueeze(0).repeat(num1, 1, 1)
            T1[:, 0, 0] = c1.squeeze(1)
            T1[:, 1, 1] = -c1.squeeze(1)
            T1[:, 1, 0] = T1[:, 0, 1] = s1.squeeze(1)
            T1[:, 2, 2] = -1
            T1[:, :3, 3] = p1

            theta2 = torch.rand(num2) * torch.pi
            c2 = torch.cos(theta2).unsqueeze(1)
            s2 = torch.sin(theta2).unsqueeze(1)
            p2 = torch.cat([torch.zeros((len(c2),1)), -0.2 -1.006 * s2, 0.025 + 1.006 * c2], axis=1)
            T2 = torch.eye(4).unsqueeze(0).repeat(num2, 1, 1)
            T2[:, 1, 2] = s2.squeeze(1)
            T2[:, 2, 1] = -s2.squeeze(1)
            T2[:, 1, 1] = T2[:, 2, 2] = -c2.squeeze(1)
            T2[:, :3, 3] = p2

            T0 = torch.cat([T1, T2], axis=0)
            T0 = T0[torch.randperm(len(T0))]

            if pc.shape == (1000, 3) or len(pc) == 1:
                pc = pc.squeeze()
                TT = init_parameter(pc)
                return TT.unsqueeze(0).repeat(num_samples, 1, 1) @ T0
            else:
                assert len(pc) == len(num_repeats)
                TT = []
                for (pcd, rep) in zip(pc, num_repeats):
                    TT1 = init_parameter(pcd)
                    TT1 = TT1.unsqueeze(0).repeat(rep, 1, 1)
                    TT.append(TT1)
                return torch.cat(TT, dim=0) @ T0
        
        elif self.init_dist['arch'] == 'spherical':
            radius = self.init_dist['radius']
            R0 = Rotation.random(num_samples).as_matrix()
            p0 = torch.Tensor([0, 0, -radius]).unsqueeze(0).repeat(num_samples, 1)
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            T0[:, :3, :3] = torch.Tensor(R0)
            T0[:, :3, 3] = (torch.Tensor(R0) @ p0.unsqueeze(2)).squeeze(2)
            return T0

        else:
            T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
            return T0

    def x_1(self, t, x0, x1):
        return x1
        # sigma_min = 1e-5
        # T1 = deepcopy(x1)
        # Gaussian_w = bracket_so3(log_SO3(x0[:, :3, :3]))
        # Gaussian_p = x0[:, :3, 3]
        # T1[:, :3, :3] = T1[:, :3, :3] @ exp_so3(bracket_so3(sigma_min * Gaussian_w))
        # T1[:, :3, 3] = T1[:, :3, 3] + sigma_min * Gaussian_p
        # return T1

    def x_t(self, t, x0, x1): # x_t -> geodesic from x0 to x1 at time = t
        R0 = x0[:, :3, :3]
        P0 = x0[:, :3, 3]
        R1 = x1[:, :3, :3]
        P1 = x1[:, :3, 3]
        xt = torch.eye(4).unsqueeze(0).repeat(len(x0), 1, 1).to(x1)
        xt[:, :3, :3] = torch.einsum('nij,njk->nik', R0, exp_so3(t.reshape(-1, 1, 1) * log_SO3(torch.einsum('nij,njk->nik', inv_SO3(R0), R1))))
        xt[:, :3, 3] = P0 + t.reshape(-1,1) * (P1 - P0)
        return xt

    def u_t(self, t, x0, x1): # d(x_t)/dt -> vector at xt = (w, v)
        R0 = x0[:, :3, :3]
        P0 = x0[:, :3, 3]
        R1 = x1[:, :3, :3]
        P1 = x1[:, :3, 3]
        ut = torch.zeros((len(x0), 6)).to(x1)
        ut[:, 0:3] = bracket_so3(log_SO3(torch.einsum('nij,njk->nik', inv_SO3(R0), R1)))
        ut[:, 0:3] = torch.einsum('nij, nj -> ni', R0, ut[:, 0:3]) # w_b to w_s
        ut[:, 3:6] = P1 - P0
        return ut

    def get_latent_vector(self, y): 
        return self.latent_feature.forward(y.transpose(2, 1))

    # def dist(self, u1, u2):
    #     return se3v_dist(u1, u2).mean()

    @torch.no_grad()
    def sample(self, num_samples, pc): 
        # Initial T0
        T0 = self.init_distribution(num_samples, pc, None).to(pc)
        # Generate samples
        NFE = self.ode_steps
        self.count_nfe(init=True)
        trajs = self.ode_solver(
            func = self.velocity_field,
            x0 = T0,
            t = torch.linspace(0, 1, NFE).to(pc),
            v = self.get_latent_vector(pc),
            device = pc.device
        ) # t * n * x
        generated_samples = deepcopy(trajs[-1].detach())
        counts = self.count_nfe(display=True)
        return generated_samples, counts

    def _process(self, pc, T0=None): # multi T0
        if T0 == None:
            T0 = self.init_distribution(10, pc, None)
        else:
            T0 = T0.reshape(-1, 4, 4)
        NFE = self.ode_steps
        trajs = self.ode_solver(
            func = self.velocity_field,
            x0 = T0,
            t = torch.linspace(0, 1, NFE).to(pc),
            v = self.get_latent_vector(pc),
            device = pc.device
        )
        return trajs.detach()

    def train_step(self, data, losses, optimizer): # 임의의 x1, 임의의 t에 대해 x(x_t), u(VF), v(model), 이를 바탕으로 backprop
        x1 = data['Ts_grasp']
        y = data['pc']

        num_repeats = (x1[:, :, 3, 3] == 1).sum(1)
        x1 = x1[x1[:, :, 3, 3] == 1]
        batch_size = len(x1)

        optimizer.zero_grad()

        t = torch.rand(batch_size, 1).to(x1)
        x0 = self.init_distribution(batch_size, y, num_repeats).to(x1)

        x_t = self.x_t(t, x0, self.x_1(t, x0, x1))
        u_t = self.u_t(t, x0, self.x_1(t, x0, x1))

        v = self.get_latent_vector(y)
        v = v.repeat_interleave(num_repeats, dim=0)

        v_t = self(x_t, t, v)

        loss = ((u_t - v_t) ** 2).mean()
        # loss = (0.2 * (u_t[:,:3] - v_t[:,:3])**2 + 1.8 * (u_t[:,3:] - v_t[:,3:])**2).mean()
        # loss = self.dist(u_t, v_t)
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'train/loss_': loss.item(),
        }

    def val_step(self, data, losses):
        x1 = data['Ts_grasp']
        y = data['pc']

        num_repeats = (x1[:, :, 3, 3] == 1).sum(1)
        x1 = x1[x1[:, :, 3, 3] == 1]
        batch_size = len(x1)

        t = torch.rand(batch_size, 1).to(x1)
        x0 = self.init_distribution(batch_size, y, num_repeats).to(x1)

        x_t = self.x_t(t, x0, self.x_1(t, x0, x1))
        u_t = self.u_t(t, x0, self.x_1(t, x0, x1))

        v = self.get_latent_vector(y)
        v = v.repeat_interleave(num_repeats, dim=0)

        v_t = self(x_t, t, v)

        loss = ((u_t - v_t) ** 2).mean()
        # loss = (0.2 * (u_t[:,:3] - v_t[:,:3])**2 + 1.8 * (u_t[:,3:] - v_t[:,3:])**2).mean()
        # loss = self.dist(u_t, v_t)
        loss.backward()

        return {
            'loss': loss.item(),
            'valid/loss_': loss.item(),
        }

    def eval_step(self, ys, num_grasps, *args, **kwargs): # val_data의 분포와 sample의 분포의 mmd_loss 비교
        Ts_grasp_list = []

        for y in ys:
            y = y.unsqueeze(0)

            # Sampling
            Ts_grasp, _ = self.sample(num_grasps, y) ######### sample(num_samples, point cloud data(batch*point_num*point_dim)) / 확인 필요.

            Ts_grasp_list += [Ts_grasp]

        return Ts_grasp_list

    def vis_step(self, ys, num_grasps=10, *args, **kwargs):
        Ts_grasp_list = []
        img_dict = {}

        for idx, y in enumerate(ys):
            y = y.unsqueeze(0)

            # Sampling
            Ts_grasp, _ = self.sample(num_grasps, y) ## ↓↓↓ 확인 필요 ↓↓↓
            process = self._process(y, self.init_distribution(num_grasps, y, None))

            Ts_grasp_list += [Ts_grasp.cpu().numpy()]

            ###################################### ↓ visualize with plot
            # fig1 = plot_grasping(y, Ts_grasp)
            # fig1.canvas.draw()
            # img1 = np.array(fig1.canvas.renderer._renderer)[:, :, 0:3]
            # fig2 = plot_grasping_process(y, process)
            # fig2.canvas.draw()
            # img2 = np.array(fig2.canvas.renderer._renderer)[:, :, 0:3]
            # img = np.concatenate([img1, img2], axis=1)
            # image = wandb.Image(img, caption = f'case_{idx+1}')
            # img_dict[f'case_{idx+1}$'] = image
            # plt.close(fig1)
            # plt.close(fig2)
            ###################################### ↓ visualize with plotly
            PLOT = PlotlyPlot()
            # PLOT.plotly_mesh(mesh, color='aquamarine')
            scale = 0.125
            PLOT.plotly_pc(y * scale)
            T = deepcopy(Ts_grasp)
            T[:,:3,3]*=scale
            for vidx in range(len(T)):
                PLOT.plotly_gripper(T[vidx])

            PLOT.fig.update_scenes(aspectmode='data')
            img_dict[f'case_{idx+1}$'] = PLOT.fig
            ######################################

        return Ts_grasp_list, img_dict


# class GraspRCFM_Euc(torch.nn.Module): 
#     def __init__(self, velocity_field, latent_feature, prob_path='OT', **kwargs):
#         super().__init__()

#         self.velocity_field = velocity_field
#         self.latent_feature = latent_feature
#         self.prob_path = prob_path

#         if self.prob_path == 'OT':
#             pass
#         else:
#             print(f"Prob Path: {self.prob_path} has not been implemented!")
#             raise NotImplementedError

#         self.count = 0
#         self.num_samples_at_a_time = None

#     def count_nfe(self, init=False, display=False):
#         if init:
#             self.count = 0
#         elif display:
#             return self.count
#         else:
#             self.count += 1

#     def forward(self, x, t, v):
#         self.count_nfe()
#         return self.velocity_field(x, t, v)

#     def init_distribution_(self, num_samples):
#         R0 = Rotation.random(num_samples).as_matrix()
#         p0 = torch.randn(num_samples, 3)
#         T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
#         T0[:, :3, :3] = torch.Tensor(R0)
#         T0[:, :3, 3] = p0
#         return T0

#     def init_distribution(self, num_samples):
#         w_std = 1
#         p_std = 1
#         T0 = torch.eye(4).unsqueeze(0).repeat(num_samples, 1, 1)
#         w_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * w_std)
#         p_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
#         Gaussian_w = w_distribution.sample((num_samples,))
#         Gaussian_p = p_distribution.sample((num_samples,))

#         T0[:, :3, :3] = T0[:, :3, :3] @ exp_so3(bracket_so3(Gaussian_w))
#         T0[:, :3, 3] = T0[:, :3, 3] + Gaussian_p
#         return T0
    
#     def x_1(self, t, x0, x1):
#         return x1
#         sigma_min = 1e-5
#         T1 = deepcopy(x1)
#         Gaussian_w = bracket_so3(log_SO3(x0[:, :3, :3]))
#         Gaussian_p = x0[:, :3, 3]
#         T1[:, :3, :3] = T1[:, :3, :3] @ exp_so3(bracket_so3(sigma_min * Gaussian_w))
#         T1[:, :3, 3] = T1[:, :3, 3] + sigma_min * Gaussian_p
#         return T1

#     def x_t(self, t, x0, x1): # x_t -> geodesic from x0 to x1 at time = t
#         R0 = x0[:, :3, :3]
#         P0 = x0[:, :3, 3]
#         R1 = x1[:, :3, :3]
#         P1 = x1[:, :3, 3]
#         xt = torch.eye(4).unsqueeze(0).repeat(len(x0), 1, 1).to(x1)
#         xt[:, :3, :3] = exp_so3(log_SO3(R0) + t.reshape(-1, 1, 1) * (log_SO3(R1) - log_SO3(R0)))
#         xt[:, :3, 3] = P0 + t.reshape(-1,1) * (P1 - P0)
#         return xt

#     def u_t(self, t, x0, x1): # d(x_t)/dt -> vector at xt = (w, v)
#         R0 = x0[:, :3, :3]
#         P0 = x0[:, :3, 3]
#         R1 = x1[:, :3, :3]
#         P1 = x1[:, :3, 3]
#         ut = torch.zeros((len(x0), 6)).to(x1)
#         ut[:, 0:3] = bracket_so3(log_SO3(R1) - log_SO3(R0))
#         ut[:, 3:6] = P1 - P0
#         return ut

#     def get_latent_vector(self, y): 
#         return self.latent_feature.forward(y.transpose(2, 1))

#     # def dist(self, u1, u2):
#     #     return se3v_dist(u1, u2).mean()

#     @torch.no_grad()
#     def sample(self, num_samples, pc): 
#         # Initial T0
#         T0 = self.init_distribution(num_samples).to(pc)
#         # Generate samples
#         NFE = 20
#         self.count_nfe(init=True)
#         trajs = ode_solver_RK4_exp(
#             func = self.velocity_field,
#             x0 = T0,
#             t = torch.linspace(0, 1, NFE).to(pc),
#             v = self.get_latent_vector(pc),
#             device = pc.device
#         ) # t * n * x
#         generated_samples = deepcopy(trajs[-1].detach())
#         counts = self.count_nfe(display=True)
#         return generated_samples, counts

#     def _process(self, pc, T0=None): # multi T0
#         if T0 == None:
#             T0 = self.init_distribution(10)
#         else:
#             T0 = T0.reshape(-1, 4, 4)
#         NFE = 20
#         trajs = ode_solver_RK4_exp(
#             func = self.velocity_field,
#             x0 = T0,
#             t = torch.linspace(0, 1, NFE).to(pc),
#             v = self.get_latent_vector(pc),
#             device = pc.device
#         )
#         return trajs.detach()

#     def train_step(self, data, losses, optimizer): # 임의의 x1, 임의의 t에 대해 x(x_t), u(VF), v(model), 이를 바탕으로 backprop
#         x1 = data['Ts_grasp']
#         y = data['pc']

#         num_repeats = (x1[:, :, 3, 3] == 1).sum(1)
#         x1 = x1[x1[:, :, 3, 3] == 1]
#         batch_size = len(x1)

#         optimizer.zero_grad()

#         t = torch.rand(batch_size, 1).to(x1)
#         x0 = self.init_distribution(batch_size).to(x1)

#         x_t = self.x_t(t, x0, self.x_1(t, x0, x1))
#         u_t = self.u_t(t, x0, self.x_1(t, x0, x1))

#         v = self.get_latent_vector(y)
#         v = v.repeat_interleave(num_repeats, dim=0)

#         v_t = self(x_t, t, v)

#         loss = ((u_t - v_t) ** 2).mean()
#         # loss = self.dist(u_t, v_t)
#         loss.backward()
#         optimizer.step()

#         return {
#             'loss': loss.item(),
#             'train/loss_': loss.item(),
#         }

#     def val_step(self, data, losses):
#         x1 = data['Ts_grasp']
#         y = data['pc']

#         num_repeats = (x1[:, :, 3, 3] == 1).sum(1)
#         x1 = x1[x1[:, :, 3, 3] == 1]
#         batch_size = len(x1)

#         t = torch.rand(batch_size, 1).to(x1)
#         x0 = self.init_distribution(batch_size).to(x1)

#         x_t = self.x_t(t, x0, self.x_1(t, x0, x1))
#         u_t = self.u_t(t, x0, self.x_1(t, x0, x1))

#         v = self.get_latent_vector(y)
#         v = v.repeat_interleave(num_repeats, dim=0)

#         v_t = self(x_t, t, v)

#         loss = ((u_t - v_t) ** 2).mean()
#         # loss = self.dist(u_t, v_t)
#         loss.backward()

#         return {
#             'loss': loss.item(),
#             'valid/loss_': loss.item(),
#         }

#     def eval_step(self, val_ys, *args, **kwargs): # val_data의 분포와 sample의 분포의 mmd_loss 비교
#         Ts_grasp_list = []

#         for val_x, val_y in zip(val_xs, val_ys):
#             num_samples = len(val_x)

#             # Sampling
#             Ts_grasp, _ = self.sample(num_samples, val_y) ######### sample(num_samples, point cloud data(batch*point_num*point_dim)) / 확인 필요.

#             Ts_grasp_list += [Ts_grasp]

#         return Ts_grasp_list

#     def vis_step(self, ys, num_samples=10, *args, **kwargs):
#         ys = ys[:, None]

#         Ts_grasp_list = []
#         img_dict = {}

#         for idx, y in enumerate(ys):
#             # Sampling
#             Ts_grasp, _ = self.sample(num_samples, y) ## ↓↓↓ 확인 필요 ↓↓↓
#             process = self._process(y, self.init_distribution(num_samples))

#             Ts_grasp_list += [Ts_grasp.cpu().numpy()]

#             ###################################### ↓ visualize with plot
#             fig1 = plot_grasping(y, Ts_grasp)
#             fig1.canvas.draw()
#             img1 = np.array(fig1.canvas.renderer._renderer)[:, :, 0:3]
#             fig2 = plot_grasping_process(y, process)
#             fig2.canvas.draw()
#             img2 = np.array(fig2.canvas.renderer._renderer)[:, :, 0:3]
#             img = np.concatenate([img1, img2], axis=1)
#             image = wandb.Image(img, caption = f'case_{idx+1}')
#             img_dict[f'case_{idx+1}$'] = image
#             plt.close(fig1)
#             plt.close(fig2)
#             ######################################

#         return Ts_grasp_list, img_dict
#     def voidfunc(self):
#         pass