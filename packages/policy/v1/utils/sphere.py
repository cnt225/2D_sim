import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib import animation

epsilon = 1e-6

def normalize(x): # move points onto the sphere
    x_norm = torch.norm(x, dim=1).reshape(-1,1)
    return x / (x_norm + epsilon)

def sphere_log(x0, x1): # log_x0 (x1)
    n1 = normalize(x0)
    n2 = normalize(x1)
    n2tn1 = torch.einsum('ij,ij->i', n1, n2).reshape(-1,1)
    # print(n2tn1)
    alpha = torch.acos(n2tn1).reshape(-1,1)
    # print(alpha)
    return (n2 - n2tn1 * n1) * alpha / (torch.sin(alpha) + epsilon)

def sphere_exp(x0, xt): # exp_x0 (xt)
    n1 = normalize(x0)
    nt = xt
    nt_norm = torch.norm(nt, dim=1).reshape(-1,1)
    exp = n1 * torch.cos(nt_norm).reshape(-1,1) + nt / (nt_norm + epsilon) * torch.sin(nt_norm)
    return normalize(exp)

@torch.no_grad()
def rejection_sampling_sphere(sample_size, start, size): # used on 'sample_from_checkerboard_sphere'
    samples = []
    M = max(np.sqrt(1-start**2),np.sqrt(1-(start+size)**2))
    while len(samples) < sample_size:
        x = start + torch.rand(1) * size 
        u = torch.rand(1) * M 
        if u < torch.sqrt(1 - x**2):
            samples.append(x)
    return torch.cat(samples).reshape(-1,1)

@torch.no_grad()
def RK4_sphere(func, x0, t, **kwargs): # RK4 on sphere
    '''
    func : vector field(input : xt[bs, 3], t[bs, 1] -> output : vt[bs, 3])
    x0 : ode start point (x0, [bs, 3])
    t : nfe count from 0 to 1 (t, [nfe])
    output : xts[nfe, bs, 3]
    '''
    Xs = []
    xt = x0
    Xs.append(xt.unsqueeze(0))
    L = len(t)-1
    bs = len(x0)
    for i in range(L):
        t_start = t[i]
        t_end = t[i+1]
        h = t_end - t_start
        
        t0 = torch.Tensor(t_start).expand(bs).reshape(-1, 1)
        t1 = t0
        k1 = func(xt, t1)

        t2 = t0 + h/2
        k2 = func(xt + h * k1 / 2, t2)
        k3 = func(xt + h * k2 / 2, t2)

        t3 = t0 + h
        k4 = func(xt + h * k3, t3)

        xt = xt + h/6 * (k1 + 2 * k2 + 2 * k3 + k4) # exponential form으로 수정 필요
        xt = normalize(xt)
        Xs.append(xt.unsqueeze(0))

    return torch.cat(Xs, dim=0)

@torch.no_grad()
def partial_data_list(num_samples, partial):
    return torch.randperm(num_samples)[:partial]

@torch.no_grad()
def sample_from_100(num_samples, **kwargs): # distribution only on (1,0,0)
    return torch.Tensor([[1,0,0]] * num_samples)

@torch.no_grad()
def sample_from_Uniform(num_samples, **kwargs):
    Xs = []
    for _ in range(num_samples):
        vec = torch.randn(3)
        vec_norm = torch.linalg.vector_norm(vec)
        X = (vec/vec_norm).reshape(-1,3)
        Xs.append(X)
    X = torch.cat(Xs, dim=0)
    return X

@torch.no_grad()
def sample_from_checkerboard_sphere(num_samples, **kwargs): # distribution on 12(long) * 6(lat) checkerboard
    Xs = []
    PI = math.pi
    num_grid = 6
    phi_range = (-PI, PI) # longitude
    z_range = (-1, 1) # height
    num_oversamples = (int(num_samples/(2 * num_grid * num_grid))+1) * 2 * num_grid * num_grid
    for i in range(2*num_grid):
        for j in range(num_grid):
            if (i+j)%2 ==1:
                unit_phi = (phi_range[1] - phi_range[0])/(2*num_grid)
                phi_min = phi_range[0] + unit_phi * i
                phi = torch.tensor([[phi_min]],dtype=torch.float32) + torch.rand(int(num_oversamples/(num_grid**2)),1,dtype=torch.float32)*unit_phi

                unit_z = (z_range[1] - z_range[0])/num_grid
                z_min = z_range[0] + unit_z * j
                z = rejection_sampling_sphere(int(num_oversamples/(num_grid**2)), z_min, unit_z)

                x = torch.cos(phi) * torch.sqrt(1-z**2)
                y = torch.sin(phi) * torch.sqrt(1-z**2)

                X = torch.cat([x, y, z],dim=1)
                Xs.append(X)
    X = torch.cat(Xs, dim=0)
    return X[partial_data_list(num_oversamples,num_samples)]
    

@torch.no_grad()
def sample_from_circle(num_samples, **kwargs): # circle distribution near (1,0,0)
    Xs = []
    cnt=0
    done = 0
    while not done:
        yz = torch.randn(2)
        yz_norm = torch.linalg.vector_norm(yz)
        if yz_norm > 0.5:
            continue
        X = torch.Tensor([1-yz_norm**2, yz[0], yz[1]])
        X = (X / torch.linalg.vector_norm(X)).reshape(-1,3)
        Xs.append(X)
        cnt = cnt + 1
        if cnt >= num_samples:
            break
    X = torch.cat(Xs, dim=0)
    return X

@torch.no_grad()
def animation_model_result_sphere(func, device, **kwargs):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    UNIFORM = sample_from_Uniform(1000)
    pt_num = 10000
    
    nfe = 100
    func = func.to('cpu')
    X0 = sample_from_circle(pt_num).to('cpu')
    T = torch.linspace(0, 1, nfe).to('cpu')
    trajs = RK4_sphere(func, X0, T) # nfe x num_samples x xdim

    def init():
        Utn = UNIFORM.cpu().numpy()
        ax.scatter(Utn[:,0],Utn[:,1],Utn[:,2],color='g',alpha = 0.15)
        ax.set_xlabel("X")
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2) # not available
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)
        # ax.view_init(90,0)
        return fig,

    def animate(i):
        plt.cla()
        # ax.set_xlabel("X")
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)
        Utn = UNIFORM.cpu().numpy()
        ax.scatter(Utn[:,0],Utn[:,1],Utn[:,2],color='g',alpha = 0.15)
        Xtn = trajs[i].detach().cpu().numpy()
        ax.scatter(Xtn[:,0],Xtn[:,1],Xtn[:,2],color = 'r', s=15)
        # Vtn = func(trajs[i],T[i].repeat(len(trajs[i])).reshape(-1,1)).cpu().numpy()
        # ax.quiver(Xtn[:,0],Xtn[:,1],Xtn[:,2],Vtn[:,0],Vtn[:,1],Vtn[:,2])
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=nfe, interval=nfe, blit=True)    

    ani.save('moving_animation_model_result_sphere.gif')