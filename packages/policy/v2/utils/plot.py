import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import torch

def plot_se3(se3_traj, ax = None, alpha=1):
    ax.quiver(se3_traj[0,3], se3_traj[1,3], se3_traj[2,3], se3_traj[0,0], se3_traj[1,0], se3_traj[2,0], color = 'tab:red', alpha = alpha)
    ax.quiver(se3_traj[0,3], se3_traj[1,3], se3_traj[2,3], se3_traj[0,1], se3_traj[1,1], se3_traj[2,1], color = 'tab:green', alpha = alpha)
    ax.quiver(se3_traj[0,3], se3_traj[1,3], se3_traj[2,3], se3_traj[0,2], se3_traj[1,2], se3_traj[2,2], color = 'tab:blue', alpha = alpha)
    ax.plot(se3_traj[0,3],se3_traj[1,3],se3_traj[2,3], color = 'k', alpha = alpha)
    return None

def plot_traj(traj, ax=None, xlim = None, ylim = None, zlim = None, alpha = 1, title = None):
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = None

    for SE3s in traj[:]:
        for SE3 in SE3s:
            plot_se3(SE3, ax=ax)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if xlim is not None:
        ax.set_xlim3d(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim3d(ylim[0],ylim[1])
    if zlim is not None:
        ax.set_zlim3d(zlim[0],zlim[1])
    if title is not None:
        ax.set_title(title)
    return fig

def plot_gripper(SE3, ax = None, alpha = 1, color = 'k'):
    if SE3.shape == (4, 4):
        SE3 = SE3.reshape(1, 4, 4)
    unit1 = 0.066 * 8 # 0.56
    unit2 = 0.041 * 8 # 0.32
    unit3 = 0.046 * 8 # 0.4
    pbase = torch.Tensor([0, 0, 0, 1]).reshape(1, -1)
    pcenter = torch.Tensor([0, 0, unit1, 1]).reshape(1, -1)
    pleft = torch.Tensor([unit2, 0, unit1, 1]).reshape(1, -1)
    pright = torch.Tensor([-unit2, 0, unit1, 1]).reshape(1, -1)
    plefttip = torch.Tensor([unit2, 0, unit1+unit3, 1]).reshape(1, -1)
    prighttip = torch.Tensor([-unit2, 0, unit1+unit3, 1]).reshape(1, -1)
    hand = torch.cat([pbase, pcenter, pleft, pright, plefttip, prighttip], dim=0).unsqueeze(0).repeat(SE3.shape[0], 1, 1).to(SE3)
    hand = torch.einsum('nij, nkj -> nik', SE3, hand).cpu().numpy()
    for i in range(SE3.shape[0]):
        phandx = [hand[i,0,4], hand[i,0,2], hand[i,0,1], hand[i,0,0], hand[i,0,1], hand[i,0,3], hand[i,0,5]]
        phandy = [hand[i,1,4], hand[i,1,2], hand[i,1,1], hand[i,1,0], hand[i,1,1], hand[i,1,3], hand[i,1,5]]
        phandz = [hand[i,2,4], hand[i,2,2], hand[i,2,1], hand[i,2,0], hand[i,2,1], hand[i,2,3], hand[i,2,5]]
        ax.plot(phandx, phandy, phandz, color = color)
    return None

def plot_grasping(pc, SE3s, ax=None, xlim = [-2, 2], ylim = [-2, 2], zlim = [-2, 2], alpha = 1, title = None, color = 'g'): # pc : (batch, point_num, point_dim)
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = None
    pc = pc.detach().cpu().numpy()
    ax.scatter(pc[0,:,0], pc[0,:,1], pc[0,:,2], alpha = 1)
    for SE3 in SE3s:
        plot_gripper(SE3, ax=ax, alpha = alpha, color=color)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if xlim is not None:
        ax.set_xlim3d(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim3d(ylim[0],ylim[1])
    if zlim is not None:
        ax.set_zlim3d(zlim[0],zlim[1])
    if title is not None:
        ax.set_title(title)
    return fig

def plot_grasping_process(pc, SE3s, ax=None, xlim = [-2, 2], ylim = [-2, 2], zlim = [-2, 2], alpha = 1, title = None): # pc : (batch, point_num, point_dim)
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = None
    pc = pc.detach().cpu().numpy()
    ax.scatter(pc[0,:,0], pc[0,:,1], pc[0,:,2], alpha = 1)
    SE3_nt = torch.permute(SE3s, (1,0,2,3))
    for n in range(len(SE3_nt)):
        SE3_traj = SE3_nt[n]
        for i in range(len(SE3_traj)):
            if i==0:
                lc = 'r'
            elif i==len(SE3_traj)-1:
                lc = 'b'
            else:
                lc = 'k'
            plot_gripper(SE3_traj[i], ax=ax, alpha = alpha, color=lc)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if xlim is not None:
        ax.set_xlim3d(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim3d(ylim[0],ylim[1])
    if zlim is not None:
        ax.set_zlim3d(zlim[0],zlim[1])
    if title is not None:
        ax.set_title(title)
    return fig

def plot_vector(pc, ut, SE3s, ax=None, xlim = [-2, 2], ylim = [-2, 2], zlim = [-2, 2], alpha = 1, title = None): # pc : (batch, point_num, point_dim)
    if ax is None:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = None
    pc = pc.detach().cpu().numpy()
    ax.scatter(pc[0,:,0], pc[0,:,1], pc[0,:,2], alpha = 1)
    U_nt = torch.permute(ut, (1,0,2))
    SE3_nt = torch.permute(SE3s, (1,0,2,3))
    for n in range(len(SE3_nt)):
        SE3_traj = SE3_nt[n]
        U_traj = U_nt[n]
        for i in range(len(SE3_traj)):
            X, Y, Z = SE3_traj[i, 0, 3], SE3_traj[i, 1, 3], SE3_traj[i, 2, 3]
            ax.quiver(X, Y, Z, U_traj[i,0], U_traj[i,1], U_traj[i,2], color = 'tab:purple', alpha = alpha)
            ax.quiver(X, Y, Z, U_traj[i,3], U_traj[i,4], U_traj[i,5], color = 'tab:pink', alpha = alpha)
            if i==0:
                lc = 'r'
            elif i==len(SE3_traj)-1:
                lc = 'b'
            else:
                lc = 'k'
            # plot_se3(SE3_traj[i], ax=ax)
            plot_gripper(SE3_traj[i], ax=ax, alpha = alpha, color=lc)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if xlim is not None:
        ax.set_xlim3d(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim3d(ylim[0],ylim[1])
    if zlim is not None:
        ax.set_zlim3d(zlim[0],zlim[1])
    if title is not None:
        ax.set_title(title)
    return fig

def show_animation_3d(figure):
    ax = figure.get_axes()[0]
    def animate(i):
        ax.view_init(elev=10., azim=i*10)
        return figure,
    ani = animation.FuncAnimation(figure, animate, frames=50, interval=100, blit=True)
    return HTML(ani.to_jshtml())