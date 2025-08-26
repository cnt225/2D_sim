import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import torch
import numpy as np
import plotly
from plotly import graph_objects as go
from plotly import express as px
import trimesh

class PlotlyPlot():
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        fig = go.Figure()
        fig.update_layout(width=800, height=600, margin=dict(t=10, l=10, r=10, b=10))
        scale = 0.15
        X = [scale, scale, scale, scale, -scale, -scale, -scale, -scale]
        Y = [scale, scale, -scale, -scale, scale, scale, -scale, -scale]
        Z = [scale, -scale, scale, -scale, scale, -scale, scale, -scale]
        # fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(size=1,color='black'), showlegend=True))
        self.cube_scale = scale
        self.fig = fig    

    def plotly_vector(self, x, y, z, u, v, w, color = 'black', sizemode = 'absolute', sizeref = 0.2):
        self.fig.add_trace(go.Scatter3d(x=[x, x+u], y=[y, y+v], z=[z, z+w], mode='lines', line=dict(color=color, width=5), showlegend=False))
        self.fig.add_trace(go.Cone(x=[x+u], y=[y+v], z=[z+w], u=[u], v=[v], w=[w], sizemode=sizemode, sizeref=sizeref, anchor='tip', colorscale=[[0, color], [1, color]], showscale=False))

    def plotly_frame(self, SE3):    
        self.plotly_vector(SE3[0,3], SE3[1,3], SE3[2,3], SE3[0,0], SE3[1,0], SE3[2,0], color = 'red')
        self.plotly_vector(SE3[0,3], SE3[1,3], SE3[2,3], SE3[0,1], SE3[1,1], SE3[2,1], color = 'green')
        self.plotly_vector(SE3[0,3], SE3[1,3], SE3[2,3], SE3[0,2], SE3[1,2], SE3[2,2], color = 'blue')

    def plotly_SE3traj(self, traj):
        for SE3s in traj[:]:
            for SE3 in SE3s:
                self.plotly_frame(SE3)

    def plotly_gripper(self, SE3, color='black', i='', showlegend=False):
        unit1 = 0.066 #* 8 # 0.56
        unit2 = 0.041 #* 8 # 0.32
        unit3 = 0.046 #* 8 # 0.4
        pbase = torch.Tensor([0, 0, 0, 1]).reshape(1, -1)
        pcenter = torch.Tensor([0, 0, unit1, 1]).reshape(1, -1)
        pleft = torch.Tensor([unit2, 0, unit1, 1]).reshape(1, -1)
        pright = torch.Tensor([-unit2, 0, unit1, 1]).reshape(1, -1)
        plefttip = torch.Tensor([unit2, 0, unit1+unit3, 1]).reshape(1, -1)
        prighttip = torch.Tensor([-unit2, 0, unit1+unit3, 1]).reshape(1, -1)
        hand = torch.cat([pbase, pcenter, pleft, pright, plefttip, prighttip], dim=0).to(SE3)
        hand = torch.einsum('ij, kj -> ik', SE3, hand).cpu()

        phandx = [hand[0,4], hand[0,2], hand[0,1], hand[0,0], hand[0,1], hand[0,3], hand[0,5]]
        phandy = [hand[1,4], hand[1,2], hand[1,1], hand[1,0], hand[1,1], hand[1,3], hand[1,5]]
        phandz = [hand[2,4], hand[2,2], hand[2,1], hand[2,0], hand[2,1], hand[2,3], hand[2,5]]
        self.fig.add_trace(go.Scatter3d(x=phandx, y=phandy, z=phandz, mode='lines', line=dict(color=color, width=10), showlegend=showlegend, name='gripper'+str(i)))

    def plotly_pc(self, pc, color='burlywood', i='', showlegend=False):
        pc = pc.detach().cpu()
        self.fig.add_trace(go.Scatter3d(x=pc[0,:,0], y=pc[0,:,1], z=pc[0,:,2], mode='markers', marker=dict(size=5, color=color), showlegend=showlegend, name='pc'+str(i)))

    def plotly_mesh(self, mesh, color='aquamarine', i='', showlegend=False):
        xyz = np.array(mesh.vertices)
        ijk = np.array(mesh.triangles)
        self.fig.add_trace(go.Mesh3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], i=ijk[:,0], j=ijk[:,1], k=ijk[:,2], color=color, showlegend=showlegend, name='mesh'+str(i)))

    def plotly_meshgripper(self, SE3, dist = [0.04, 0.04], color='violet', i='', opacity=0.5):
        hand_mesh = trimesh.load('assets/meshes/visual/hand.stl')
        left_finger_mesh = trimesh.load('assets/meshes/visual/finger.stl')
        right_finger_mesh = trimesh.load('assets/meshes/visual/finger.stl')
        SE3 = SE3.cpu()
        XY_ROT = torch.eye(4)
        XY_ROT[:3,:3] = torch.Tensor([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

        hand_xyz = torch.Tensor(hand_mesh.vertices)
        hand_xyz = torch.cat([hand_xyz, torch.ones(hand_xyz.shape[0],1)], axis=1)
        hand_xyz = torch.einsum('hi, ij, kj->kh', SE3, XY_ROT, hand_xyz)[:, :3]
        hand_ijk = torch.Tensor(hand_mesh.faces)

        left_finger_xyz = torch.Tensor(left_finger_mesh.vertices)
        left_finger_xyz = torch.cat([left_finger_xyz, torch.ones(left_finger_xyz.shape[0],1)], axis=1)
        left_finger_ijk = torch.Tensor(left_finger_mesh.faces)
        left_T = torch.eye(4)
        left_T[:3, 3] = torch.Tensor([0, dist[0], 0.0584])
        left_finger_xyz = torch.einsum('gh, hi, ij, kj->kg', SE3, XY_ROT, left_T, left_finger_xyz)[:, :3]

        right_finger_xyz = torch.Tensor(right_finger_mesh.vertices)
        right_finger_xyz = torch.cat([right_finger_xyz, torch.ones(right_finger_xyz.shape[0],1)], axis=1)
        right_finger_ijk = torch.Tensor(right_finger_mesh.faces)
        right_T = torch.eye(4)
        right_T[0, 0] = right_T[1, 1] = -1
        right_T[:3, 3] = torch.Tensor([0, -dist[1], 0.0584])
        right_finger_xyz = torch.einsum('gh, hi, ij, kj->kg', SE3, XY_ROT, right_T, right_finger_xyz)[:, :3]

        self.fig.add_trace(go.Mesh3d(x=hand_xyz[:,0], y=hand_xyz[:,1], z=hand_xyz[:,2], i=hand_ijk[:,0], j=hand_ijk[:,1], k=hand_ijk[:,2], color=color, showlegend=False, name='meshgripper', opacity=opacity))
        self.fig.add_trace(go.Mesh3d(x=left_finger_xyz[:,0], y=left_finger_xyz[:,1], z=left_finger_xyz[:,2], i=left_finger_ijk[:,0], j=left_finger_ijk[:,1], k=left_finger_ijk[:,2], color=color, showlegend=False, name='meshgripper', opacity=opacity))
        self.fig.add_trace(go.Mesh3d(x=right_finger_xyz[:,0], y=right_finger_xyz[:,1], z=right_finger_xyz[:,2], i=right_finger_ijk[:,0], j=right_finger_ijk[:,1], k=right_finger_ijk[:,2], color=color, showlegend=False, name='meshgripper', opacity=opacity))

    def plotly_meshgripper_open(self, SE3, color='purple', i=''):
        self.plotly_meshgripper(SE3, dist = [0.04, 0.04], color=color, i=i)

    def plotly_meshgripper_close(self, SE3, color='purple', i=''):
        self.plotly_meshgripper(SE3, dist = [0, 0], color=color, i=i)

    def plotly_Ngrippers(self, pc, SE3s, color='black'): # pc : (batch, point_num, point_dim)
        for i, SE3 in enumerate(SE3s):
            self.plotly_gripper(SE3, color, i)

    def plotly_Ngrippers_process(self, pc, SE3s): # pc : (batch, point_num, point_dim)
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
                self.plotly_gripper(SE3_traj[i], lc, n)

    def plotly_delete(self, object_name):
        candidate = [self.fig.data[i].name!=None and self.fig.data[i].name.startswith(object_name) for i in range(len(self.fig.data))]
        self.fig.data = [self.fig.data[i] for i in range(len(self.fig.data)) if candidate[i]==False]

    def show(self):
        self.fig.show()

    def save_html(self, name='NULL.html'):
        self.fig.write_html(name)

    def save_json(self, name='NULL.json'):
        self.fig.write_json(name)