import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from utils.Lie import *

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    elif s_act == "selu":
        return nn.SELU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        return self.net(x)
    
class FC_image(nn.Module):
    def __init__(
        self,
        in_chan=784,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
        out_chan_num=1,
    ):
        super(FC_image, self).__init__()
        assert in_chan == out_chan + 1 
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.out_chan_num = out_chan_num
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for i_layer, [n_hidden, act] in enumerate(zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        dim = np.sqrt(int(self.out_chan/self.out_chan_num))
        x = x.view(-1, self.in_chan)
        out = self.net(x)
        out = out.reshape(-1, self.out_chan_num, dim, dim)
        return out

class vf_FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec, self).__init__()
        assert in_chan == out_chan + 1
        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))
    
class vf_FC_vec_onSphere(nn.Module): # added for sphere toy
    def __init__(
        self,
        in_chan=4,
        out_chan=3,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec_onSphere, self).__init__()
        assert in_chan == out_chan + 1
        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x, t):
        p0 = self.net(torch.cat([x, t], dim=1))
        # return p0
        return p0 - (torch.einsum('ij,ij->i', p0, x).reshape(-1,1)) * x

class vf_FC_vec_ImgToSphere(nn.Module): # added for sphere toy
    def __init__(
        self,
        in_chan=4,
        lat_chan=3,
        out_chan=3,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec_ImgToSphere, self).__init__()
        assert in_chan == out_chan + 1
        self.in_chan = in_chan
        self.lat_chan = lat_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan + lat_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x, t, v):
        p0 = self.net(torch.cat([x, t, v], dim=1))
        # return p0
        return p0 - (torch.einsum('ij,ij->i', p0, x).reshape(-1,1)) * x

class vf_FC_image(nn.Module):
    def __init__(
        self,
        in_chan=784,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
        out_chan_num=1,
    ):
        super(vf_FC_image, self).__init__()
        assert in_chan == out_chan + 1 
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.out_chan_num = out_chan_num
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for i_layer, [n_hidden, act] in enumerate(zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x, t):
        dim = np.int(np.sqrt(self.out_chan/self.out_chan_num))
        x = x.view(-1, self.out_chan)
        out = self.net(torch.cat([x, t], dim=1))
        out = out.reshape(-1, self.out_chan_num, dim, dim)
        return out


class cfw_FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(cfw_FC_vec, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))
    
class UNET4MAPP(nn.Module):
    def __init__(
        self,
        unet
    ):
        super(UNET4MAPP, self).__init__()
        self.unet = unet
        
    def forward(self, t, x):
        bs = len(t)
        out_temp = self.unet(x, t)
        return torch.cat([
            out_temp[:, 0:1, :, :].mean(dim=(2, 3)),
            out_temp[:, 1:, :, :].view(bs, -1)
        ], dim=1)
        
class UNET24MAPP(nn.Module):
    def __init__(
        self,
        unet2,
        fixed_sigma=False
    ):
        super(UNET24MAPP, self).__init__()
        self.unet2 = unet2
        self.fixed_sigma = fixed_sigma
        
    def forward(self, t, x):
        bs = len(t)
        out_temp = self.unet2(t, x)
        if self.fixed_sigma:
            return out_temp
        else:
            return torch.cat([
                out_temp[:, 0:1, :, :].mean(dim=(2, 3)),
                out_temp[:, 1:, :, :].view(bs, -1)
            ], dim=1)

class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""

    def __init__(self, net, sigma=1):
        super().__init__()
        self.net = net
        sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        self.register_parameter("sigma", sigma)

    def log_likelihood(self, x, z):
        decoder_out = self.net(z)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - decoder_out) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        return loglik

    def forward(self, z):
        return self.net(z)

    def sample(self, z):
        x_hat = self.net(z)
        return x_hat + torch.randn_like(x_hat) * self.sigma

"""
ConvNet for (1, 28, 28) image, following architecture in (Ghosh et al., 2019)
"""

class ConvNet28(nn.Module):
    def __init__(
        self, in_chan=1, out_chan=64, nh=32, out_activation="linear", activation="relu"
    ):
        """nh: determines the numbers of conv filters"""
        super(ConvNet28, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation

        layers = [
            self.conv1,
            get_activation(activation),
            self.conv2,
            get_activation(activation),
            self.conv3,
            get_activation(activation),
            self.conv4,
            get_activation(activation),
            self.fc1,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)


class DeConvNet28(nn.Module):
    def __init__(
        self,
        in_chan=1,
        out_chan=1,
        nh=32,
        out_activation="sigmoid",
        activation="relu",
    ):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet28, self).__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(
            nh * 32, nh * 16, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.conv2 = nn.ConvTranspose2d(
            nh * 16, nh * 8, kernel_size=2, stride=2, padding=1, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(nh * 8, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation

        layers = [
            self.fc1,
            get_activation(activation),
            self.conv1,
            get_activation(activation),
            self.conv2,
            get_activation(activation),
            self.conv3,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.net(x)
        return x

"""
ConvNet for (3, 32, 32) image, following architecture in (Ghosh et al., 2019)
"""

class ConvNet32(nn.Module):
    def __init__(
        self, in_chan=1, out_chan=64, nh=32, out_activation="linear", activation="relu"
    ):
        """nh: determines the numbers of conv filters"""
        super(ConvNet32, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=4, bias=True, stride=2)
        # self.bn1 = nn.BatchNorm2d(nh * 4)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=4, bias=True, stride=2)
        # self.bn2 = nn.BatchNorm2d(nh * 8)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=4, bias=True, stride=2)
        # self.bn3 = nn.BatchNorm2d(nh * 16)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=2, bias=True, stride=2)
        # self.bn4 = nn.BatchNorm2d(nh * 32)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation

        layers = [
            self.conv1,
            #   self.bn1,
            get_activation(activation),
            self.conv2,
            #   self.bn2,
            get_activation(activation),
            self.conv3,
            #   self.bn3,
            get_activation(activation),
            self.conv4,
            #   self.bn4,
            get_activation(activation),
            self.fc1,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)


class DeConvNet32(nn.Module):
    def __init__(
        self,
        in_chan=1,
        out_chan=1,
        nh=32,
        out_activation="sigmoid",
        activation="relu",
    ):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet32, self).__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        # self.bn1 = nn.BatchNorm2d(nh * 32)
        self.conv1 = nn.ConvTranspose2d(
            nh * 32, nh * 16, kernel_size=4, stride=2, padding=1, bias=True
        )
        # self.bn2 = nn.BatchNorm2d(nh * 16)
        self.conv2 = nn .ConvTranspose2d(
            nh * 16, nh * 8, kernel_size=4, stride=2, padding=1, bias=True
        )
        # self.bn3 = nn.BatchNorm2d(nh * 8)
        self.conv3 = nn.ConvTranspose2d(nh * 8, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation

        layers = [
            self.fc1,
            #   self.bn1,
            get_activation(activation),
            self.conv1,
            #   self.bn2,
            get_activation(activation),
            self.conv2,
            #   self.bn3,
            get_activation(activation),
            self.conv3,
        ]

        if self.out_activation == "tanh":
            layers.append(nn.Tanh())
        elif self.out_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.net(x)
        return x


"""
ConvNet for (3, 64, 64) image, following architecture in (Ghosh et al., 2019)
"""

class ConvNet64(nn.Module):
    def __init__(
        self, in_chan=1, out_chan=64, nh=32, out_activation="linear", activation="relu"
    ):
        """nh: determines the numbers of conv filters"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=5, bias=True, stride=2)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=5, bias=True, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 16, kernel_size=5, bias=True, stride=2)
        self.conv4 = nn.Conv2d(nh * 16, nh * 32, kernel_size=5, bias=True, stride=2)
        self.fc1 = nn.Conv2d(nh * 32, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation
        
        layers = [
            self.conv1,
            get_activation(activation),
            self.conv2,
            get_activation(activation),
            self.conv3,
            get_activation(activation),
            self.conv4,
            get_activation(activation),
            self.fc1,
        ]

        out_act = get_activation(out_activation)
        if out_act is not None:
            layers.append(out_act)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(2).squeeze(2)


class DeConvNet64(nn.Module):
    def __init__(
        self,
        in_chan=1,
        out_chan=1,
        nh=32,
        out_activation="sigmoid",
        activation="relu",
    ):
        """nh: determines the numbers of conv filters"""
        super().__init__()
        self.fc1 = nn.ConvTranspose2d(in_chan, nh * 32, kernel_size=8, bias=True)
        self.conv1 = nn.ConvTranspose2d(
            nh * 32, nh * 16, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv2 = nn.ConvTranspose2d(
            nh * 16, nh * 8, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv3 = nn.ConvTranspose2d(
            nh * 8, nh * 4, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv4 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = out_activation
        self.activation = activation

        layers = [
            self.fc1,
            get_activation(activation),
            self.conv1,
            get_activation(activation),
            self.conv2,
            get_activation(activation),
            self.conv3,
            get_activation(activation),
            self.conv4,
        ]

        out_act = get_activation(out_activation)
        if out_act is not None:
            layers.append(out_act)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 4:
            pass
        else:
            x = x.unsqueeze(2).unsqueeze(2)
        x = self.net(x)
        return x
    
############################################################

class vf_FC_vec_grasp(nn.Module): # added for grasping
    def __init__(
        self,
        in_chan=13,
        lat_chan=2+6,
        out_chan=6,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec_grasp, self).__init__()
        # assert in_chan == out_chan + 7  # Commented out for motion compatibility
        self.in_chan = in_chan
        self.lat_chan = lat_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan + lat_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x_t, target_T, t, v, g):
        n = x_t.shape[0]
        assert x_t.shape == (n, 4, 4)
        assert target_T.shape == (n, 4, 4)
        
        # Flatten both current and target poses
        current_flatten = torch.cat([x_t[:,0:3,0],x_t[:,0:3,1],x_t[:,0:3,2],x_t[:,0:3,3]], dim=1) # [B, 12]
        target_flatten = torch.cat([target_T[:,0:3,0],target_T[:,0:3,1],target_T[:,0:3,2],target_T[:,0:3,3]], dim=1) # [B, 12]
        
        # Concatenate: current(12) + target(12) + time(1) + geometry(3) + pointcloud_features(2048) = 2076
        input_features = torch.cat([current_flatten, target_flatten, t, g, v], dim=1)
        p0 = self.net(input_features)
        return p0 # output : twist (6)


class vf_FC_vec_motion(nn.Module): # added for motion planning
    def __init__(
        self,
        in_chan=25,     # current(12) + target(12) + time(1) = 25
        lat_chan=2048,  # pointcloud features from DGCNN
        out_chan=6,     # SE(3) twist vector
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super(vf_FC_vec_motion, self).__init__()
        
        self.in_chan = in_chan
        self.lat_chan = lat_chan
        self.out_chan = out_chan
        
        if l_hidden is None:
            l_hidden = [2048, 1024, 512, 512, 512]
        if activation is None:
            activation = ['relu'] * len(l_hidden)
        if out_activation is None:
            out_activation = 'linear'
            
        l_neurons = l_hidden + [out_chan]
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan + lat_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x_t, target_pose, t, v, g=None):
        """
        Motion planning velocity field
        
        Args:
            x_t: [B, 4, 4] current SE(3) pose (from CFM interpolation)
            target_pose: [B, 4, 4] target SE(3) pose (goal condition)
            t: [B, 1] time parameter
            v: [B, lat_chan] pointcloud features
        
        Returns:
            velocity: [B, 6] SE(3) twist vector [ω, v]
        """
        n = x_t.shape[0]
        assert x_t.shape == (n, 4, 4)
        assert target_pose.shape == (n, 4, 4)
        
        # Flatten SE(3) matrices
        current_flatten = torch.cat([
            x_t[:,0:3,0], x_t[:,0:3,1], 
            x_t[:,0:3,2], x_t[:,0:3,3]
        ], dim=1)  # [B, 12]
        
        target_flatten = torch.cat([
            target_pose[:,0:3,0], target_pose[:,0:3,1],
            target_pose[:,0:3,2], target_pose[:,0:3,3]  
        ], dim=1)   # [B, 12]
        
        # Combine all inputs
        combined_input = torch.cat([
            current_flatten,  # current pose [12D]
            target_flatten,   # target pose [12D] 
            t,               # time [1D]
            v                # pointcloud features [lat_chan]
        ], dim=1)
        if g is not None:
            # append radii(3)
            if g.dim() == 1:
                g = g.unsqueeze(0).repeat(x_t.shape[0], 1)
            combined_input = torch.cat([combined_input, g], dim=1)
        
        # Predict velocity
        velocity = self.net(combined_input)  # [B, 6]
        return velocity
    