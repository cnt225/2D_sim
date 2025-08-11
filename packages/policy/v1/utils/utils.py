import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rotate, InterpolationMode
from torch.distributed import init_process_group
# from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def torch_clip_acos(x, eps=1.0e-6):
    return torch.acos(torch.clamp(x, min=-1+eps, max=1-eps))

def draw_figure_of_t(t, nrow=10):
    fig = plt.figure()
    ncol = int(len(t)/nrow) + 1
    for i, tt in enumerate(t):
        # row = (i + 1)%10
        # col = int(i/10) + 1
        ax = fig.add_subplot(ncol, nrow, i+1)
        ax.arrow(0, 0, tt[0].item(), tt[1].item())
        ax.scatter(0, 0)
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))
        ax.axis('off')
    fig.tight_layout()
    plt.close()
    return fig

def data_aug_by_random_rotate(d, t):
    r_angle = (torch.rand(1)).item()*360
    d = rotate(d.unsqueeze(0), angle=r_angle, interpolation = InterpolationMode.BILINEAR).squeeze(0)
    rad_angle = r_angle * torch.pi/180
    rot_matrix = torch.tensor([
        [np.cos(rad_angle), -np.sin(rad_angle)], 
        [np.sin(rad_angle), np.cos(rad_angle)]
    ], dtype=torch.float32)
    t = (rot_matrix@t.unsqueeze(-1)).squeeze(-1)
    return d, t

def ddp_setup():
    init_process_group(backend="nccl")
    
def spline_coeffs_fitting(model, x1, num_interpolants=[2, 4, 4], z_dim=128, img_size=[3, 32, 32]):
    # model: ae
    # x1: bs x ...
    bs = len(x1)
    zero_img = torch.zeros_like(x1)
    
    latent_interpolants = torch.cat(
        [(1-t/(sum(num_interpolants)-1)) * model.encode(zero_img).unsqueeze(1) \
            + t/(sum(num_interpolants)-1) * model.encode(x1).unsqueeze(1) for t in range(sum(num_interpolants))], dim=1)
    images_interpolants = model.decode(latent_interpolants.view(-1, z_dim)).view(bs, -1, *img_size)[:, 1:-1]

    t = torch.linspace(0, 1, sum(num_interpolants)).to(x1)
    t = torch.cat([t[0:1], t[num_interpolants[0]:num_interpolants[0]+num_interpolants[1]], t[-1:]])
    
    x = torch.cat([
        zero_img.unsqueeze(1),
        images_interpolants[:, num_interpolants[0]:num_interpolants[0]+num_interpolants[1]],
        x1.unsqueeze(1)
    ], dim=1).view(bs, -1, torch.prod(torch.tensor(img_size)).item()) 

    coeffs = natural_cubic_spline_coeffs(t, x) 
    return coeffs

# class NaturalCubicSplineV2(NaturalCubicSpline):
#     def __init__(self, coeffs):
#         super(NaturalCubicSplineV2, self).__init__(coeffs)
        
#     def batchwise_evaluate(self, t):
#         bs = len(t)
#         indices = torch.arange(bs)
#         fractional_part, index = self._interpret_t(t)
#         fractional_part = fractional_part.unsqueeze(-1)
#         inner = self._c[indices, index, :] + self._d[indices, index, :] * fractional_part
#         inner = self._b[indices, index, :] + inner * fractional_part
#         return self._a[indices, index, :] + inner * fractional_part
    
#     def batchwise_derivative(self, t, order=1):
#         bs = len(t)
#         indices = torch.arange(bs)
#         fractional_part, index = self._interpret_t(t)
#         fractional_part = fractional_part.unsqueeze(-1)
#         if order == 1:
#             inner = 2 * self._c[indices, index, :] + 3 * self._d[indices, index, :] * fractional_part
#             deriv = self._b[indices, index, :] + inner * fractional_part
#         elif order == 2:
#             deriv = 2 * self._c[indices, index, :] + 6 * self._d[indices, index, :] * fractional_part
#         else:
#             raise ValueError('Derivative is not implemented for orders greater than 2.')
#         return deriv

def transform_control_points(Ts, control_pts):
    if type(Ts) == np.ndarray:
        assert type(control_pts) == np.ndarray, "Type of 'control_pts' must be numpy array as 'Ts'"

        Ts_repeated = Ts.repeat(len(control_pts), axis=0)
        control_pts_repeated = np.tile(control_pts, (len(Ts), 1))

        control_pts_transformed = (Ts_repeated @ np.concatenate([control_pts_repeated, np.ones_like(control_pts_repeated[:, :1])], axis=1)[:, :, None])[:, :3, 0]
        control_pts_transformed = control_pts_transformed.reshape(len(Ts), len(control_pts), -1)

    if type(Ts) == torch.Tensor:
        assert type(control_pts) == torch.Tensor, "Type of 'control_pts' must be torch tensor as 'Ts'"

        Ts_repeated = Ts.repeat_interleave(len(control_pts), dim=0)
        control_pts_repeated = control_pts.repeat(len(Ts), 1).to(Ts_repeated)

        control_pts_transformed = (Ts_repeated @ torch.cat([control_pts_repeated, torch.ones_like(control_pts_repeated[:, :1])], dim=1)[:, :, None])[:, :3]
        control_pts_transformed = control_pts_transformed.reshape(len(Ts), len(control_pts), -1)

    return control_pts_transformed

def calculate_accuracy(pred, target):
    num_correct = (pred == target).sum()
    num_total = len(target)

    accuracy = num_correct / num_total

    return accuracy

def init_parameter(pc, type = 'Mug', device = None):
    def show_pc(pc1, pc2, pc3):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pc1[:,0], pc1[:,1], pc1[:,2], color='b')
        ax.scatter(pc2[:,0], pc2[:,1], pc2[:,2], color='r')
        ax.scatter(pc3[:,0], pc3[:,1], pc3[:,2], color='g')
        fig.show()

    def kmeans(data, k, init_centroids=None, num_epochs=100):
        num_samples, num_features = data.size()
        if init_centroids == None:
            centroids = data[torch.randperm(num_samples)[:k]].to(device)
        else:
            centroids = init_centroids
        
        for _ in range(num_epochs):
            distances = torch.cdist(data, centroids).to(device)
            assignments = torch.argmin(distances, dim=1)
            new_centroids = torch.zeros(k, num_features).to(device)
            for cluster in range(k):
                cluster_points = data[assignments == cluster]
                if len(cluster_points) > 0:
                    new_centroids[cluster] = cluster_points.mean(dim=0)
            if torch.all(new_centroids == centroids):
                break
            
            centroids = new_centroids
        
        return centroids, assignments
    
    def dist_c(pt):
        return torch.sqrt(torch.sum((pt[0]-pt[1])**2)) + torch.sqrt(torch.sum((pt[2]-pt[1])**2)) + torch.sqrt(torch.sum((pt[0]-pt[2])**2))

    def norm_(vector):
        return torch.linalg.vector_norm(vector)
    # def inv2(matrix):
    #     a, b, c, d = matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1]
    #     det = a * d - b * c
    #     return torch.tensor([[d/det, -b/det],[-c/det, a/det]])
    # def inv3(matrix):
    #     a, b, c = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    #     d, e, f = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    #     g, h, i = matrix[2, 0], matrix[2, 1], matrix[2, 2]
        
    #     det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)        
    #     inv_det = 1.0 / det
        
    #     a_inv = (e * i - f * h) * inv_det
    #     b_inv = (c * h - b * i) * inv_det
    #     c_inv = (b * f - c * e) * inv_det
        
    #     d_inv = (f * g - d * i) * inv_det
    #     e_inv = (a * i - c * g) * inv_det
    #     f_inv = (c * d - a * f) * inv_det
        
    #     g_inv = (d * h - e * g) * inv_det
    #     h_inv = (b * g - a * h) * inv_det
    #     i_inv = (a * e - b * d) * inv_det
        
    #     return torch.tensor([[a_inv, b_inv, c_inv],
    #                         [d_inv, e_inv, f_inv],
    #                         [g_inv, h_inv, i_inv]])  

    def info_circle(_3points): # center, radius, direction
        a = _3points[0]
        b = _3points[1]
        c = _3points[2]
        sintheta = norm_(torch.cross(b-a, c-b)) / norm_(b-a) / norm_(c-b)
        if torch.abs(sintheta)<0.01:
            return torch.Tensor([0, 0, 0]).to(device), 0.001, torch.Tensor([0, 0, 1]).to(device)
        X = torch.cat([a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)], dim=0)
        y = torch.Tensor([1, 1, 1]).to(device)
        try:
            inv1 = torch.linalg.inv(X.T @ X)
        except:
            return torch.Tensor([0, 0, 0]).to(device), 0.001, torch.Tensor([0, 0, 1]).to(device)
        n = inv1 @ (X.T) @ (y.unsqueeze(1))
        n = n.squeeze()
        # n = torch.cross(b-a, c-b)
        n = n/norm_(n)
            
        u = torch.cross(b-a, n)
        v = torch.cross(c-b, n)

        X = torch.Tensor([[u[0], -v[0]], [u[1], -v[1]]])
        y = torch.Tensor([(c[0] - a[0])/2, (c[1] - a[1])/2])
        try:
            inv2 = torch.linalg.inv(X)
        except:
            return torch.Tensor([0, 0, 0]).to(device), 0.001, torch.Tensor([0, 0, 1]).to(device)
        t = inv2 @ (y.unsqueeze(1))
        t = t.squeeze()
            
        center = (a+b)/2 + u*t[0]
        radius = norm_(center-a)
        direction = n
        return center, radius, direction
    def calculate_num(pt, c, r, d, eps = 0.05):
        L = len(pt)
        P = pt
        C = c.unsqueeze(0).repeat(L, 1)
        D = d.unsqueeze(0).repeat(L, 1)
        K = (P-C) - (P-C) @ (d.unsqueeze(1)) @ (d.unsqueeze(0))
        L = torch.norm((P-C) - r * K / torch.norm(K, dim=1).unsqueeze(1), dim=1)
        count = torch.sum(L < eps)
        # count2 = 0
        # for i in range(len(pt)):
        #     p = pt[i]
        #     k = (p-c) - torch.dot((p-c), d) / norm_(d) / norm_(d) * d
        #     l = norm_((p-c) - r / norm_(k) * k)
        #     if l < eps:
        #         count2 += 1
        # assert count == count2, f"{count}, {count2}"
        return count

    def ransac_circle(pt, num_ransac = 100): # x, y, z, nx, ny, nz
        max_num = 0
        parameter = {}
        eps = 0.04
        for _ in range(num_ransac):
            _3points = pt[np.random.choice(range(len(pt)), 3, replace=False)]
            center, radius, direction = info_circle(_3points)

            num = calculate_num(pt, center, radius, direction, eps=eps)
            if num > max_num:
                max_num = num
                parameter = {'center':center, 'radius':radius, 'direction':direction}
            # if max_num == len(pt):
            #     eps *= 0.8
            #     max_num -= 1
        return parameter, max_num
    
    def shorter_dist(pt, pc):
        dist = torch.sqrt(torch.sum((pt.reshape(-1, 3).repeat(len(pc), 1) - pc) **2, dim=1))
        return torch.min(dist)

    def shorter_dist_sphere(pt, pc, r):
        dist = torch.sqrt(torch.sum((pt.reshape(-1, 3).repeat(len(pc), 1) - pc) **2, dim=1))
        return torch.sum(dist < r)

    def max_short_dist(pt1, pt2, pc):
        dist1 = torch.sqrt(torch.sum((pt1.reshape(-1, 3).repeat(len(pc), 1) - pc) **2, dim=1)).unsqueeze(1)
        dist2 = torch.sqrt(torch.sum((pt2.reshape(-1, 3).repeat(len(pc), 1) - pc) **2, dim=1)).unsqueeze(1)
        dist = torch.cat([dist1, dist2], dim=1)
        mini = torch.min(dist, dim=1).values
        return torch.argmax(mini)

    def draw_pc_circle(pc, pc1,pc2,pc3, center_list, radius_list, direction_list):
        COLOR = ['r','g','b','k']
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], color='b', alpha=0.2)
        ax.scatter(pc1[:,0], pc1[:,1], pc1[:,2], color='r')
        ax.scatter(pc2[:,0], pc2[:,1], pc2[:,2], color='g')
        ax.scatter(pc3[:,0], pc3[:,1], pc3[:,2], color='k')
        for i, (center, radius, direction) in enumerate(zip(center_list, radius_list, direction_list)):
            theta = torch.linspace(0, 2*3.1415926536, 20)
            xdir = pc[0] - center
            if torch.abs(norm_(torch.cross(xdir, direction)) / norm_(xdir))<0.01:
                xdir = pc[1] - center
                if torch.abs(norm_(torch.cross(xdir, direction)) / norm_(xdir))<0.01:
                    xdir = pc[2] - center
            xdir = xdir - torch.dot(xdir, direction) * direction
            xdir = xdir / norm_(xdir)
            ydir = torch.cross(direction, xdir)
            pos = center.reshape(-1, 3).repeat((len(theta),1))+ \
                    radius * torch.cos(theta).reshape(-1, 1) * xdir.reshape(-1, 3) + \
                    radius * torch.sin(theta).reshape(-1, 1) * ydir.reshape(-1, 3)
            ax.plot(pos[:,0], pos[:,1], pos[:,2], color = COLOR[i])
        fig.show()
        return ax
        # show_animation_3d(fig)

    assert pc.shape == (1000, 3)
    if type == 'Mug':
        if device == None:
            device = pc.device
        dist = torch.norm(pc, dim=1)
        # index10 = torch.topk(dist, 10).indices
        # index100 = torch.topk(dist, 200).indices[10:]
        # index1000 = torch.topk(dist, 1000).indices[200:]
        # print(index)
        # print(pc[index].shape)
        # show_pc(pc[index10],pc[index100],pc[index1000])
        index = torch.topk(dist, 250).indices
        feature_points = pc[index]

        K = 3
        centroids = torch.zeros((3,3)).to(device)
        for _ in range(1000):
            center = feature_points[torch.randint(len(feature_points),(3,))]
            # print(center)
            # print(centroids)
            if dist_c(center)>dist_c(centroids):
                centroids = center

        _, assignments = kmeans(feature_points, K, centroids)
        F1 = feature_points[assignments == 0]
        F2 = feature_points[assignments == 1]
        F3 = feature_points[assignments == 2]
        # show_pc(F1,F2,F3)
        parameter1, m1 = ransac_circle(F1)
        parameter2, m2 = ransac_circle(F2)
        parameter3, m3 = ransac_circle(F3)

        # print(parameter1, parameter2, parameter3)
        # print(len(F1), len(F2), len(F3))
        # print(m1, m2, m3)
        c12 = torch.abs(torch.dot(parameter1['direction'], parameter2['direction']))
        c13 = torch.abs(torch.dot(parameter1['direction'], parameter3['direction']))
        c23 = torch.abs(torch.dot(parameter2['direction'], parameter3['direction']))
        d12 = norm_(parameter1['center'] - parameter2['center'])
        d13 = norm_(parameter1['center'] - parameter3['center'])
        d23 = norm_(parameter2['center'] - parameter3['center'])
        e12 = c12 + d12
        e13 = c13 + d13
        e23 = c23 + d23
        if e12>e13 and e12>e23:
            para1, para2, trd = parameter1, parameter2, F3
            mode = (0, 1)
            notmode = 2
        elif e13>e12 and e13>e23:
            para1, para2, trd = parameter1, parameter3, F2
            mode = (0, 2)
            notmode = 1
        else:
            para1, para2, trd = parameter2, parameter3, F1
            mode = (1, 2)
            notmode = 0
        
        rad = min(para1['radius'], para2['radius'])
        sd1 = shorter_dist_sphere(para1['center'], pc, rad/2)
        sd2 = shorter_dist_sphere(para2['center'], pc, rad/2)
        if sd1>sd2:
            zdirection = para2['center'] - para1['center']
            bottom = para1['center']
        else:
            zdirection = para1['center'] - para2['center']
            bottom = para2['center']
        zdirection = zdirection / norm_(zdirection)
        
        farpt = pc[max_short_dist(para1['center'], para2['center'], pc)] ## trd
        ydirection = farpt - bottom
        ydirection = ydirection - (torch.dot(ydirection, zdirection)) * zdirection
        ydirection = -ydirection / norm_(ydirection)
        assert torch.dot(ydirection, zdirection)<0.001, "direction error"
        xdirection = torch.cross(ydirection, zdirection)
        T = torch.eye(4)
        T[:3, :3] = torch.cat([xdirection.unsqueeze(1), ydirection.unsqueeze(1), zdirection.unsqueeze(1)], dim=1)
        # center_list = [parameter1['center'], parameter2['center'], parameter3['center']]
        # radius_list = [parameter1['radius'], parameter2['radius'], parameter3['radius']]
        # direction_list = [parameter1['direction'], parameter2['direction'], parameter3['direction']]

        # center_list2 = [center_list[mode[0]], center_list[mode[1]], center_list[notmode]]
        # radius_list2 = [radius_list[mode[0]], radius_list[mode[1]], radius_list[notmode]]
        # direction_list2 = [direction_list[mode[0]], direction_list[mode[1]], direction_list[notmode]]
        # ax = draw_pc_circle(pc, F1,F2,F3, center_list2, radius_list2, direction_list2)
        # plot_se3(T, ax)
        return T
