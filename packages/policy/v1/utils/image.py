import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2

from utils.sphere import sample_from_Uniform, normalize

PI = math.pi

def fill_center(img, color = (0,0,0)): # numpy
    start = (int(img.shape[0]/2), int(img.shape[1]/2))
    img[start]=color
    white = []
    white.append(start)
    dx = [0,1,0,-1]
    dy = [1,0,-1,0]
    while white:
        a = white.pop(0)
        for x, y in zip(dx, dy):
            if a[0]+x < 0 or a[0]+x > img.shape[0]-1 or a[1]+y < 0 or a[1]+y > img.shape[1]-1:
                continue
            if all(img[a[0]+x][a[1]+y] == [255,255,255]):
                img[a[0]+x][a[1]+y] = color
                white.append((a[0]+x, a[1]+y))

def image_shape(img_size = (128, 128), shape = 3, color = (0, 0, 0)): # torch
    '''
    image size : tuple (x, y)
    shape : triangle, square, pentagon, etc...
    color : tuple (R, G, B)

    output : (x, y, 3)
    '''
    assert shape >= 3, "you should put shape more than 3"
    img = np.zeros(img_size+(3,), np.uint8)
    img[:]=(255,255,255)
    circle_radius = 0.4 * min(img_size[0], img_size[1])
    X = np.zeros((shape, 2), dtype=np.int32)
    for i in range(shape):
        X[i,1] = img_size[0] - int(img_size[0]/2 + np.cos(i / shape * 2 * PI) * circle_radius)
        X[i,0] = int(img_size[1]/2 + np.sin(i / shape * 2 * PI) * circle_radius)
    cv2.polylines(img, [X], True, color)
    fill_center(img, color)
    return torch.Tensor(img)

@torch.no_grad()
def sample_from_rectangle_on_sphere(num_samples, **kwargs):
    data = []
    l_data = 0
    while l_data<num_samples:
        X = sample_from_Uniform(100)
        # longitude -pi/2 to pi/2, latitude -pi/3 to pi/3
        longi = torch.atan2(X[:, 1], X[:, 0])
        lati = torch.asin(X[:, 2])
        X = X [(lati<PI/3)&(lati>-PI/3)&(longi<PI/2)&(longi>-PI/2)]
        data.append(X)
        l_data += X.shape[0]
    return torch.cat(data, dim=0)[:num_samples]

@torch.no_grad()
def image_to_sphere(img, num_samples):
    img_size = img.shape
    data = []
    l_data = 0
    max_iter = 0
    while l_data<num_samples and max_iter<30000:
        X = sample_from_rectangle_on_sphere(100)
        # longitude -pi/6 to pi/6, latitude -pi/6 to pi/6
        longi = torch.atan2(X[:, 1], X[:, 0])
        lati = torch.asin(X[:, 2])
        x = torch.clamp(((PI/6 - lati)/(PI/3) * img_size[0]).to(torch.int64), min = 0, max = img_size[0]-1)
        y = torch.clamp(((longi + PI/6)/(PI/3) * img_size[1]).to(torch.int64), min = 0, max = img_size[1]-1)
        # print(img[x.to(torch.long),y.to(torch.long)])
        # print(torch.Tensor([255,255,255]))
        idx = (img[x.to(torch.long),y.to(torch.long)] != torch.Tensor([255,255,255])).all(axis=1)
        X = X[idx]
        data.append(X)
        l_data += X.shape[0]
        max_iter += 1
    assert max_iter!=30000, 'couldn\'t find samples (func : image_to_sphere)'

    return torch.cat(data, dim=0)[:num_samples]

@torch.no_grad()
def polygon_to_sphere(num_samples, img = None, coord = None, size=10):
    # img : h * w * 3  /  coord : n * 2 (0, 0 ~ size, size)
    if img == None:
        if coord == None:
            assert 0, "NO INPUT"
        else:
            pt_num = coord.shape[0]
            coord[:,0] = ((coord[:,0]/size)*2-1)*PI/6
            coord[:,1] = ((coord[:,1]/size)*2-1)*PI/6
            T = torch.rand(num_samples, pt_num)
            total = torch.sum(T, dim=1, keepdim = True)
            PT = ((T/total) @ coord)
            Z = torch.sin(PT[:,1]).reshape(-1,1)
            YX = torch.tan(PT[:,0]).reshape(-1,1)
            X = torch.sqrt((1-Z*Z)/(1+YX*YX)).reshape(-1,1)
            Y = X * YX.reshape(-1,1)
            # print(X.shape, Y.shape, Z.shape)
            return torch.cat([X,Y,Z], dim=1)
    else:
        assert 0, "FUTURE WORK"
        return None

@torch.no_grad()
def RK4_sphere_I2S(func, x0, t, v, **kwargs): # RK4 on sphere
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
        k1 = func(xt, t1, v)

        t2 = t0 + h/2
        k2 = func(xt + h * k1 / 2, t2, v)
        k3 = func(xt + h * k2 / 2, t2, v)

        t3 = t0 + h
        k4 = func(xt + h * k3, t3, v)

        xt = xt + h/6 * (k1 + 2 * k2 + 2 * k3 + k4) # exponential form으로 수정 필요
        xt = normalize(xt)
        Xs.append(xt.unsqueeze(0))

    return torch.cat(Xs, dim=0)