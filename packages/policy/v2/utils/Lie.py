import numpy as np
from numpy.linalg import inv
import torch
import math
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PI = math.pi
EPS = 1e-6
FUNCTION_LIST = ['bracket_so3', 'bracket_se3', 'Lie_bracket', 'proj_minus_one_plus_one', 'is_SO3', 'is_SE3', 'inv_SO3', 'inv_SE3', 
                 'log_SO3', 'log_SE3', 'exp_so3', 'exp_se3', 'large_Ad', 'small_Ad', 
                 'Q_to_SO3', 'SO3_to_Q', 'getNullspace', 'revoluteTwist']

def bracket_so3(w):
    n = w.shape[0]
    # nx3x3 skew --> nx3 vector
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1), w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    # nx3 vector --> nx3x3 skew
    elif w.shape == (n, 3):
        zero1 = torch.zeros(n, 1, 1).to(w)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:,2],  w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1,  -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0],  zero1], dim=2)], dim=1)
    else:
        raise ValueError(f"bracket_so3 : Shape should be n*3 or n*3*3. Current shape : {w.shape}")
    return W

def bracket_se3(V):
    if isinstance(V, str):
        return 'trace error'
    n = V.shape[0]
    # nx4x4 skew --> nx6 vector
    if V.shape == (n, 4, 4):
        out = torch.cat([-V[:, 1, 2].unsqueeze(-1), V[:, 0, 2].unsqueeze(-1),
                         -V[:, 0, 1].unsqueeze(-1), V[:, :3, 3]], dim=1)
    # nx6 vector --> nx4x4 skew
    elif V.shape == (n, 6):
        W = bracket_so3(V[:, 0:3])
        out = torch.cat([torch.cat([W, V[:, 3:].unsqueeze(-1)], dim=2),
                         torch.zeros(n, 1, 4).to(V)], dim=1)
    else:
        raise ValueError(f"bracket_se3 : Shape should be n*6 or n*4*4. Current shape : {V.shape}")
    return out

def Lie_bracket(u, v):
    assert u.shape[0] == v.shape[0], f'Lie_bracket : dim(u)({u.shape[0]})!=dim(v)({v.shape[0]})'
    n = u.shape[0]
    if u.shape == (n, 3):
        u = bracket_so3(u)
    elif u.shape == (n, 6):
        u = bracket_se3(u)
    n = v.shape[0]
    if v.shape ==(n, 3):
        v = bracket_so3(v)
    elif v.shape == (n, 6):
        v = bracket_se3(v)
    return torch.einsum('nij, njk -> nik', u, v) - torch.einsum('nij, njk -> nik', v, u)

def proj_minus_one_plus_one(x):
    x = torch.min(x, (1 - EPS) * (torch.ones(x.shape).to(x)))
    x = torch.max(x, (-1 + EPS) * (torch.ones(x.shape).to(x)))
    return x

def is_SO3(R):
    return torch.all(torch.le(torch.sum(torch.abs(torch.einsum("nij, nkj -> nik", R, R)-torch.eye(3).to(R).reshape(1, 3, 3).repeat(R.shape[0],1,1)), dim=(1,2)), \
           100 * EPS * 9 * torch.ones(R.shape[0]).to(R)))

def is_SE3(T):
    R = T[:,:3,:3]
    # assert is_SO3(R), f'is_SO3 : R = {T[not torch.le(torch.sum(torch.abs(torch.einsum("nij, nkj -> nik", R, R)-torch.eye(3).to(R).reshape(1, 3, 3).repeat(R.shape[0],1,1)), dim=(1,2)), EPS * 9 * torch.ones(R.shape[0]).to(R))]}'
    return is_SO3(R) and torch.equal(T[:,3,0:3], torch.zeros_like(T[:,3,0:3])) and torch.equal(T[:,3,3], torch.ones_like(T[:,3,3]))

def inv_SO3(R):
    n = R.shape[0]
    assert R.shape == (n, 3, 3), "inv_SO3 : R shape error"
    assert is_SO3(R), "inv_SO3 : R component error"
    return R.transpose(1, 2).to(R)

def inv_SE3(T):
    n = T.shape[0]
    assert T.shape == (n, 4, 4), "inv_SE3 : T shape error"
    assert is_SE3(T), "inv_SE3 : T component error"
    R, p = T[:, :3, :3], T[:, :3, 3].unsqueeze(-1)
    T_inv = T.new_zeros(n, 4, 4)
    T_inv[:, :3, :3] = inv_SO3(R)
    T_inv[:, :3, 3] = - (inv_SO3(R) @ p).view(n, 3)
    T_inv[:, 3, 3] = 1
    return T_inv

def log_SO3(R):
    # return logSO3(R)
    n = R.shape[0]
    assert R.shape == (n, 3, 3), "log_SO3 : R shape error"
    assert is_SO3(R), "log_SO3 : R component error"
    trace = torch.sum(R[:, range(3), range(3)], dim=1).to(R)
    omega = torch.zeros(R.shape).to(R)
    theta = torch.acos(proj_minus_one_plus_one((trace - 1) / 2)).to(R)
    temp = theta.unsqueeze(-1).unsqueeze(-1).to(R)

    omega[(torch.abs(trace + 1) > EPS) * (theta > EPS)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > EPS) * (theta > EPS)]
    
    omega_temp = (R[torch.abs(trace + 1) <= EPS] - torch.eye(3).to(R)) / 2
    
    omega_vector_temp = torch.sqrt(omega_temp[:, range(3), range(3)] + torch.ones(3).to(R))
    omega_vector_temp[torch.isnan(omega_vector_temp)] = 0
    w1 = omega_vector_temp[:, 0]
    w2 = omega_vector_temp[:, 1] * (torch.sign(omega_temp[:, 0, 1]) + (w1 == 0))
    w3 = omega_vector_temp[:, 2] * torch.sign(4 * torch.sign(omega_temp[:, 0, 2]) + 2 * (w1 == 0) * torch.sign(omega_temp[:, 1, 2]) + 1 * (w1 == 0) * (w2 == 0))
    omega_vector = torch.cat([w1.unsqueeze(1), w2.unsqueeze(1), w3.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= EPS] = bracket_so3(omega_vector) * PI

    return omega

def logSO3(SO3):
    nBatch = len(SO3)
    trace = torch.einsum('xii->x', SO3)
    regularID = (trace + 1).abs() >= EPS
    singularID = (trace + 1).abs() < EPS
    theta = torch.acos(proj_minus_one_plus_one((trace - 1) / 2)).view(nBatch, 1, 1)
    so3mat = SO3.new_zeros(nBatch, 3, 3)
    # regular
    if any(regularID):
        so3mat[regularID, :, :] = (SO3[regularID] - SO3[regularID].transpose(1, 2)) / (2 * theta[regularID].sin()) * theta[regularID]
    # singular
    if any(singularID):
        if all(SO3[singularID, 2, 2] != -1):
            r = SO3[singularID, 2, 2]
            w = SO3[singularID, :, 2]
            w[:, 2] += 1
        elif all(SO3[singularID, 1, 1] != -1):
            r = SO3[singularID, 1, 1]
            w = SO3[singularID, :, 1]
            w[:, 1] += 1
        elif all(SO3[singularID, 0, 0] != -1):
            r = SO3[singularID, 0, 0]
            w = SO3[singularID, :, 0]
            w[:, 0] += 1
        else:
            print(f'ERROR: all() is somewhat ad-hoc. should be fixed.')
            exit(1)
        so3mat[singularID, :, :] = bracket_so3(torch.pi / (2 * (1 + r)).sqrt().view(-1, 1) * w)
    # trace == 3 (zero rotation)
    if any((trace - 3).abs() < 1e-10):
        so3mat[(trace - 3).abs() < 1e-10] = 0
    return so3mat

# def log_SO3_T(T): # only log on R
#     # dim T = n,4,4
#     R = T[:, 0:3, 0:3]  # dim n,3,3
#     p = T[:, 0:3, 3].unsqueeze(-1)  # dim n,3,1
#     n = T.shape[0]
#     W = log_SO3(R.to(device))  # n,3,3

#     return torch.cat([torch.cat([W, p], dim=2), torch.zeros(n, 1, 4, device=device)], dim=1)  # n,4,4

def log_SE3(T):
    ##### ↓ Not Checked ↓ #####
    #dim T = n,4,4
    n = T.shape[0]
    assert T.shape == (n, 4, 4), "log_SE3 : T shape error"
    assert is_SE3(T), "log_SE3 : T component error"
    R = T[:,0:3,0:3] # dim n,3,3
    W = log_SO3(R) # n,3,3    
    assert not isinstance(W, str), 'trace error'
    
    trace = torch.einsum('nii->n', R)
    regularID = (trace-3).abs() >= EPS
    zeroID = (trace-3).abs() < EPS
    S = torch.zeros(n, 4, 4).to(T)
    if any(zeroID):
        S[zeroID, :3, 3] = T[zeroID, :3, 3]
    if any(regularID):
        nRegular = sum(regularID)
        so3 = log_SO3(T[regularID, :3, :3])
        theta = (torch.acos(proj_minus_one_plus_one(0.5*(trace[regularID]-1)))).reshape(nRegular, 1, 1)
        wmat = so3 / theta
        identity33 = torch.zeros_like(so3).to(T)
        identity33[:, 0, 0] = identity33[:, 1, 1] = identity33[:, 2, 2] = 1
        invG = (1 / theta) * identity33 - 0.5 * wmat + (1 / theta - 0.5 / (0.5 * theta).tan()) * wmat @ wmat
        S[regularID, :3, :3] = so3
        S[regularID, :3, 3] = theta.view(nRegular, 1) * (invG @ T[regularID, :3, 3].view(nRegular, 3, 1)).reshape(nRegular, 3)
    return S

def exp_so3(w):
    n = w.shape[0]
    if w.shape == (n, 3, 3):
        so3vec = bracket_so3(w)
    elif w.shape == (n, 3):
        so3vec = w
    else:
        assert 0, f'exp_so3 : Shape should be n*3 or n*3*3. Current shape : {w.shape}'
    # Rodrigues' rotation formula
    theta = so3vec.norm(dim=1).view(n, 1)
    regularID = theta.view(n).abs() >= EPS
    theta_regularID = theta[regularID]
    wmat = bracket_so3(so3vec[regularID] / theta_regularID)
    R = w.new_zeros(n, 3, 3)
    R[:, 0, 0] = R[:, 1, 1] = R[:, 2, 2] = 1
    R[regularID] += theta_regularID.sin().view(theta_regularID.shape[0], 1, 1) * wmat
    R[regularID] += (1 - theta_regularID.cos()).view(theta_regularID.shape[0], 1, 1) * wmat @ wmat
    return R

# def exp_so3_from_screw(S):
#     n = S.shape[0]
#     if S.shape == (n, 4, 4):
#         S1 = bracket_so3(S[:, :3, :3]).clone()
#         S2 = S[:, 0:3, 3].clone()
#         S = torch.cat([S1, S2], dim=1)
#     # shape(S) = (n,6,1)
#     w = S[:, :3]  # dim= n,3
#     v = S[:, 3:].unsqueeze(-1)  # dim= n,3

#     T = torch.cat([torch.cat([exp_so3(w), v], dim=2),
#                    torch.zeros(n, 1, 4).to(S)], dim=1)
#     T[:, -1, -1] = 1
#     return T

def exp_se3(S):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = bracket_se3(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    elif S.shape == (n, 6):
        S = S
    else:
        assert 0, f'exp_se3 : Shape should be n*6 or n*4*4. Current shape : {S.shape}'
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:]  # dim= n,3
    theta = w.norm(dim=1)
    zeroID = theta.abs() < EPS
    T = S.new_zeros(n, 4, 4)
    if zeroID.any():
        T[zeroID, 0, 0] = T[zeroID, 1, 1] = T[zeroID, 2, 2] = T[zeroID, 3, 3] = 1
        T[zeroID, :3, 3] = v[zeroID]
    if (~zeroID).any():
        nNonZero = (~zeroID).sum()
        _theta = theta[~zeroID].reshape(nNonZero, 1)
        wmat = bracket_so3(w[~zeroID] / _theta)
        # G = eye * theta + (1-cos(theta)) [w] + (theta - sin(theta)) * [w]^2
        G = S.new_zeros(nNonZero, 3, 3)
        G[:, 0, 0] = G[:, 1, 1] = G[:, 2, 2] = _theta.view(nNonZero)
        G += (1 - _theta.cos()).view(nNonZero, 1, 1) * wmat
        G += (_theta - _theta.sin()).view(nNonZero, 1, 1) * wmat @ wmat
        # output
        T[~zeroID, :3, :3] = exp_so3(w[~zeroID])
        T[~zeroID, :3, 3] = (G @ (v[~zeroID] / _theta).view(nNonZero, 3, 1)).view(nNonZero, 3)
        T[~zeroID, 3, 3] = 1
    return T
    

def large_Ad(T):
    # R     0
    # [p]R  R
    n = T.shape[0]
    assert T.shape == (n, 4, 4), "large_Ad : T shape error"
    assert is_SE3(T), "large_Ad : T component error"
    R, p = T[:, :3, :3], T[:, :3, 3]
    Adj = T.new_zeros(n, 6, 6)
    Adj[:, :3, :3] = Adj[:, 3:, 3:] = R
    Adj[:, 3:, :3] = bracket_so3(p) @ R
    return Adj


def small_Ad(S):
    # [w] 0
    # [v] [w]
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = bracket_se3(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    elif S.shape == (n, 6):
        S = S
    else:
        assert 0, f'exp_se3 : Shape should be n*6 or n*4*4. Current shape : {S.shape}'
    w = S[:, :3]
    v = S[:, 3:]
    adj = S.new_zeros(n, 6, 6)
    adj[:, :3, :3] = adj[:, 3:, 3:] = bracket_so3(w)
    adj[:, 3:, :3] = bracket_so3(v)
    return adj

def se3v_dist(SR1, SR2):
    SR1 = SR1.reshape(-1, 6)
    SR2 = SR2.reshape(-1, 6)
    w1 = SR1[:, :3]
    v1 = SR1[:, 3:6]
    w2 = SR2[:, :3]
    v2 = SR2[:, 3:6]
    R1 = exp_so3(w1)
    R2 = exp_so3(w2)
    dist_R = bracket_so3(log_SO3(torch.einsum('nij,njk->nik', inv_SO3(R1), R2)))
    dist_P = v2 - v1
    geodesic = torch.sqrt(torch.sum(dist_R ** 2 + dist_P ** 2, dim = 1))
    return geodesic

def SE3_geodesic(SE3_1, SE3_2):
    bsize1 = SE3_1.shape[0]
    bsize2 = SE3_2.shape[0]
    assert SE3_1.shape == (bsize1, 4, 4) and SE3_2.shape == (bsize2, 4, 4), "SE3_geodesic : shape error"
    dist_R = bracket_so3(log_SO3(torch.einsum('nij,mjk->nmik', inv_SO3(SE3_1[:, 0:3, 0:3]), SE3_2[:, 0:3, 0:3]).reshape(-1, 3, 3))).reshape(bsize1, bsize2, 3)
    dist_P = SE3_1[:, 0:3, 3].unsqueeze(1).repeat(1, bsize2, 1) - SE3_2[:, 0:3, 3].unsqueeze(0).repeat(bsize1, 1, 1)
    geodesic = torch.sqrt(torch.sum(dist_R ** 2 + dist_P ** 2, dim = 2))
    assert geodesic.shape == (bsize1, bsize2)
    return geodesic

################################################################################################################
################################################### APPENDIX ###################################################
################################################################################################################


def Q_to_SO3(quaternions):
    assert quaternions.shape[1] == 4

    # initialize
    K = quaternions.shape[0]
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 0] ** 2
    yy = quaternions[:, 1] ** 2
    zz = quaternions[:, 2] ** 2
    ww = quaternions[:, 3] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 0] * quaternions[:, 1]
    xz = s[:, 0] * quaternions[:, 0] * quaternions[:, 2]
    yz = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xw = s[:, 0] * quaternions[:, 0] * quaternions[:, 3]
    yw = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    zw = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R

def SO3_to_Q(R):
    n = R.shape[0]
    assert R.shape == (n, 3, 3), "SO3_to_Q : R shape error"
    assert is_SO3(R), "SO3_to_Q : R component error"
    W = log_SO3(R) # n,3,3
    w = bracket_so3(W) #n,3
    theta = torch.norm(w, dim=1).unsqueeze(1)
    zeroID = theta.squeeze().abs() < EPS
    w_hat = R.new_zeros(n, 3)
    w_hat[~zeroID] = w[~zeroID]/(theta[~zeroID]) # n,3
    return torch.cat([w_hat[:,0].unsqueeze(-1)*torch.sin(theta/2),
                      w_hat[:,1].unsqueeze(-1)*torch.sin(theta/2),
                      w_hat[:,2].unsqueeze(-1)*torch.sin(theta/2),
                      torch.cos(theta/2)], dim=1)

def getNullspace(tensor2dim):
    if tensor2dim.is_complex():
        print('ERROR : getNullspace() fed by a complex number')
        exit(1)
    U, S, V = torch.Tensor.svd(tensor2dim, some=False, compute_uv=True)
    # threshold = torch.max(S) * torch.finfo(S.dtype).eps * max(U.shape[0], V.shape[1])
    # rank = torch.sum(S > threshold, dtype=int)
    rank = len(S)
    # return V[rank:, :].T.cpu().conj()
    return V[:, rank:]

def revoluteTwist(twist):
    nJoint, mustbe6 = twist.shape
    if mustbe6 != 6:
        print(f'[ERROR] revoluteTwist: twist.shape = {twist.shape}')
        exit(1)
    w = twist[:, :3]
    v = twist[:, 3:]
    w_normalized = w / w.norm(dim=1).view(nJoint, 1)
    proejctedTwists = torch.empty_like(twist)
    proejctedTwists[:, :3] = w_normalized
    wdotv = torch.sum(v * w_normalized, dim=1).view(nJoint, 1)
    proejctedTwists[:, 3:] = v - wdotv * w_normalized
    return proejctedTwists
