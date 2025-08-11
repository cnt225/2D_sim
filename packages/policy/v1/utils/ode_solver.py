import torch
from copy import deepcopy
from utils.Lie import log_SO3, exp_so3, Lie_bracket, bracket_so3

def get_ode_solver(name):
    try:
        return {
            'Euler' : ode_solver_simple,
            'RK4' : ode_solver_RK4_MK,
            'RK4exp' : ode_solver_RK4_exp,
        }[name]
    except:
        raise ("ODE solver {} not available".format(name))

@torch.no_grad()
def ode_solver_simple(func, x0, t, v, device = None): # x0 = (n, 4, 4), t = (T), v = (n, v)
    # t = 0, 0.01, 0.02 ... 1
    t = t.to(device)
    dt = t[1:] - t[:-1]
    v = v.repeat(x0.shape[0], 1).to(device)
    result = x0.new_zeros(t.shape + x0.shape).to(device)
    result[0] = x0.to(device)
    for i in range(t.shape[0]-1):
        x_next = deepcopy(result[i]).to(device)
        t_i = t[i].reshape(-1, 1).repeat(x0.shape[0], 1).to(device)
        s = func(x_next, t_i, v)
        w_s = s[:, 0:3]
        w_b = torch.einsum('nji, nj  -> ni', x_next[:,:3,:3], w_s) # w_s to w_b
        dp = s[:, 3:6]
        x_next[:,0:3,0:3] = torch.einsum('nij,njk->nik', x_next[:,0:3,0:3], exp_so3(dt[i] * w_b))
        x_next[:,0:3,3] += dt[i] * dp
        result[i+1] = x_next
    return result

@torch.no_grad()
def ode_solver_RK4(func, x0, t, v): # x0 = (n, 4, 4), t = (T), v = (n, v)
    # t = 0, 0.01, 0.02 ... 1
    raise NotImplementedError('RK4_general Not Available')

    dt = t[1:] - t[:-1]
    result = x0.new_zeros(t.shape + x0.shape)
    result[0] = x0
    for i in range(t.shape[0]-1):
        x1 = deepcopy(result[i])
        t1 = t[i].reshape(-1, 1).repeat(x0.shape[0], 1)
        h = dt[i].reshape(-1, 1).repeat(x0.shape[0], 1)

        V1 = func(x1, t1, v)
        w1 = V1[:, 0:3]
        v1 = V1[:, 3:6]

        t2 = t1 + h/2
        x2 = deepcopy(x1)
        u1 = 1/2 * h * w1
        x2[:,0:3,0:3] = torch.einsum('nij, njk -> nik', x2[:,0:3,0:3], exp_so3(u1))
        x2[:,0:3,3] += 1/2 * h * v1
        V2 = func(x2, t2, v)
        w2 = V2[:, 0:3]
        v2 = V2[:, 3:6]
        
        x3 = deepcopy(x1)
        u2 = 1/2 * h * w2
        x3[:,0:3,0:3] = torch.einsum('nij, njk -> nik', x3[:,0:3,0:3], exp_so3(u2))
        x3[:,0:3,3] += 1/2 * h * v2
        V3 = func(x3, t2, v)
        w3 = V3[:, 0:3]
        v3 = V3[:, 3:6]

        t3 = t1 + h
        x4 = deepcopy(x1)
        u3 = h * w3
        x4[:,0:3,0:3] = torch.einsum('nij, njk -> nik', x4[:,0:3,0:3], exp_so3(u3))
        x4[:,0:3,3] += h * v3
        V4 = func(x4, t3, v)
        w4 = V4[:, 0:3]
        v4 = V4[:, 3:6]

        X = deepcopy(x1)
        w = h / 6 * (w1 + 2 * w2 + 2 * w3 + w4)
        X[:, 0:3, 0:3] = torch.einsum('nij, njk -> nik', X[:,0:3,0:3], exp_so3(w))
        X[:, 0:3, 3] += 1/6 * h * (v1 + 2 * v2 + 2 * v3 + v4)
        result[i+1] = X
    return result

@torch.no_grad()
def ode_solver_RK4_MK(func, x0, t, v, device = None):
    # func.eval()
    dt = t[1:] - t[:-1]
    v = v.repeat(x0.shape[0], 1).to(device)
    result = x0.new_zeros(t.shape + x0.shape).to(device)
    result[0] = x0.to(device)
    for i in range(t.shape[0]-1):
        x1 = deepcopy(result[i]).to(device)
        t_i = t[i].reshape(-1, 1).repeat(x0.shape[0], 1).to(device)
        h = dt[i].reshape(-1, 1).repeat(x0.shape[0], 1).to(device)

        V1 = func(x1, t_i, v)
        V1[:, 0:3] = torch.einsum('nji, nj -> ni', x1[:, 0:3, 0:3], V1[:, 0:3]) # w_s to w_b
        w1 = bracket_so3(V1[:, 0:3])
        v1 = V1[:, 3:6]
        I1 = deepcopy(w1).to(device)

        u2 = h.reshape(-1,1,1) * 1/2 * w1
        u2 += 1/12 * h.reshape(-1,1,1) * Lie_bracket(I1, u2)
        x2 = deepcopy(x1).to(device)
        x2[:,0:3,0:3] = torch.einsum('nij, njk -> nik', x2[:,0:3,0:3], exp_so3(u2))
        x2[:,0:3,3] += 1/2 * h * v1
        V2 = func(x2, t_i + h/2, v)
        V2[:, 0:3] = torch.einsum('nji, nj -> ni', x2[:, 0:3, 0:3], V2[:, 0:3]) # w_s to w_b
        w2 = bracket_so3(V2[:, 0:3])
        v2 = V2[:, 3:6]

        u3 = h.reshape(-1,1,1) * 1/2 * w2
        u3 += 1/12 * h.reshape(-1,1,1) * Lie_bracket(I1, u3)
        x3 = deepcopy(x1).to(device)
        x3[:,0:3,0:3] = torch.einsum('nij, njk -> nik', x3[:,0:3,0:3], exp_so3(u3))
        x3[:,0:3,3] += 1/2 * h * v2
        V3 = func(x3, t_i + h/2, v)
        V3[:, 0:3] = torch.einsum('nji, nj -> ni', x3[:, 0:3, 0:3], V3[:, 0:3]) # w_s to w_b
        w3 = bracket_so3(V3[:, 0:3])
        v3 = V3[:, 3:6]

        u4 = h.reshape(-1,1,1) * w3
        u4 += 1/6 * h.reshape(-1,1,1) * Lie_bracket(I1, u4)
        x4 = deepcopy(x1).to(device)
        x4[:,0:3,0:3] = torch.einsum('nij, njk -> nik', x4[:,0:3,0:3], exp_so3(u4))
        x4[:,0:3,3] += h * v3
        V4 = func(x4, t_i + h, v)
        V4[:, 0:3] = torch.einsum('nji, nj -> ni', x4[:, 0:3, 0:3], V4[:, 0:3]) # w_s to w_b
        w4 = bracket_so3(V4[:, 0:3])
        v4 = V4[:, 3:6]

        I2 = (2 * (w2 - I1) + 2 * (w3 - I1) - (w4 - I1)) / h.reshape(-1,1,1)
        u = h.reshape(-1,1,1) * (1/6 * w1 + 1/3 * w2 + 1/3 * w3 + 1/6 * w4)
        u += 1/4 * h.reshape(-1,1,1) * Lie_bracket(I1, u) + 1/24 * h.reshape(-1,1,1) * h.reshape(-1,1,1) * Lie_bracket(I2, u)
        x_next = deepcopy(x1).to(device)
        x_next[:,0:3,0:3] = torch.einsum('nij,njk->nik', x_next[:,0:3,0:3], exp_so3(u))
        x_next[:,0:3,3] += 1/6 * h * (v1 + 2 * v2 + 2 * v3 + v4)
        result[i+1] = x_next
    return result

@torch.no_grad()
def ode_solver_RK4_exp(func, x0, t, v, device = None):
    # func.eval()
    dt = t[1:] - t[:-1]
    v = v.repeat(x0.shape[0], 1).to(device)
    result = x0.new_zeros(t.shape + x0.shape).to(device)
    result[0] = x0.to(device)
    for i in range(t.shape[0]-1):
        x1 = deepcopy(result[i]).to(device)
        t_i = t[i].reshape(-1, 1).repeat(x0.shape[0], 1).to(device)
        h = dt[i].reshape(-1, 1).repeat(x0.shape[0], 1).to(device)

        V1 = func(x1, t_i, v)
        w1 = V1[:, 0:3]
        v1 = V1[:, 3:6]

        x2 = deepcopy(x1).to(device)
        x2[:,0:3,0:3] = exp_so3(bracket_so3(log_SO3(x2[:, :3, :3])) + 1/2 * h * w1)
        x2[:,0:3,3] += 1/2 * h * v1
        V2 = func(x2, t_i + h/2, v)
        w2 = V2[:, 0:3]
        v2 = V2[:, 3:6]

        x3 = deepcopy(x1).to(device)
        x3[:,0:3,0:3] = exp_so3(bracket_so3(log_SO3(x3[:, :3, :3])) + 1/2 * h * w2)
        x3[:,0:3,3] += 1/2 * h * v2
        V3 = func(x3, t_i + h/2, v)
        w3 = V3[:, 0:3]
        v3 = V3[:, 3:6]

        x4 = deepcopy(x1).to(device)
        x4[:,0:3,0:3] = exp_so3(bracket_so3(log_SO3(x4[:, :3, :3])) + h * w3)
        x4[:,0:3,3] += h * v3
        V4 = func(x4, t_i + h, v)
        w4 = V4[:, 0:3]
        v4 = V4[:, 3:6]

        x_next = deepcopy(x1).to(device)
        x_next[:,0:3,0:3] = exp_so3(bracket_so3(log_SO3(x_next[:, :3, :3])) + 1/6 * h * (w1 + 2 * w2 + 2 * w3 + w4))
        x_next[:,0:3,3] += 1/6 * h * (v1 + 2 * v2 + 2 * v3 + v4)
        result[i+1] = x_next
    return result