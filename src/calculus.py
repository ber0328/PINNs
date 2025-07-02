"""
    Modul pro praci s gradienty, derivacemi a podobnymi...
    TODO: zobecnit pro n-dimenzionalni prostory. (pokud to bude potreba)
"""

import torch
from torch.autograd import grad


def compute_derivatives_2d(model: torch.nn.Module, x: torch.Tensor, i: int = 0) -> torch.Tensor:
    x.requires_grad_(True)
    u_i = model(x)[:, i:i+1]

    grads = grad(u_i, x, torch.ones_like(u_i), create_graph=True)[0]
    u_i_x = grads[:, 0:1]
    u_i_y = grads[:, 1:2]

    return u_i, u_i_x, u_i_y


def compute_2nd_derivatives_2d(model: torch.nn.Module, x: torch.Tensor, i: int = 0) -> torch.Tensor:
    x.requires_grad_(True)

    u_i, u_i_x, u_i_y = compute_derivatives_2d(model, x, i)

    u_i_xx = grad(u_i_x, x, torch.ones_like(u_i_x), create_graph=True)[0][:, 0:1]
    u_i_yy = grad(u_i_y, x, torch.ones_like(u_i_y), create_graph=True)[0][:, 1:2]

    return u_i, u_i_x, u_i_y, u_i_xx, u_i_yy


def laplacian_2d(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    u, u_x, u_y, u_xx, u_yy = compute_2nd_derivatives_2d(model, x)

    return u, u_x, u_y, u_xx, u_yy, u_xx + u_yy


def compute_derivatives_ns(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x.requires_grad_(True)
    out = model(x)
    psi, p = out[:, 0], out[:, 1]

    psi_grad = grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    v, u = -psi_grad[:, 0:1], psi_grad[:, 1:2]
    u_grad = grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
    v_grad = grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_x, v_y = v_grad[:, 0:1], v_grad[:, 1:2]
    p_grad = grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]

    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

    return psi, p, u, v, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y


