"""
    Modul pro praci s gradienty, derivacemi a podobnymi...
    TODO: zobecnit pro n-dimenzionalni prostory. (pokud to bude potreba)
"""

import torch
from torch.autograd import grad


def compute_derivatives_2d(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x.requires_grad_(True)
    u = model.forward(x)

    grads = grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    return u, u_x, u_y


def compute_2nd_derivatives_2d(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x.requires_grad_(True)

    u, u_x, u_y = compute_derivatives_2d(model, x)

    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    return u, u_x, u_y, u_xx, u_yy


def laplacian_2d(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    _, _, _, u_xx, u_yy = compute_2nd_derivatives_2d(model, x)

    return u_xx + u_yy
