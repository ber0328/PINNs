"""
    Modul pro praci s gradienty, derivacemi a podobnymi...
    TODO: zobecnit pro n-dimenzionalni prostory. (pokud to bude potreba)
"""

import torch
from torch.autograd import grad
import numpy as np
from typing import List
from torch._dynamo import disable


def compute_derivatives_2d(model: torch.nn.Module, x: torch.Tensor, i: int = 0) -> torch.Tensor:
    x.requires_grad_(True)
    u_i = model(x)[:, i]

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


def L2_norm(f, g, input_dim: int, u_bounds: List, l_bounds: List,
            device: str = 'cpu', n: int = 500) -> torch.Tensor:
    weight_list = []
    value_list = []

    for i in range(input_dim):
        vals, weights = np.polynomial.legendre.leggauss(n)
        vals = torch.from_numpy(vals).to(device).float()
        weights = torch.from_numpy(weights).to(device).float()

        vals = 0.5 * (vals + 1) * (u_bounds[i] - l_bounds[i]) + l_bounds[i]
        weights = 0.5 * (u_bounds[i] - l_bounds[i]) * weights

        value_list.append(vals)
        weight_list.append(weights)

    # indexing je 'ij', bo to jinak rve
    VAR_LIST = list(torch.meshgrid(*value_list, indexing='ij'))
    WEIGHT_LIST = list(torch.meshgrid(*weight_list, indexing='ij'))

    WEIGHT_LIST = [weight.flatten().unsqueeze(1) for weight in WEIGHT_LIST]
    VAR_LIST = [var.flatten().unsqueeze(1) for var in VAR_LIST]

    W = 1
    for weight in WEIGHT_LIST:
        W *= weight

    input = torch.cat(VAR_LIST, dim=1)

    f_vals, g_vals = f(input), g(input)
    integral = torch.sqrt(torch.sum(((f_vals - g_vals)**2) * W))

    return integral


def nabla(input: torch.Tensor, output: torch.Tensor, retain: bool = False):
    return grad(output, input, torch.ones_like(output), create_graph=True, retain_graph=retain)[0]


def div(input: torch.Tensor, output: torch.Tensor, retain: bool = False, device: str = 'cpu'):
    div = torch.zeros(output.shape[0], device=device)

    for i in range(output.shape[1]):
        output_i = grad(output[:, i], input, torch.ones_like(output[:, i]), retain_graph=retain)[0]
        div += output_i[:, i]

    return div.unsqueeze(1)


def dir_derivative(input: torch.Tensor, output: torch.Tensor, direction: torch.Tensor, time_dep: bool = True, retain: bool = False):
    gradient = grad(output, input, torch.ones_like(output), create_graph=True, retain_graph=retain)[0]
    if time_dep:
        gradient = gradient[:, :-1]
    return (gradient @ direction)


def material_derivative(inputs: torch.Tensor, outputs: torch.Tensor, device: str = 'cpu',
                        time_dependant: bool = True) -> torch.Tensor:
    mat_der = torch.zeros_like(outputs, device=device)

    for i in range(outputs.shape[1]):
        grad_ui = grad(outputs[:, i], inputs, torch.ones_like(outputs[:, i]), create_graph=True, retain_graph=True)[0]
        if time_dependant:
            mat_der[:, i] = grad_ui[:, -1] + torch.sum(outputs * grad_ui[:, :-1], dim=1)
        else:
            mat_der[:, i] = torch.sum(outputs * grad_ui, dim=1)

    return mat_der


def laplacian(input: torch.Tensor, output: torch.Tensor, device: str = 'cpu',
              time_dependant: bool = True) -> torch.Tensor:
    lap = torch.zeros_like(output, device=device)

    for i in range(output.shape[1]):
        grad_ui = grad(output[:, i], input, torch.ones_like(output[:, i]), create_graph=True, retain_graph=True)[0]

        lap_ui = torch.zeros(output.shape[0], device=device)

        ran = input.shape[1] - 1 if time_dependant else input.shape[1]
        for j in range(ran):
            grad_grad_ui = grad(grad_ui[:, j], input, torch.ones_like(grad_ui[:, j]), create_graph=True, retain_graph=True)[0]
            lap_ui += grad_grad_ui[:, j]

        lap[:, i] = lap_ui

    return lap
