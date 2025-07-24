from torch import no_grad, Tensor
from src.data.abstract_domain import AbstractDomain
from src.data.cube_domain import CubeContext
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from scipy.interpolate import griddata
import numpy as np
from dataclasses import dataclass
import torch

Function = Callable[[Tensor], Tensor]


@dataclass
class PlotContext():
    l_bounds: List = None
    u_bounds: List = None
    function_name: str = 'u'
    x_label: str = 'x'
    y_label: str = 'y'
    title: str = 'Graph'
    N: int = 1000
    colour_map: str = 'jet'
    device: str = 'cpu'
    patches: List = None
    figsize: Tuple = (6, 4)
    vmin: float = 0
    vmax: float = 5


def plot_function_on_domain(ctx: PlotContext) -> None:
    ctx.domain.generate_points()
    points = ctx.domain.get_all_points()

    points_x = points[:, 0].cpu().detach().numpy()
    points_y = points[:, 1].cpu().detach().numpy()

    x_lin = np.linspace(min(points_x), max(points_x), ctx.N)
    y_lin = np.linspace(min(points_y), max(points_y), ctx.N)

    values = ctx.function(points).cpu().detach().numpy().squeeze()

    X, Y = np.meshgrid(x_lin, y_lin)
    Z = griddata((points_x, points_y), values, (X, Y), method='cubic')

    fig, ax = plt.subplots(figsize=ctx.figsize)
    contour = ax.contourf(X, Y, Z, levels=100, cmap='jet')
    fig.colorbar(contour, ax=ax, label=ctx.function_name)
    ax.set_xlabel(ctx.x_label)
    ax.set_ylabel(ctx.y_label)
    ax.set_title(ctx.title)

    for patch in ctx.patches:
        ax.add_patch(patch)

    plt.show()


def plot_domain(domain: AbstractDomain, time: int = 0) -> None:
    points = domain.get_all_points()

    points = points.cpu().detach().numpy()

    points_x = points[:, 0]
    points_y = points[:, 1]

    plt.scatter(points_x, points_y)
    plt.show()


def plot_loss_values(values: List[float], x_label: str, y_label: str,
                     title: str = "Loss values") -> None:
    n = range(len(values))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n, values)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_yscale('log')

    x_ticks = ax.get_xticks()
    new_x_labels = [str(tick * 100) for tick in x_ticks]
    ax.set_xticklabels(new_x_labels)
    ax.set_title(title)
    plt.show()


def plot_vector_field_2d(function: Function, plot_ctx: PlotContext):
    N = 10
    pts_x = torch.linspace(plot_ctx.l_bounds[0], plot_ctx.u_bounds[0], N, device=plot_ctx.device)
    pts_y = torch.linspace(plot_ctx.l_bounds[1], plot_ctx.u_bounds[1], N, device=plot_ctx.device)

    X, Y = torch.meshgrid(pts_x, pts_y)
    X, Y = X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)
    inputs = torch.cat((X, Y), dim=1)

    with torch.no_grad():
        Z1, Z2 = function(inputs)
        Z1 = Z1.cpu().detach().numpy().reshape((N, N))
        Z2 = Z2.cpu().detach().numpy().reshape((N, N))

    X = X.cpu().detach().numpy().reshape((N, N))
    Y = Y.cpu().detach().numpy().reshape((N, N))

    plt.quiver(X, Y, Z1, Z2)
    plt.show()


def plot_function_on_2d_cube(function: Function, plot_ctx: PlotContext):
    pts_x = torch.linspace(plot_ctx.l_bounds[0], plot_ctx.u_bounds[0], plot_ctx.N, device=plot_ctx.device)
    pts_y = torch.linspace(plot_ctx.l_bounds[1], plot_ctx.u_bounds[1], plot_ctx.N, device=plot_ctx.device)

    X, Y = torch.meshgrid(pts_x, pts_y)
    X, Y = X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)
    inputs = torch.cat((X, Y), dim=1)

    with torch.no_grad():
        Z = function(inputs).cpu().detach().numpy().reshape((plot_ctx.N, plot_ctx.N))

    X = X.cpu().detach().numpy().reshape((plot_ctx.N, plot_ctx.N))
    Y = Y.cpu().detach().numpy().reshape((plot_ctx.N, plot_ctx.N))

    fig, ax = plt.subplots(figsize=plot_ctx.figsize)
    contour = ax.contourf(X, Y, Z, levels=100, cmap=plot_ctx.colour_map, vmin=plot_ctx.vmin, vmax=plot_ctx.vmax)
    fig.colorbar(contour, ax=ax, label=plot_ctx.function_name)
    ax.set_xlabel(plot_ctx.x_label)
    ax.set_ylabel(plot_ctx.y_label)
    ax.set_title(plot_ctx.title)

    for patch in plot_ctx.patches:
        ax.add_patch(patch)

    plt.show()
