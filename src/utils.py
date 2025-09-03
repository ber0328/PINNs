from torch import no_grad, Tensor
from src.data.abstract_domain import AbstractDomain
from src.data.cube_domain import CubeContext
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict
from scipy.interpolate import griddata
from dataclasses import dataclass
import torch
from copy import copy
Function = Callable[[Tensor], Tensor]


@dataclass
class PlotContext():
    l_bounds: List = None
    u_bounds: List = None
    function_names: List[str] = None
    x_label: str = 'x'
    y_label: str = 'y'
    titles: List[str] = None
    N: int = 1000
    colour_map: str = 'jet'
    device: str = 'cpu'
    patches: List = None
    figsize: Tuple = (6, 4)
    fontsize: int = 12
    vmin: float = 0
    vmax: float = 5
    save_img: bool = False
    save_path: str = 'plot.png'


def plot_points(pts_list: Dict[str, torch.Tensor], title="") -> None:
    for label, pts in pts_list.items():
        pts = pts.cpu().detach().numpy()

        points_x = pts[:, 0]
        points_y = pts[:, 1]

        plt.scatter(points_x, points_y, label=label, s=1)

    plt.title(title)
    plt.legend()
    plt.show()


def plot_loss_values(loss_values: Dict[str, List[float]], plot_ctx: PlotContext) -> None:
    _, ax = plt.subplots(figsize=(6, 4))

    for label, values in loss_values.items():
        n = range(len(values))
        ax.plot(n, values, label=label)

    ax.set_xlabel(xlabel=plot_ctx.x_label)
    ax.set_ylabel(ylabel=plot_ctx.y_label)
    ax.set_yscale('log')

    x_ticks = ax.get_xticks()
    new_x_labels = [str(tick * 100) for tick in x_ticks]
    ax.set_xticklabels(new_x_labels)
    ax.set_title(plot_ctx.titles[0])
    ax.legend()

    if plot_ctx.save_img:
        plt.savefig(plot_ctx.save_path)
    else:
        plt.show()


def plot_vector_field_2d(functions: List[Function], plot_ctx: PlotContext, N: int = 10):
    pts_x = torch.linspace(plot_ctx.l_bounds[0], plot_ctx.u_bounds[0], N, device=plot_ctx.device)
    pts_y = torch.linspace(plot_ctx.l_bounds[1], plot_ctx.u_bounds[1], N, device=plot_ctx.device)

    X, Y = torch.meshgrid(pts_x, pts_y)
    X, Y = X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)
    inputs = torch.cat((X, Y), dim=1)
    directions = []
    norms = []

    for function in functions:
        v = function(inputs)
        v_x, v_y = v[:, 0:1], v[:, 1:2]
        norm = torch.sqrt(v_x**2 + v_y**2)
        norms.append(norm.cpu().detach().numpy().reshape((N, N)))
        v_x_unit, v_y_unit = torch.div(v_x, norm), torch.div(v_y, norm)
        v_x_unit = v_x_unit.cpu().detach().numpy().reshape((N, N))
        v_y_unit = v_y_unit.cpu().detach().numpy().reshape((N, N))
        directions.append((v_x_unit, v_y_unit))

    X = X.cpu().detach().numpy().reshape((N, N))
    Y = Y.cpu().detach().numpy().reshape((N, N))

    fig, ax = plt.subplots(1, len(functions), figsize=plot_ctx.figsize)

    if len(functions) == 1:
        ax = [ax]

    for i in range(len(functions)):
        contour = ax[i].contourf(X, Y, norms[i], levels=100, cmap=plot_ctx.colour_map,
                                 vmin=plot_ctx.vmin, vmax=plot_ctx.vmax)
        fig.colorbar(contour, ax=ax[i], label=plot_ctx.function_names[i])
        ax[i].set_xlabel(plot_ctx.x_label, fontsize=plot_ctx.fontsize)
        ax[i].set_ylabel(plot_ctx.y_label, fontsize=plot_ctx.fontsize)
        ax[i].set_title(plot_ctx.titles[i], fontsize=plot_ctx.fontsize)
        v_x_unit, v_y_unit = directions[i]
        ax[i].quiver(X, Y, v_x_unit, v_y_unit, color='w')

        for patch in plot_ctx.patches:
            ax[i].add_patch(patch)

    if plot_ctx.save_img:
        plt.savefig(plot_ctx.save_path)
    else:
        plt.show()


def plot_function_on_2d_cube(functions: List[Function], plot_ctx: PlotContext):
    pts_x = torch.linspace(plot_ctx.l_bounds[0], plot_ctx.u_bounds[0], plot_ctx.N, device=plot_ctx.device)
    pts_y = torch.linspace(plot_ctx.l_bounds[1], plot_ctx.u_bounds[1], plot_ctx.N, device=plot_ctx.device)

    X, Y = torch.meshgrid(pts_x, pts_y)
    X, Y = X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)
    inputs = torch.cat((X, Y), dim=1)
    Z_vals = []

    for function in functions:
        Z_vals.append(function(inputs).cpu().detach().numpy().reshape((plot_ctx.N, plot_ctx.N)))

    X = X.cpu().detach().numpy().reshape((plot_ctx.N, plot_ctx.N))
    Y = Y.cpu().detach().numpy().reshape((plot_ctx.N, plot_ctx.N))

    fig, ax = plt.subplots(1, len(functions), figsize=plot_ctx.figsize)

    if len(functions) == 1:
        ax = [ax]

    for i, Z in enumerate(Z_vals):
        contour = ax[i].contourf(X, Y, Z, levels=100, cmap=plot_ctx.colour_map, vmin=plot_ctx.vmin, vmax=plot_ctx.vmax)
        fig.colorbar(contour, ax=ax[i], label=plot_ctx.function_names[i])
        ax[i].set_xlabel(plot_ctx.x_label)
        ax[i].set_ylabel(plot_ctx.y_label)
        ax[i].set_title(plot_ctx.titles[i], fontsize=plot_ctx.fontsize)

    for i in range(len(functions)):
        for patch in plot_ctx.patches:
            print((plot_ctx.patches))
            ax[i].add_patch(patch)

    if plot_ctx.save_img:
        plt.savefig(plot_ctx.save_path)
    else:
        plt.show()
