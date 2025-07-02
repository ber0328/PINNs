from torch import no_grad, Tensor
from src.data.abstract_domain import AbstractDomain
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from scipy.interpolate import griddata
import numpy as np
from dataclasses import dataclass


Function = Callable[[Tensor], Tensor]


@dataclass
class PlotContext():
    function: Function = None
    domain: AbstractDomain = None
    function_name: str = 'u'
    x_label: str = 'x'
    y_label: str = 'y'
    title: str = 'Graph'
    N: int = 1000
    colour_map: str = 'jet'
    patches: List = None
    figsize: Tuple = (6, 4)


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


def plot_domain(domain: AbstractDomain) -> None:
    points = domain.get_all_points()

    points = points.cpu().detach().numpy()

    points_x = points[:, 0]
    points_y = points[:, 1]

    plt.scatter(points_x, points_y)
    plt.show()


def plot_loss_values(values: List[float], x_label: str, y_label: str) -> None:
    n = range(len(values))

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n, values)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    ax.set_yscale('log')

    x_ticks = ax.get_xticks()
    new_x_labels = [str(tick * 100) for tick in x_ticks]
    ax.set_xticklabels(new_x_labels)

    plt.show()
