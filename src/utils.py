from torch import no_grad, Tensor
from src.data.abstract_domain import AbstractDomain
import matplotlib.pyplot as plt
from typing import Callable, List
from scipy.interpolate import griddata
import numpy as np


Function = Callable[[Tensor], Tensor]


# TODO: dataclass?
def plot_function_on_domain(function: Function, domain: AbstractDomain,
                            function_name='u', x_label='x', y_lable='y',
                            title='Graph', N=1000) -> None:
    domain.generate_points(N)
    points = domain.get_all_points()

    points_x = points[:, 0].cpu().detach().numpy()
    points_y = points[:, 1].cpu().detach().numpy()

    x_lin = np.linspace(min(points_x), max(points_x), N)
    y_lin = np.linspace(min(points_y), max(points_y), N)

    with no_grad():
        values = function(points).cpu().detach().numpy().squeeze()

    X, Y = np.meshgrid(x_lin, y_lin)
    Z = griddata((points_x, points_y), values, (X, Y), method='cubic')

    fig, ax = plt.subplots()
    contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
    fig.colorbar(contour, ax=ax, label=function_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_lable)
    ax.set_title(title)
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
