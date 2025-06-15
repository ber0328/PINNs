import numpy as np
from torch import no_grad
from torch.nn import Module
from src.data.square_domain import SquareDomain2D
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def graph_prediction(model: Module, domain: SquareDomain2D, N=10000):
    test_pts = domain.gen_rand_int(N).to(model.device)
    test_x = test_pts[:, 0:1]
    test_t = test_pts[:, 1:2]

    with no_grad():
        u_pred = model(test_pts)

    _, axes = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

    x = test_x.detach().cpu().numpy().squeeze()
    t = test_t.detach().cpu().numpy().squeeze()
    z = u_pred.detach().cpu().numpy().squeeze()

    grid_t = np.linspace(domain.t_0, domain.t_1, N // 20)
    grid_x = np.linspace(domain.x_0, domain.x_1, N // 20)
    grid_t, grid_x = np.meshgrid(grid_t, grid_x)
    grid_z = griddata((t, x), z, (grid_t, grid_x), method="cubic")

    ax = axes
    ax.contourf(grid_t, grid_x, grid_z, levels=100, cmap="jet", vmin=None, vmax=None)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
