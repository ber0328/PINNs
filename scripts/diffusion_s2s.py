import torch
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
from numpy import pi
from typing import Tuple, Dict, List
sys.path.append('..')
from src import train, utils
from src import calculus as calc
import src.data.cube_domain as cb
from src.models.mlp_model import MLPModel


# definice device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTimePair:
    def __init__(self, models: List, time_stamps: List):
        self.models = models
        self.time_stamps = time_stamps

    def get_model_time_pair(self, i: int):
        return self.models[i], self.time_stamps[i]


# konstanty (parametry problemu)
ALPHA = 1.14
A = 2
B = 2
T_MAX = 1
l_bounds = [0, 0, 0]
u_bounds = [A, B, T_MAX]


# dulezite funkce
def initial_condition(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(pi * x[:, 0:1] / A) * torch.sin(pi * x[:, 1:2] / B) - 3 * torch.sin(5 * pi * x[:, 0:1] / A) * torch.sin(5 * pi * x[:, 1:2] / B)


def analytical_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(pi * x[:, 0:1] / A) * torch.sin(pi * x[:, 1:2] / B) * torch.exp(-ALPHA * pi**2 * x[:, 2:3] * (A**2 + B**2) / ((A * B)**2)) +\
           (-3) * torch.sin(5 * pi * x[:, 0:1] / A) * torch.sin(5 * pi * x[:, 1:2] / B) * torch.exp(-ALPHA * 25 * pi**2 * x[:, 2:3] * (A**2 + B**2) / ((A * B)**2))


# moznost dale rozvijet tento zpusob sbirani loss?
pde_loss_values = []
initial_loss_values = []
boundary_loss_values = []
iteration = 0
first_iteration = True
temp_init_condition = lambda x: initial_condition(x)


def pde_residuum(pde_input: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    # PDE ztrata:
    pde_output = model(pde_input)
    u_t = grad(pde_output, pde_input, torch.ones_like(pde_output), create_graph=True)[0][:, -1:]
    laplacian = calc.laplacian(pde_input, pde_output, device=device)
    return u_t - ALPHA * laplacian


def loss_fn(model: torch.nn.Module, domain: cb.CubeDomain) -> torch.Tensor:
    # PDE ztrata:
    pde_input = domain.interior.requires_grad_(True)
    pde_res = pde_residuum(pde_input, model)
    pde_loss = torch.mean((pde_res)**2)

    # ztrata na hranicich:
    side_input = domain.get_side_points(2).requires_grad_(True)
    side_output = model(side_input)
    side_loss = torch.mean(side_output**2)

    # ztrata na pocatku
    initial_input = domain.sides[-1][0].requires_grad_(True)
    init_output = model(initial_input)
    init_values = temp_init_condition(initial_input)
    init_loss = torch.mean((init_output - init_values)**2)

    # valid???
    global iteration
    if iteration % 100 == 99:
        pde_loss_values.append(pde_loss.item())
        boundary_loss_values.append(side_loss.item())
        initial_loss_values.append(init_loss.item())

    iteration += 1
    return [pde_loss, side_loss, 500 * init_loss]


# definice domeny
domain_ctx = cb.CubeContext(
    l_bounds=l_bounds,
    u_bounds=u_bounds,
    dim=3,
    N_int=2_000,
    N_sides=[(100, 100), (100, 100), (1_000, 100)],
    device=device
)

domain = cb.CubeDomain(domain_ctx)


train_ctx = train.TrainingContext(
    domain=domain,
    epochs=5_000,
    loss_fn=loss_fn,
    monitor_lr=True,
    monitor_gradient=True
)

loss_plot_ctx = utils.PlotContext(
    x_label='Epochs',
    y_label='Loss',
)


def train_models_for_division(models, division):
    loss_values = []

    for i in range(5):
        train_ctx.model = models[i]
        train_ctx.optimizer = torch.optim.Adam(train_ctx.model.parameters(), lr=4e-3)
        train_ctx.scheduler = ReduceLROnPlateau(train_ctx.optimizer, factor=0.75, patience=200)

        domain.ctx.l_bounds[-1] = division[i]
        domain.ctx.u_bounds[-1] = division[i+1]

        loss_values += train.train_switch_to_lbfgs(train_ctx, lbfgs_lr=0.1)
        prev_model = models[i]
        temp_init_condition = lambda x: prev_model(x)

    return loss_values


divisions = [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
             [0.0, 0.05, 0.1, 0.15, 0.2, 1.0],
             [0.0, 0.01, 0.02, 0.04, 0.1, 1.0],
             [0.0, 0.125, 0.25, 0.375, 0.5, 1.0]]


# provedeni experimentu pro kazde deleni intervalu
for i, division in enumerate(divisions):
    models = [MLPModel(3, 1, [64, 64, 128], l_bounds, u_bounds).to(device),
              MLPModel(3, 1, [64, 64, 64], l_bounds, u_bounds).to(device),
              MLPModel(3, 1, [64, 64, 64], l_bounds, u_bounds).to(device),
              MLPModel(3, 1, [64, 64, 64], l_bounds, u_bounds).to(device),
              MLPModel(3, 1, [64, 64, 64], l_bounds, u_bounds).to(device)]
    model_time = ModelTimePair(models, division)

    loss_values = train_models_for_division(models, division)

    loss_plot_ctx.save_path = f'images/total_loss_test_{i}.png'
    loss_plot_ctx.title = f'Total loss | test {i}'
    utils.plot_loss_values({'Total loss': loss_values}, loss_plot_ctx)

    loss_plot_ctx.save_path = f'images/loss_by_parts_{i}.png'
    loss_plot_ctx.title = f'Loss by parts | test {i}'
    utils.plot_loss_values({'PDE loss': pde_loss_values, 'Init loss': initial_loss_values, 'Boundary loss': boundary_loss_values}, loss_plot_ctx)

    with open('l2_results.txt', 'a') as f:
        total_l2 = 0
        for j in range(5):
            model_l2 = calc.L2_norm(analytical_solution, models[j], 3, [2, 2, division[j+1]], [0, 0, division[j]], device, 50)
            f.write(f'Test {i} - model {j} l2 error: {model_l2}\n')
            total_l2 += model_l2**2

        f.write(f'Test {i} total l2 error: {torch.sqrt(total_l2)}\n\n')


# plot_ctx = utils.PlotContext(
#     l_bounds=l_bounds,
#     u_bounds=u_bounds,
#     device=device,
#     patches=[],
#     colour_map='inferno',
#     vmin=0,
#     vmax=4,
#     N=100,
# )

# time_stamps = [0.01, 0.4, 0.6, 0.8, 1.0]

# for i, model in enumerate(model_time_dict.keys()):
#     plot_ctx.title = f'Model {i + 1} prediction at t={time_stamps[i]}'
#     utils.plot_function_on_2d_cube(plot_ctx=plot_ctx, function=lambda x: 
#                                    model(torch.cat([x, torch.full((x.shape[0], 1), time_stamps[i], device=device)], dim=1)))

#     plot_ctx.title = f'Actual temperature at t={time_stamps[i]}'
#     utils.plot_function_on_2d_cube(plot_ctx=plot_ctx,  function=lambda x: 
#                                    analytical_solution(torch.cat([x, torch.full((x.shape[0], 1), time_stamps[i], device=device)], dim=1)))
