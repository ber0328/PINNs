# prvotni import

from typing import List
import torch
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
from numpy import pi
import matplotlib.pyplot as plt
sys.path.append('..')

# %%
# vlastni import
from src import train, utils
from src import calculus as calc
import src.data.cube_domain as cb
import src.models.mlp_model as mm

# %%
# volba device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# definice ulohy
ALPHA = 114e-4  # diffusivity constant
A = 100
C = 5
T_0 = 10.0
l_bounds = [-1.0, -1.0, 0.0]  # lower bounds of the cube domain
u_bounds = [1.0, 1.0, T_0]

def heat_in(x: torch.Tensor) -> torch.Tensor:
    return C * torch.exp(-A * (x[:, 0:1]**2 + x[:, 1:2]**2))

def heat_loss(x: torch.Tensor) -> torch.Tensor:
    return 0

def init_temp(x: torch.Tensor) -> torch.Tensor:
    return 0

# %%
# definice ztratovych funkci

def compute_residuum(input: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    output = model(input)
    laplacian = calc.laplacian(input, output, device=device)
    u_t = grad(output, input, grad_outputs=torch.ones_like(output), create_graph=True)[0][:, -1:]
    h_vals = heat_in(input)
    return u_t - ALPHA * laplacian - h_vals


def loss_interior(input: torch.Tensor, model: torch.nn.Module):
    residuum = compute_residuum(input, model)
    return torch.mean(residuum ** 2)


def neumann_bounday(input: torch.Tensor, model: torch.nn.Module, normal: torch.Tensor):
    output = model(input)
    u_n = calc.dir_derivative(input, output, normal, True, True)
    return torch.mean(u_n ** 2)


def loss_init(input: torch.Tensor, model: torch.nn.Module):
    output = model(input)
    init_vals = init_temp(input)
    return torch.mean((output - init_vals) ** 2)

left_condition = torch.tensor([])
first_iteration = True

def loss_fn(model: torch.nn.Module, domain: cb.CubeDomain):
    # interior loss
    interior_input = domain.interior.requires_grad_(True)
    loss_int = loss_interior(interior_input, model)

    # pocatecni ztrata
    if first_iteration:
        init_input = domain.sides[-1][0].requires_grad_(True)
        loss_init_val = loss_init(init_input, model)
    else:
        output_init = model(domain.sides[-1][0])
        loss_init_val = torch.mean((output_init - left_condition) ** 2)

    # neumann ztrata
    # leva hranice
    input_left = domain.sides[0][0].requires_grad_(True)
    loss_neumann_left = neumann_bounday(input_left, model, normal=torch.tensor([-1.0, 0.0],device=device,requires_grad=True))
    # prava hranice
    input_right = domain.sides[0][1].requires_grad_(True)
    loss_neumann_right = neumann_bounday(input_right, model, normal=torch.tensor([1.0, 0.0],device=device,requires_grad=True))
    # horni hranice
    input_top = domain.sides[1][0].requires_grad_(True)
    loss_neumann_top = neumann_bounday(input_top, model, normal=torch.tensor([0.0, 1.0],device=device,requires_grad=True))
    # dolni hranice
    input_bottom = domain.sides[1][1].requires_grad_(True)
    loss_neumann_bottom = neumann_bounday(input_bottom, model, normal=torch.tensor([0.0, -1.0],device=device,requires_grad=True))

    side_loss = loss_neumann_left + loss_neumann_right + loss_neumann_top + loss_neumann_bottom

    return [loss_int, loss_init_val, side_loss]

# %%
model_naive_ctx = mm.ModelContext(
    input_dim=3,
    output_dim=1,
    layer=[64, 64, 64, 64],
    l_bounds=l_bounds,
    u_bounds=u_bounds,
    last_layer_activation='tanh',
    fourier_features=True,
    fourier_frequencies=10,
    fourier_scale=1.0
)

model_naive = mm.MLPModel(model_naive_ctx).to(device)

model_curriculum = mm.MLPModel(model_naive_ctx).to(device)

optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=5e-5)
optimizer_curriculum = torch.optim.Adam(model_curriculum.parameters(), lr=5e-3)

scheduler_naive = ReduceLROnPlateau(optimizer_naive, mode='min', factor=0.75, patience=200, verbose=True)
scheduler_curriculum = ReduceLROnPlateau(optimizer_curriculum, mode='min', factor=0.5, patience=100, verbose=True)

# %%
cube_ctx = cb.CubeContext(
    l_bounds=l_bounds,
    u_bounds=u_bounds,
    N_int=20_000,
    N_sides=[[100, 100], [100, 100], [100, 100]],
    dim=3,
    device=device,
    int_sampling='Biased',
    bias_pts=[(torch.tensor([0, 0], device=device), 1000, 0.4)]
)

domain = cb.CubeDomain(cube_ctx)

# %%
# Prvne naivni zpusob
train_naive_ctx = train.TrainingContext(
    model=model_naive,
    domain=domain,
    loss_fn=loss_fn,
    optimizer=optimizer_naive,
    epochs=60_000,
    resample_freq=2000,
    resample=True
)

domain.generate_points()
naive_loss_vals = train.train_switch_to_lbfgs(train_naive_ctx, lbfgs_lr=0.1)

# %%
# Dale zkusime curriculum strategii

tratin_curriculum_ctx = train.TrainingContext(
    model=model_curriculum,
    domain=domain,
    loss_fn=loss_fn,
    optimizer=optimizer_curriculum,
    epochs=1_500,
    resample_freq=1_000,
)


# prvni iterace
domain.ctx.l_bounds = [-1.0, -1.0, 0.0]
domain.ctx.u_bounds = [1.0, 1.0, 1.0]
first_iteration = True
curriculum_loss_vals = train.train_switch_to_lbfgs(tratin_curriculum_ctx, lbfgs_lr=0.1, epochs_with_lbfgs=500)

DELTA_TS = [2.0, 3.0, 4.0, 5.0]
for delta_t in DELTA_TS:
    # posunuti domeny
    print(f"Training in interval: [0, {float(delta_t)}")
    domain.ctx.u_bounds = [1.0, 1.0, delta_t]
    # nastaveni nove pocatecni podminky

    # trenovani v novem casovem useku
    curriculum_loss_vals += train.train_switch_to_lbfgs(tratin_curriculum_ctx, lbfgs_lr=0.1, epochs_with_lbfgs=500)

# %%
plot_ctx = utils.PlotContext(
    l_bounds=l_bounds,
    u_bounds=u_bounds,
    device=device,
    patches=[],
    colour_map='inferno',
    vmin=0,
    vmax=4,
    N=100,
    x_label="Epochs",
    y_label="Loss",
)

utils.plot_loss_values({'curriculum loss': curriculum_loss_vals, 'Naive loss': naive_loss_vals}, plot_ctx)
utils.plot_loss_values({}, plot_ctx)

# %%
# vykresleni vysledku

TIME = 0

def s2s_u_t(input: torch.Tensor) -> torch.Tensor:
    time = torch.full_like(input[:, 0:1], TIME, device=device)
    return model_curriculum(torch.cat((input, time), dim=1))

def naive_u_t(input: torch.Tensor) -> torch.Tensor:
    time = torch.full_like(input[:, 0:1], TIME, device=device)
    return model_naive(torch.cat((input, time), dim=1))

plot_ctx.function_names = ['u']

for TIME in [0.0, 1, 2, 3, 4, 5]:
    plot_ctx.titles = [f"Curriculum temperature at t={TIME} s"]
    utils.plot_function_on_2d_cube([s2s_u_t], plot_ctx)

for TIME in [0.0, 1, 2, 3, 4, 5]:
    plot_ctx.title = f"Naive temperature at t={TIME} s"
    utils.plot_function_on_2d_cube([naive_u_t], plot_ctx)
