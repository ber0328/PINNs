# --- prvotni import
from re import U
from sympy import plot
import torch
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import sys
from numpy import pi
sys.path.append('../..')

from src import train, utils
from src import calculus as calc
import src.data.cube_domain as cb
import src.models.mlp_model as mm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_BOUNDS = [-1.0, -1.0]
U_BOUNDS = [1.0, 1.0]


# --- define int residual & loss


def res(model: torch.nn.Module, pde_in: torch.Tensor):
    pde_out = model(pde_in)
    u, v_1, v_2 = pde_out[:, 0], pde_out[:, 1], pde_out[:, 2]
    grad_u = grad(u, pde_in, torch.ones_like(u), create_graph=True)[0]
    
    res_1 = grad_u + pde_out[:, 1:3]
    
    v_1_x = grad(v_1, pde_in, torch.ones_like(v_1), create_graph=True)[0][:, 0]
    v_2_y = grad(v_2, pde_in, torch.ones_like(v_2), create_graph=True)[0][:, 1]

    res_2 = v_1_x + v_2_y
    
    return res_1, res_2


def loss(model: torch.nn.Module, domain: cb.CubeDomain):
    pde_in = domain.interior.requires_grad_(True)
    res_1, res_2 = res(model, pde_in)
    
    loss_1 = torch.mean(res_1[:, 0]**2 + res_1[:, 1]**2)
    loss_2 = torch.mean(res_2**2)
    
    return [loss_1, loss_2]


# --- define network hard boundary enforcement


def g_D(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x[:, 0] + 1)


def u_0(x: torch.Tensor) -> torch.Tensor:
    return 1 - x[:, 0]**2


def v_0(x: torch.Tensor) -> torch.Tensor:
    return 1 - x[:, 1]**2


def h(x: torch.Tensor) -> torch.Tensor:
    return -x[:, 1]


def dec(model_in: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
    u_tilde = g_D(model_in) + u_0(model_in) * model_out[:, 0]
    v_1_tilde = model_out[:, 1]
    v_2_tilde = h(model_in) + v_0(model_in) * model_out[:, 2] 

    return torch.cat([u_tilde.unsqueeze(1),
                      v_1_tilde.unsqueeze(1),
                      v_2_tilde.unsqueeze(1)], dim=1)


# --- define network & optim & sched


model_ctx = mm.ModelContext(
    input_dim=2,
    output_dim=3,
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    layer=[64, 64, 64],
    hard_enforce_boundary=True,
    decorator=dec
)

model = mm.MLPModel(model_ctx).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optim, factor=0.5, patience=2_000)


# --- define domain


domain_ctx = cb.CubeContext(
    l_bounds = L_BOUNDS,
    u_bounds=U_BOUNDS,
    dim=2,
    N_int=5_000,
    N_sides=[(0, 0), (0, 0)],
    device=device
)

domain = cb.CubeDomain(domain_ctx)


# --- train


train_ctx = train.TrainingContext(
    model=model,
    domain=domain,
    optimizer=optim,
    loss_fn=loss,
    epochs=20_000,
)

total_losses, component_losses = train.simple_train(train_ctx)


# --- show results


plot_ctx = utils.PlotContext(
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    function_names=['u'],
    titles=['Model prediction'],
    N = 100,
    device=device,
    patches=[],
    colour_map='inferno',
    vmin=0.0,
    vmax=1.0,
)

utils.plot_function_on_2d_cube([lambda x: model(x)[:, 0]], plot_ctx)

plot_ctx.vmin = -9.0
plot_ctx.vmax = 24.0

def top(x: torch.Tensor):
    x.requires_grad_(True)
    out = model(x)
    return calc.dir_derivative(x, out, torch.tensor([0, -1], dtype=torch.float32, device=device), False, True)    

utils.plot_function_on_2d_cube([lambda x: top(x)], plot_ctx)

utils.plot_function_on_2d_cube([lambda x: model(x)[:, 1]], plot_ctx)

utils.plot_function_on_2d_cube([lambda x: torch.abs(res(model, x.requires_grad_(True))[0])], plot_ctx)
utils.plot_function_on_2d_cube([lambda x: torch.abs(res(model, x.requires_grad_(True))[1])], plot_ctx)
