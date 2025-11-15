# %%
# prvotni import

import torch
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import sys
from numpy import pi
sys.path.append('../..')

# %%
# vlastni import
from src import train, utils
from src import calculus as calc
import src.data.cube_domain as cb
import src.models.mlp_model as mm

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
ALPHA = 1.0
T_src = 3
L_BOUNDS = [-1.0, -1.0]
U_BOUNDS = [1.0, 1.0]

# %%

def pde_residuum(pde_input: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    # PDE ztrata:
    pde_output = model(pde_input)
    #print(u_t)
    laplacian = calc.laplacian(pde_input, pde_output, device=device)
    return T_src - ALPHA * laplacian

def loss_fn(model: torch.nn.Module, domain: cb.CubeDomain):
    int_pts = domain.interior.requires_grad_(True)
    res = pde_residuum(int_pts, model)
    int_loss = torch.mean((res)**2)
    
    left_pts = domain.sides[0][0].requires_grad_(True)
    right_pts = domain.sides[0][1].requires_grad_(True)

    left_out, right_out = model(left_pts), model(right_pts)
    left_loss = torch.mean(left_out**2)
    right_loss = torch.mean((right_out - 1)**2)
    
    top_pts = domain.sides[1][1]
    bot_pts = domain.sides[1][0]
    
    top_mask = (top_pts[:, 0] > -1/3) & (top_pts[:, 0] < 1/3)
    bot_mask = (bot_pts[:, 0] > -1/3) & (bot_pts[:, 0] < 1/3)
    
    top_pts = top_pts[top_mask].requires_grad_(True)
    bot_pts = bot_pts[bot_mask].requires_grad_(True)
    
    top_out, bot_out = model(top_pts), model(bot_pts)
    
    top_d_dn = calc.dir_derivative(top_pts, top_out, torch.tensor([0, 1], dtype=torch.float32, device=device), False, True)
    bot_d_dn = calc.dir_derivative(bot_pts, bot_out, torch.tensor([0, -1], dtype=torch.float32, device=device), False, True)

    top_loss = torch.mean((top_d_dn - 1)**2)
    bot_loss = torch.mean((bot_d_dn - 1)**2)

    return [int_loss, left_loss, right_loss, top_loss, bot_loss]

# %%
model_ctx = mm.ModelContext(
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    input_dim=2,
    output_dim=1,
    layer=[64, 64, 64]
)

model = mm.MLPModel(model_ctx).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, patience=200, factor=0.75)

# %%
domain_ctx = cb.CubeContext(
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    dim=2,
    N_int=5_000,
    N_sides=[(500, 500), (500, 500), (500, 500)],
    int_sampling='Latin',
    device=device,
)

domain = cb.CubeDomain(domain_ctx)

# %%
train_ctx = train.TrainingContext(
    model=model,
    optimizer=optimizer,
    domain=domain,
    loss_fn=loss_fn,
    epochs=20_000,
    monitor_lr=True,
)

total_loss_values, component_loss_values = train.train_switch_to_lbfgs(train_ctx, lbfgs_lr=0.1, epochs_with_lbfgs=500)

# %%
plot_ctx = utils.PlotContext(
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    patches=[],
    titles=["Model prediction"],
    function_names=["u"],
    colour_map='inferno',
    vmin=-1,
    vmax=1,
    device=device,
    N=100,
    save_img=True,
)

plot_ctx.save_path="./heat_hd_results/prediction_standard.png"

utils.plot_function_on_2d_cube([model], plot_ctx)


utils.plot_loss_values({"inner loss": component_loss_values[0],
                        "left-right loss": component_loss_values[1],
                        "top loss": component_loss_values[2],
                        "bot loss": component_loss_values[3]}, plot_ctx)
plot_ctx.save_path="./heat_hd_results/loss_comp_standard.png"


# %%
plot_ctx.vmin = 0
plot_ctx.vmax = 0.01
plot_ctx.titles = ["residuals"]
plot_ctx.save_path="./heat_hd_results/residuals_standard.png"


utils.plot_function_on_2d_cube([lambda x: torch.abs(pde_residuum(x.requires_grad_(True), model))], plot_ctx)

# %%
def loss_hard_enforce(model: torch.nn.Module, domain: cb.CubeDomain):
    int_pts = domain.interior.requires_grad_(True)
    res = pde_residuum(int_pts, model)
    int_loss = torch.mean(res**2)
    
    return [int_loss]


def f_loss(model: torch.nn.Module, domain: cb.CubeDomain):
    top_pts = domain.sides[1][1]
    bot_pts = domain.sides[1][0]
    
    top_mask = (top_pts[:, 0] > -1/3) & (top_pts[:, 0] < 1/3)
    bot_mask = (bot_pts[:, 0] > -1/3) & (bot_pts[:, 0] < 1/3)
    
    top_pts = top_pts[top_mask].requires_grad_(True)
    bot_pts = bot_pts[bot_mask].requires_grad_(True)
    
    top_out, bot_out = model(top_pts), model(bot_pts)
    dirichelt_loss_top = torch.mean(top_out**2)
    dirichelt_loss_bot = torch.mean(bot_out**2)
    
    grad_y_top = grad(top_out, top_pts, torch.ones_like(top_out), create_graph=True)[0][:, 1]
    grad_y_bot = grad(bot_out, bot_pts, torch.ones_like(bot_out), create_graph=True)[0][:, 1]

    neumann_loss_top = torch.mean((grad_y_bot - 1)**2)
    neumann_loss_bot = torch.mean((grad_y_top - 1)**2)
    
    return [dirichelt_loss_top, dirichelt_loss_bot, neumann_loss_top, neumann_loss_bot]

def q_0_loss(model: torch.nn.Module, domain: cb.CubeDomain):
    top_pts = domain.sides[1][1]
    bot_pts = domain.sides[1][0]
    left_pts = domain.sides[0][0].requires_grad_(True)
    right_pts = domain.sides[0][1].requires_grad_(True)
    
    top_mask = (top_pts[:, 0] > -1/3) & (top_pts[:, 0] < 1/3)
    bot_mask = (bot_pts[:, 0] > -1/3) & (bot_pts[:, 0] < 1/3)
    
    top_pts = top_pts[top_mask].requires_grad_(True)
    bot_pts = bot_pts[bot_mask].requires_grad_(True)
    
    top_out = model(top_pts)
    bot_out = model(bot_pts)
    left_out = model(left_pts)
    right_out = model(right_pts)
    
    top_loss = torch.mean((top_out - 1)**2)
    bot_loss = torch.mean((bot_out - 1)**2)
    left_loss = torch.mean(left_out**2)
    right_loss = torch.mean((right_out - 1)**2)
    
    return [top_loss, bot_loss, left_loss, right_loss]

# %%
domain_ctx.N_int = 1
model_ctx.layer = [32, 32, 32]

f = mm.MLPModel(model_ctx).to(device)
q_0 = mm.MLPModel(model_ctx).to(device)

f_optim = torch.optim.Adam(f.parameters(), lr=0.0001)
q_0_optim = torch.optim.Adam(q_0.parameters(), lr=0.0001)

train_ctx.model = f
train_ctx.optimizer = f_optim
train_ctx.loss_fn = f_loss

f_loss, _ = train.simple_train(train_ctx)

train_ctx.model = q_0
train_ctx.optimizer = q_0_optim
train_ctx.loss_fn = q_0_loss

q_0_loss, _ = train.simple_train(train_ctx)


# %%
plot_ctx.vmin= -0.23
plot_ctx.vmax = 0.14
plot_ctx.titles = ["f model"]
plot_ctx.save_path="./heat_hd_results/f_model.png"


utils.plot_function_on_2d_cube([f], plot_ctx)


plot_ctx.vmin= 0
plot_ctx.vmax = 1
plot_ctx.titles = ["q_0 model"]
plot_ctx.save_path="./heat_hd_results/q_0_model.png"

utils.plot_function_on_2d_cube([q_0], plot_ctx)

# %%
def smoothing_fn(x):
    X = (x[:, 0] - 1) * (x[:, 0] + 1)
    Y1 = torch.where(x[:, 1] > 1/3, (x[:, 1] - 1/3)**2, 0.0)
    Y2 = torch.where(x[:, 1] < -1/3, (x[:, 1] + 1/3)**2, 0.0)
    
    return X * (Y1 + Y2)

def model_decorator(model_in : torch.Tensor, model_out: torch.Tensor):
    return f(model_in) * (q_0(model_in) + torch.mul(smoothing_fn(model_in), model_out))

# %%
model_ctx = mm.ModelContext(
    input_dim=2,
    output_dim=1,
    layer=[32, 32, 32],
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    hard_enforce_boundary=True,
    decorator=model_decorator
)

model_hd = mm.MLPModel(model_ctx).to(device)
hd_optim = torch.optim.Adam(model_hd.parameters(), lr=1e-4)
domain_ctx.N_int = 5_000

train_ctx.model = model_hd
train_ctx.optimizer = hd_optim
train_ctx.loss_fn = loss_hard_enforce

hd_loss, _ = train.simple_train(train_ctx)

plot_ctx.vmin = 0
plot_ctx.vmax = 1
plot_ctx.titles = ["Results hard-enforcement"]
plot_ctx.save_path="./heat_hd_results/prediction_hd.png"
utils.plot_function_on_2d_cube([lambda x: torch.abs(pde_residuum(x.requires_grad_(True), model_hd))], plot_ctx)


plot_ctx.vmin = 0
plot_ctx.vmax = 0.01
plot_ctx.titles = ["Residuals hd"]
plot_ctx.save_path="./heat_hd_results/residuals_hd.png"
utils.plot_function_on_2d_cube([lambda x: torch.abs(pde_residuum(x.requires_grad_(True), model_hd))], plot_ctx)