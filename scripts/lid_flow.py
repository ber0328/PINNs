import torch
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import datetime
sys.path.append('..')
from src import train, utils
from src import calculus as calc
import src.data.cube_domain as cb
import src.models.mlp_model as mm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_TEMP_SAVE_PATH = './most_eff_model.pt'
NU = 0.001
RE = 1 / NU
T_MAX = 5
L_BOUNDS = [0, 0, 0]
U_BOUNDS = [1, 1, T_MAX]
A = 0.3  # used during training to select A percentage of points with smallest detection metric


# NOTE: input is assumed to have requires_grad=True
def pde_residuum(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    u, p = output[:, :-1], output[:, -1:]
    _ = 0.0 * (u.sum() + p.sum())
    lhs = calc.material_derivative(input, u, device)
    rhs = - calc.nabla(input, p, True) + NU * calc.laplacian(input, u, device)
    return lhs - rhs


def continuity_residuum(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    u, v = output[:, 0:1], output[:, 1:2]
    u_x = grad(u, input, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    v_y = grad(v, input, torch.ones_like(v), create_graph=True)[0][:, 1:2]
    return u_x + v_y


count = 0
det_metric_computed = False
det_metric = 0
include_ri = True


def compute_detection_metric(u, input, dim):
    det_metric = 0

    for i in range(dim):
        abs_nabla_i = torch.abs(grad(u[:, i], input, torch.ones_like(u[:, i]), create_graph=True)[0][:, :-1])
        det_metric_i = torch.sum(abs_nabla_i, dim=1)
        det_metric += det_metric_i

    global det_metric_computed
    det_metric_computed = True

    return det_metric


def loss_fn(model: torch.nn.Module, domain: cb.CubeDomain) -> torch.Tensor:
    # pde loss
    pde_input = domain.interior.requires_grad_(True)
    pde_output = model(pde_input)
    pde_res = pde_residuum(pde_output, pde_input)
    cont_res = continuity_residuum(pde_output, pde_input)
    pde_loss = torch.mean(pde_res[:, 0:1]**2 + pde_res[:, 1:2]**2) + torch.mean(cont_res**2)

    global include_ri, det_metric

    if include_ri:
        if not det_metric_computed:
            det_metric = compute_detection_metric(pde_output, pde_input, 3)

        _, worst = torch.topk(det_metric, int(A * domain.ctx.N_int), largest=False)
        bad_pts = pde_input[worst].unsqueeze(1)
        rand_bad_pts = torch.rand((bad_pts.shape[0], 3), device=device)
        ri_loss = torch.mean((rand_bad_pts - pde_output[worst])**2)

        pde_loss += ri_loss

    # init loss
    init_input = domain.sides[-1][0].requires_grad_(True)
    init_output = model(init_input)
    u_init, p_init = init_output[:, :-1], init_output[:, -1:]
    init_loss = torch.mean(u_init[:, 0:1]**2 + u_init[:, 1:2]**2) + torch.mean(p_init**2)

    # top loss
    top_input = domain.sides[1][1].requires_grad_(True)
    u_top = model(top_input)[:, :-1]

    # side loss
    side_input = torch.cat([domain.sides[0][0], domain.sides[0][1], domain.sides[1][0]], dim=0).requires_grad_(True)
    u_side = model(side_input)[:, :-1]
    side_loss = torch.mean(torch.cat([u_side[:, 0:1]**2 + u_side[:, 1:2]**2, (u_top[:, 0:1] - 1)**2 + u_top[:, 1:2]**2], dim=0))

    return pde_loss + init_loss + side_loss


domain_ctx = cb.CubeContext(
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    dim=3,
    N_int=10_000,
    N_sides=[(1000, 1000), (1000, 1000), (1000, 1000)],
    device=device
)

domain = cb.CubeDomain(domain_ctx)


# defining the model
model_ctx = mm.ModelContext(
    input_dim=3,
    output_dim=3,
    layer=[64, 64, 128],
    u_bounds=U_BOUNDS,
    l_bounds=L_BOUNDS,
    last_layer_activation='tanh',
    fourier_features=True,
    fourier_frequencies=10,
    fourier_scale=10.0
)

model = mm.MLPModel(model_ctx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
# scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=300)


train_ctx = train.TrainingContext(
    model=model,
    optimizer=optimizer,
    domain=domain,
    loss_fn=loss_fn,
    monitor_lr=True,
    resample=False
)

loss_values = []
minimal_loss_value = sys.maxsize

print('Starting training...')
print('Starting exploration phase...')
# Total 30 * 2000 = 60_000 epochs

# as no resampling is done, one must manually generate points
domain.generate_points()

for i in range(30):
    # explore solutions
    print(f'Starting {i}-th exploration cycle.')

    print('\'Exploring\'')
    train_ctx.epochs = 500
    include_ri = True
    det_metric_computed = False
    loss_values += train.simple_train(train_ctx)

    # train current solution
    print('Done')
    train_ctx.epochs = 1500
    include_ri = False
    loss_values += train.simple_train(train_ctx)

    if loss_values[-1] < minimal_loss_value:
        print(f'Updating best performing model. Current loss: {loss_values[-1]}.')

        minimal_loss_value = loss_values[-1]
        torch.save(model.state_dict(), TRAIN_TEMP_SAVE_PATH)

print('Finishing exploration phase...')
print('Selecting best performing model and starting fine-tuning phase...')

include_ri = False
model.load_state_dict(torch.load(TRAIN_TEMP_SAVE_PATH, weights_only=True))
train_ctx.optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
train_ctx.epochs = 75_000
loss_values += train.train_switch_to_lbfgs(train_ctx, lbfgs_lr=0.1, epochs_with_lbfgs=500)

print('Finishing fine-tuning phase...')
print(f'Finishing training. Last loss: {loss_values[-1]}')

plot_ctx = utils.PlotContext(
    l_bounds=L_BOUNDS,
    u_bounds=U_BOUNDS,
    figsize=(8, 6),
    fontsize=14,
    x_label='Epochs',
    y_label='Loss',
    title='Total loss',
    N=100,
    device=device,
    vmin=0,
    vmax=1,
    patches=[],
    function_name='Velocity'
)

# vykresleni ztraty
plot_ctx.save_path = 'cavity_flow_loss.png'
utils.plot_loss_values({'Total loss': loss_values}, plot_ctx)

# plot_ctx.x_label = 'x'
# plot_ctx.y_label = 'y'


# def model_velocity_at_time(x: torch.Tensor, time: float):
#     time_dim = torch.full((x.shape[0], 1), time, device=device)
#     inp = torch.cat((x, time_dim), dim=1)
#     return model(inp)[:, :-1]


def model_pde_residuum(x: torch.Tensor):
    x.requires_grad_(True)
    output = model(x)
    return pde_residuum(output, x)


def model_div_residuum(x: torch.Tensor):
    x.requires_grad_(True)
    output = model(x)
    return continuity_residuum(output, x)


# # vykresleni vysledku
# for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
#     plot_ctx.title = f'Model at time t={t} with Re={RE}'
#     plot_ctx.save_path = f'images/cavity_flow_time_{t}.png'
#     utils.plot_vector_field_2d(lambda x: model_velocity_at_time(x, t), plot_ctx, N=25)

# vypis L2 normy residua
pde_res_l2 = calc.L2_norm(lambda x: 0, model_pde_residuum, 3, U_BOUNDS, L_BOUNDS, device, 20)
div_res_l2 = calc.L2_norm(lambda x: 0, model_div_residuum, 3, U_BOUNDS, L_BOUNDS, device, 20)

print('Writing log...')
with open('train_log.txt', 'a') as f:
    f.write(f'PDE residuum of model: {pde_res_l2}\n')
    f.write(f'Continuity condition residuum of model: {div_res_l2}\n')
    f.write(f'Last loss value: {loss_values[-1]}\n')
    f.write(f'Re: {RE}\n')
    f.write(f'A: {A}\n')


print('Saving finished model...')
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
torch.save(model.state_dict(), f'model_ns_re_1000_{date}.pt')
print('Model saved')
