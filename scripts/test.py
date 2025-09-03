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

plot_ctx = utils.PlotContext(
    l_bounds=[0, 0, 0],
    u_bounds=[1, 1, 1],
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

model_ctx = mm.ModelContext(
    input_dim=3,
    output_dim=3,
    layer=[64, 64, 128],
    u_bounds=[0, 0, 0],
    l_bounds=[1, 1, 1],
    last_layer_activation='tanh',
    fourier_features=False,
    fourier_frequencies=128,
    fourier_scale=3.0
)

model = mm.MLPModel(model_ctx).to(device)
model.to(device)

model.load_state_dict(torch.load('./model_ns_re_1000.0_2025-08-28_15-00.pt'))


def model_velocity_at_time(x: torch.Tensor, time: float):
    time_dim = torch.full((x.shape[0], 1), time, device=device)
    inp = torch.cat((x, time_dim), dim=1)
    return model(inp)[:, :-1]

for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
    plot_ctx.title = f'Model at time t={t} with Re={1000}'
    plot_ctx.save_path = f'images/cavity_flow_time_{t}.png'
    utils.plot_vector_field_2d(lambda x: model_velocity_at_time(x, t), plot_ctx, N=25)