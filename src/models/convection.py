import torch
import torch.nn as nn
from torch.autograd import grad


class ConvectionModel(nn.Module):
    def __init__(self):
        super(ConvectionModel, self).__init__()
        self.device = torch.device("cpu")
        self.fc1 = nn.Linear(2, 20)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(20, 20)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(20, 20)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(20, 20)
        self.tanh4 = nn.Tanh()
        self.fc5 = nn.Linear(20, 20)
        self.tanh5 = nn.Tanh()
        self.fc6 = nn.Linear(20, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        x = self.tanh3(x)
        x = self.fc4(x)
        x = self.tanh4(x)
        x = self.fc5(x)
        x = self.tanh5(x)
        x = self.fc6(x)

        return x

    def to(self, device):
        super(ConvectionModel, self).to(device)
        self.device = device
        return self

    def compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        u = self.forward(x)

        u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0][:, 0:1].to(self.device)
        u_y = grad(u, x, torch.ones_like(u), create_graph=True)[0][:, 1:2].to(self.device)

        return (u, u_x, u_y)

    def compute_loss(self, inter: torch.Tensor, bnd_left: torch.Tensor,
                     bnd_bot: torch.Tensor, bnd_top: torch.Tensor, beta: float) -> torch.Tensor:
        _, u_x, u_t = self.compute_gradient(inter)
        pde_loss = torch.mean((u_t + beta * u_x)**2)

        u_t_0 = self.forward(bnd_left)
        left_loss = torch.mean((u_t_0 - torch.sin(bnd_left[:, 0:1]) - torch.cos(bnd_left[:, 0:1]))**2).to(self.device)

        u_x_0 = self.forward(bnd_bot)
        u_x_1 = self.forward(bnd_top)
        top_bot_loss = torch.mean((u_x_0 - u_x_1)**2).to(self.device)

        return pde_loss + left_loss + top_bot_loss
