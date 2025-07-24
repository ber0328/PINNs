import torch
import torch.nn as nn
from typing import List, Dict


class Sinn(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) 


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layer: List,
                 u_bounds: List, l_bounds: List, last_layer_activation: str = 'tanh'):
        super(MLPModel, self).__init__()
        self.u_bounds = torch.tensor(u_bounds)
        self.l_bounds = torch.tensor(l_bounds)
        layers = []
        layers.append(nn.Linear(input_dim, layer[0]))
        previous_dim = layer[0]

        for dim in layer:
            layers.append(nn.Tanh())
            layers.append(nn.Linear(previous_dim, dim))
            previous_dim = dim

        if last_layer_activation == 'sinn':
            layers.append(Sinn())
        else:
            layers.append(nn.Tanh())
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize for (1, -1)
        x = x - self.l_bounds
        x = x / (self.u_bounds - self.l_bounds)
        x = 2 * x - 1

        return self.network(x)

    def to(self, device):
        super(MLPModel, self).to(device)
        self.u_bounds = self.u_bounds.to(device=device)
        self.l_bounds = self.l_bounds.to(device=device)
        return self
