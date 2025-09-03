import torch
import torch.nn as nn
from typing import List
from math import tau
from dataclasses import dataclass


class Sinn(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierFeature(nn.Module):
    def __init__(self, input_dim: int, frequencies: int, scale: float = 10.0):
        super(FourierFeature, self).__init__()

        B = torch.randn((input_dim, frequencies)) * scale

        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tau * x @ self.B
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


@dataclass
class ModelContext:
    input_dim: int
    output_dim: int
    layer: List[int]
    u_bounds: List[float]
    l_bounds: List[float]
    last_layer_activation: str = 'tanh'
    fourier_features: bool = False
    fourier_frequencies: int = 10
    fourier_scale: float = 10.0


class MLPModel(nn.Module):
    def __init__(self, ctx: ModelContext):
        super(MLPModel, self).__init__()
        self.u_bounds = torch.tensor(ctx.u_bounds)
        self.l_bounds = torch.tensor(ctx.l_bounds)
        self.fourier_features = ctx.fourier_features
        layers = []

        if self.fourier_features:
            layers.append(FourierFeature(ctx.input_dim, ctx.fourier_frequencies, ctx.fourier_scale))
            layers.append(nn.Linear(2 * ctx.fourier_frequencies, ctx.layer[0]))
        else:
            layers.append(nn.Linear(ctx.input_dim, ctx.layer[0]))

        previous_dim = ctx.layer[0]

        for dim in ctx.layer:
            layers.append(nn.Tanh())
            layers.append(nn.Linear(previous_dim, dim))
            previous_dim = dim

        if ctx.last_layer_activation == 'sinn':
            layers.append(Sinn())
        else:
            layers.append(nn.Tanh())

        layers.append(nn.Linear(previous_dim, ctx.output_dim))
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
