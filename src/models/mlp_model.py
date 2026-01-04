import torch
import torch.nn as nn
from typing import List
from math import tau
from dataclasses import dataclass


def out_decorator(func, modify=False):
    return func


class Sinn(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierFeatureTimeless(nn.Module):
    def __init__(self, input_dim: int, frequencies: int, scale: float = 10.0):
        super(FourierFeatureTimeless, self).__init__()

        B = torch.randn((input_dim, frequencies)) * scale

        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tau * x @ self.B
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class FourierFeature(nn.Module):
    def __init__(self, input_dim: int, frequencies: int, scale: float = 10.0):
        super(FourierFeature, self).__init__()

        B = torch.randn((input_dim - 1, frequencies)) * scale

        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tau * x[:, :-1] @ self.B
        return torch.cat([torch.cos(x), torch.sin(x), x[:, -1:]], dim=-1)


class HalfDiscontinuous(nn.Module):
    def __init__(self, eps, division_point: int):
        super(HalfDiscontinuous, self).__init__()
        self.division_point = division_point
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heaviside = torch.where(x[:self.division_point] < 0.0, 1.0, 0.0)
        heaviside = torch.cat([heaviside, torch.zeros_like(x[self.division_point:])])
        
        return torch.tanh(x) + self.eps * heaviside


@dataclass
class ModelContext:
    input_dim: int
    output_dim: int
    layer: List[int]
    u_bounds: List[float]
    l_bounds: List[float]
    last_layer_activation: str = 'tanh'
    fourier_features: str = ''
    fourier_frequencies: int = 10
    fourier_scale: float = 10.0
    normalize: bool = True
    hard_enforce_boundary: bool = False
    decorator: callable = None
    has_discontinuity: bool = False
    disc_steepeness: float = 8.0
    disc_eps: float = 0.0


class MLPModel(nn.Module):
    def __init__(self, ctx: ModelContext):
        super(MLPModel, self).__init__()
        self.u_bounds = torch.tensor(ctx.u_bounds)
        self.l_bounds = torch.tensor(ctx.l_bounds)
        self.fourier_features = ctx.fourier_features
        self.normalize = ctx.normalize
        self.ctx = ctx
        self.k = ctx.disc_steepeness
        layers = []

        if self.fourier_features == 'Timeless':
            layers.append(FourierFeatureTimeless(ctx.input_dim, ctx.fourier_frequencies, ctx.fourier_scale))
            layers.append(nn.Linear(2 * ctx.fourier_frequencies, ctx.layer[0]))
        elif self.fourier_features == 'Timedep':
            layers.append(FourierFeature(ctx.input_dim, ctx.fourier_frequencies, ctx.fourier_scale))
            layers.append(nn.Linear(2 * ctx.fourier_frequencies + 1, ctx.layer[0]))
        else:
            layers.append(nn.Linear(ctx.input_dim, ctx.layer[0]))

        previous_dim = ctx.layer[0]

        for i, dim in enumerate(ctx.layer):
            if ctx.has_discontinuity and i == len(ctx.layer) - 2:
                layers.append(HalfDiscontinuous(dim // 2))
            else:
                layers.append(nn.Tanh())
            layers.append(nn.Linear(previous_dim, dim))
            previous_dim = dim

        if ctx.last_layer_activation == 'sinn':
            layers.append(Sinn())
        elif ctx.last_layer_activation == 'disc':
            layers.append(HalfDiscontinuous(ctx.disc_eps, ctx.layer[-1]))
        else:
            layers.append(nn.Tanh())

        final_out_dim = ctx.output_dim * 3 if ctx.has_discontinuity else ctx.output_dim
        layers.append(nn.Linear(previous_dim, final_out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_temp = x
        
        if self.normalize:
            x = x - self.l_bounds
            x = x / (self.u_bounds - self.l_bounds)
            x = 2 * x - 1
        
        out = self.network(x)

        if self.ctx.has_discontinuity:
            od = self.ctx.output_dim
            base = out[:, :od]
            jump = out[:, od:2*od]
            phi = out[:, 2*od:3*od]
            
            k = self.k if self.k is not None else 8.0
            H = 0.5 * (1.0 + torch.tanh(k * phi))
            out = base + jump * H

        if self.ctx.hard_enforce_boundary:
            out = self.ctx.decorator(x_temp, out)
        
        return out

    def to(self, device):
        super(MLPModel, self).to(device)
        self.u_bounds = self.u_bounds.to(device=device)
        self.l_bounds = self.l_bounds.to(device=device)
        return self
