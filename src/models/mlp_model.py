import torch
import torch.nn as nn
from typing import List, Tuple
from dataclasses import dataclass
from src.models.modules import HalfHeavyside, Normalize, Sinn, DiscTanh, FourierFeature


@dataclass
class ModelContext:
    input_dim: int
    output_dim: int
    layer: List[int]
    
    last_layer_activation: str = 'tanh'
    
    fourier_features: bool = False
    fourier_frequencies: int = 10
    fourier_scale: float = 10.0
    fourier_separator: int = 0
    
    normalize_input: bool = True
    output_scale_factor: torch.Tensor = None
    u_bounds: Tuple = ()
    l_bounds: Tuple = ()
    
    
    hard_enforce: bool = False
    constraint_functions: Tuple = ()

    disc_steepness: int = 0
    has_discontinuity: bool = False


class MLPModel(nn.Module):
    def __init__(self, ctx: ModelContext):
        super(MLPModel, self).__init__()
        self.ctx = ctx
        self.g_0 = self.ctx.constraint_functions[0] if self.ctx.hard_enforce else None
        self.g_1 = self.ctx.constraint_functions[1] if self.ctx.hard_enforce else None
        layers = []

        if ctx.normalize_input:
            layers.append(Normalize(ctx.l_bounds, ctx.u_bounds))
        
        if ctx.fourier_features:
            layers.append(FourierFeature(ctx.input_dim, ctx.fourier_frequencies,
                                         ctx.fourier_separator, ctx.fourier_scale))
        else:
            layers.append(nn.Linear(ctx.input_dim, ctx.layer[0]))

        previous_dim = 2 * ctx.fourier_frequencies + ctx.fourier_separator\
            if ctx.fourier_features else ctx.layer[0]

        for i, dim in enumerate(ctx.layer):
            if i == len(ctx.layer) - 1 and ctx.has_discontinuity:
                layers.append(DiscTanh(previous_dim))
            else:
                layers.append(nn.Tanh())

            layers.append(nn.Linear(previous_dim, dim))
            previous_dim = dim

        if ctx.last_layer_activation == 'sinn':
            layers.append(Sinn()) 
        elif ctx.last_layer_activation == 'disc':
            layers.append(DiscTanh(previous_dim))
        elif ctx.last_layer_activation == 'half_disc':
            layers.append(HalfHeavyside(previous_dim))
        elif ctx.last_layer_activation == 'relu':
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Tanh())

        layers.append(nn.Linear(previous_dim, ctx.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        
        if self.ctx.output_scale_factor is not None:
            out = self.ctx.output_scale_factor * out
        
        if self.g_0 is not None and self.g_1 is not None:
            out = self.g_0(x) + self.g_1(x) * out
        
        return out

    def to(self, device):
        super(MLPModel, self).to(device)
        return self