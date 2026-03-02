import torch
from torch import nn
from typing import Tuple
from math import tau


class Normalize(nn.Module):
    def __init__(self, l_bounds: Tuple, u_bounds: Tuple):
        super(Normalize, self).__init__()
        self.register_buffer('u_bounds', torch.tensor(u_bounds))
        self.register_buffer('l_bounds', torch.tensor(l_bounds))
        
    def forward(self, x):
        x = x - self.l_bounds
        x = x / (self.u_bounds - self.l_bounds)
        return 2 * x - 1
    
    def to(self, device):
        super(Normalize, self).to(device)
        return self
    
    
class Sinn(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class FourierFeature(nn.Module):
    def __init__(self, input_dim: int, frequencies: int, separator: int, scale: float = 10.0):
        super(FourierFeature, self).__init__()
        B = torch.randn((input_dim - separator, frequencies)) * scale
        self.separator = separator
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tau * x[:, self.separator:] @ self.B
        return torch.column_stack([torch.cos(x), torch.sin(x), x[:, :self.separator]])


class _STEHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class DiscTanh(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.jump_size   = nn.Parameter(0.2 * (torch.rand(input_dim) - 0.5))
        self.jump_centre = nn.Parameter(0.2 * (torch.rand(input_dim) - 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heaviside = _STEHeaviside.apply(x - self.jump_centre)
        return torch.tanh(x) + self.jump_size * heaviside
    
class HalfHeavyside(nn.Module):
    def __init__(self, input_dim: int):
        super(HalfHeavyside, self).__init__()
        self.halfway = input_dim//2
        self.jump_size = nn.Parameter((torch.rand(self.halfway, dtype=torch.float32) - 0.5))
        self.jump_centre = nn.Parameter((torch.rand(self.halfway, dtype=torch.float32) - 0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heaviside = _STEHeaviside.apply(x[:, :self.halfway] - self.jump_centre)

        out = torch.cat([heaviside, torch.tanh(x[:, self.halfway:])], dim=1)
        return out