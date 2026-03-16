from typing import Tuple, List
from torch import nn, Tensor
from dataclasses import dataclass
from src.models.modules import HalfHeavyside, Normalize
import torch


@dataclass
class PerCharCtx:
    input_dim: int
    output_dim: int

    hidden_layers_per_output: List[List[int]]
    char_functions: List
    
    l_bounds: Tuple = ()
    u_bounds: Tuple = ()

    constraint_functions: Tuple = ()
    normalize_inputt: bool = True
    hard_enforce: bool = False


class PerCharModel(nn.Module):
    def __init__(self, model_ctx: PerCharCtx):
        super(PerCharModel, self).__init__()
        self.ctx = model_ctx
        self.g_0 = self.ctx.constraint_functions[0] if self.ctx.hard_enforce else None
        self.g_1 = self.ctx.constraint_functions[1] if self.ctx.hard_enforce else None

        sequentials_main = []
        for i, _ in enumerate(self.ctx.char_functions):
            sequential_main = []

            prev_dim = self.ctx.hidden_layers_per_output[i][0]

            if self.ctx.normalize_inputt:
                sequential_main.append(Normalize(self.ctx.l_bounds, self.ctx.u_bounds))

            sequential_main.append(nn.Linear(self.ctx.input_dim, prev_dim))

            for layer_dim in self.ctx.hidden_layers_per_output[i]:
                sequential_main.append(nn.Tanh())
                sequential_main.append(nn.Linear(prev_dim, layer_dim))
                
            sequential_main.append(nn.Tanh())
            sequential_main.append(nn.Linear(prev_dim, self.ctx.output_dim))
            
            sequentials_main.append(nn.Sequential(*sequential_main))

        self.experts = nn.ModuleList(sequentials_main)

    def forward(self, x: Tensor) -> Tensor:
        outs = [ex(x) for ex in self.experts]
        chars = [chi(x) for chi in self.ctx.char_functions]

        out = sum([chars[i] * outs[i] for i in range(len(outs))])

        if self.g_0 is not None and self.g_1 is not None:
            out = self.g_0(x) + self.g_1(x) * out
            
        return out

    def to(self, device):
        super(PerCharModel, self).to(device)
        return self


class MultiModel(nn.Module):
    def __init__(self, *args):
        super(MultiModel, self).__init__()
        self.models = nn.ModuleList([*args])
        
    def forward(self, x):
        return torch.column_stack([model(x) for model in self.models])
