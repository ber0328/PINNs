from typing import Tuple, List
from torch import nn, Tensor
from dataclasses import dataclass
from src.models.modules import Normalize
import torch


@dataclass
class DiscModelCtx:
    input_dim: int
    output_dims: List[int]
    
    hidden_layers_per_output: List[List[int]]

    l_bounds: Tuple = ()
    u_bounds: Tuple = ()
    
    constraint_functions: Tuple = ()
    normalize_inupt: bool = True
    hard_enforce: bool = False


class DiscontinousModel(nn.Module):
    def __init__(self, model_ctx: DiscModelCtx):
        super(DiscontinousModel, self).__init__()
        sequentials = []
        self.ctx = model_ctx
        self.g_0 = self.ctx.constraint_functions[0] if self.ctx.hard_enforce else None
        self.g_1 = self.ctx.constraint_functions[1] if self.ctx.hard_enforce else None
        
        for i, output_dim in enumerate(self.ctx.output_dims):
            sequential = []
            prev_dim = self.ctx.hidden_layers_per_output[i][0]
            
            if self.ctx.normalize_inupt:
                sequential.append(Normalize(self.ctx.l_bounds, self.ctx.u_bounds))
                
            sequential.append(nn.Linear(self.ctx.input_dim, prev_dim))

            for layer_dim in self.ctx.hidden_layers_per_output[i]:
                sequential.append(nn.Tanh())
                sequential.append(nn.Linear(prev_dim, layer_dim))

            sequential.append(nn.Tanh())
            sequential.append(nn.Linear(prev_dim, output_dim))
            
            sequentials.append(nn.Sequential(*sequential))

        self.experts = nn.ModuleList(sequentials)
                
    def forward(self, x: Tensor):
        out = torch.column_stack([ex(x) for ex in self.experts])

        if self.g_0 is not None and self.g_1 is not None:
            out = self.g_0(x) + self.g_1(x) * out
        
        return out
    
    def to(self, device):
        super(DiscontinousModel, self).to(device)
        return self
