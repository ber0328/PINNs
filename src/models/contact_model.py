from typing import Tuple, List
from torch import nn, Tensor
from dataclasses import dataclass
from src.models.modules import HalfHeavyside, Normalize
import torch


@dataclass
class ContactModelCtx:
    model_1: nn.Module
    model_2: nn.Module

    k_1: float
    k_2: float
    
    char_func: callable
    region_2_boundary: callable
    
    l_bounds: Tuple = ()
    u_bounds: Tuple = ()


class ContactModel(nn.Module):
    def __init__(self, ctx: ContactModelCtx):
        super(ContactModel, self).__init__()
        self.ctx = ctx
        self.models = nn.ModuleList([ctx.model_1, ctx.model_2])
            
    def forward(self, x):
        pre_out_1 = self.models[0](x)
        pre_out_2 = self.models[1](x)
        
        chi = self.ctx.char_func(x)
        phi_0 = self.ctx.region_2_boundary(x)
        k_1_over_k_2 = (self.ctx.k_1 / self.ctx.k_2)
        
        out_2 = chi * (k_1_over_k_2 * pre_out_1.detach() + phi_0 * pre_out_2)
        out_1 = (1 - chi) * pre_out_1
        
        out = out_1 + out_2
        
        return out
