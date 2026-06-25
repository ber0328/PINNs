from src.data.abstract_domain import AbstractDomain
from typing import Tuple, List
import torch

class ParametricDomain(AbstractDomain):
    def __init__(self, core_domain: AbstractDomain, param_dim: int, l_bounds: List, 
                 u_bounds: List):
        self.core_domain = core_domain
        self.param_dim = param_dim
        self.l_bounds = torch.tensor(l_bounds, device=core_domain.ctx.device)
        self.u_bounds = torch.tensor(u_bounds, device=core_domain.ctx.device)
        
    
    def generate_points(self) -> torch.Tensor:
        core_int = self.core_domain.get_all_points()
        param_tensor = (self.u_bounds - self.l_bounds) * \
            torch.rand(self.param_dim).to(self.core_domain.ctx.device) + self.l_bounds
        param_tensor = param_tensor.repeat((core_int.shape[0], 1))
        self.interior = torch.column_stack([core_int, param_tensor])
        
    def get_all_points(self) -> torch.Tensor:
        if hasattr(self, "interior"):
            return self.interior
        else:
            return None