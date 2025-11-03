import torch
from src.data.abstract_domain import AbstractDomain
from typing import Callable, Tuple, List
from dataclasses import dataclass
from scipy.stats import qmc
from math import prod
from itertools import product


@dataclass
class MeshContext2D:
    Nx: int = 100
    Ny: int = 100
    device: str = 'cpu'
    l_bounds: List = None
    u_bounds: List = None
    
    
class MeshDomain2D(AbstractDomain):
    def __init__(self, ctx: MeshContext2D):
        self.ctx = ctx
        self.generate_points()
        self._set_scheme()

    def generate_points(self):
        x = torch.linspace(self.ctx.l_bounds[0], self.ctx.u_bounds[0], self.ctx.Nx)
        y = torch.linspace(self.ctx.l_bounds[1], self.ctx.u_bounds[1], self.ctx.Ny)
        
        X, Y = torch.meshgrid(x, y)
        
        X_flat = X.flatten()[:, None]
        Y_flat = Y.flatten()[:, None]
                
        self.grid = torch.cat([X_flat, Y_flat], dim=1).type(torch.float32).to(self.ctx.device)
        self._select_boundaries(X_flat, Y_flat)
        
    def get_all_points(self):
        return self.grid
    
    def _set_scheme(self):
        self.mesh_scheme = []
        x_pts = torch.arange(1, self.ctx.Nx - 1)
        y_pts = torch.arange(1, self.ctx.Ny - 1)
            
        for x in x_pts:
            for y in y_pts:
                ind_x_y   = self._idx_at(x, y)
                ind_xp1_y = self._idx_at(x+1, y)
                ind_xm1_y = self._idx_at(x-1, y)
                ind_x_yp1 = self._idx_at(x, y+1)
                ind_x_ym1 = self._idx_at(x, y-1)
                self.mesh_scheme.append((
                    ind_x_y, ind_xp1_y, ind_xm1_y, ind_x_yp1, ind_x_ym1
                ))
        
        self.mesh_scheme = torch.tensor(self.mesh_scheme, device=self.ctx.device)
    
    def _select_boundaries(self, X_flat, Y_flat):
        self.boundaries = []
        #TODO: MAKE THIS ACTUALLY CLEAN
        
        # generate boundary masks
        lft_mask = torch.isclose(X_flat, torch.tensor(self.ctx.l_bounds[0], dtype=torch.float32))
        rgh_mask = torch.isclose(X_flat, torch.tensor(self.ctx.u_bounds[0], dtype=torch.float32))
        bot_mask = torch.isclose(Y_flat, torch.tensor(self.ctx.l_bounds[1], dtype=torch.float32))
        top_mask = torch.isclose(Y_flat, torch.tensor(self.ctx.u_bounds[1], dtype=torch.float32))

        # select mesh points
        X_lft = X_flat.detach().clone()[lft_mask].unsqueeze(1)
        Y_lft = Y_flat.detach().clone()[lft_mask].unsqueeze(1)
        X_rgh = X_flat.detach().clone()[rgh_mask].unsqueeze(1)
        Y_rgh = Y_flat.detach().clone()[rgh_mask].unsqueeze(1)
        X_bot = X_flat.detach().clone()[bot_mask].unsqueeze(1)
        Y_bot = Y_flat.detach().clone()[bot_mask].unsqueeze(1)
        X_top = X_flat.detach().clone()[top_mask].unsqueeze(1)
        Y_top = Y_flat.detach().clone()[top_mask].unsqueeze(1)
        # add to boundaries
        self.boundaries.append(torch.cat([X_lft, Y_lft], dim=1).type(torch.float32).to(self.ctx.device))
        self.boundaries.append(torch.cat([X_rgh, Y_rgh], dim=1).type(torch.float32).to(self.ctx.device))
        self.boundaries.append(torch.cat([X_bot, Y_bot], dim=1).type(torch.float32).to(self.ctx.device))
        self.boundaries.append(torch.cat([X_top, Y_top], dim=1).type(torch.float32).to(self.ctx.device))
    
    def _idx_at(self, n, i) -> int:
        return n * self.ctx.Nx + i
            
        