import src.data.cube_domain as cb
from dataclasses import dataclass
import torch


@dataclass
class AquariumContext(cb.CubeContext):
    ball_centre: torch.Tensor = torch.tensor([])
    ball_r: float = 0.1
    ball_N: int = 100
    ball_dim: int = -1


class AquariumDomain(cb.CubeDomain):
    def __init__(self, ctx: AquariumContext):
        super().__init__(ctx)
        self.ball_boundary = self._gen_rand_ball_bnd()

    def generate_points(self):
        super().generate_points()
        self.ball_boundary = self._gen_rand_ball_bnd()
        self.interior = self._select_pts_outside().detach()

    def get_all_points(self):
        square_pts = super().get_all_points()
        return torch.cat([square_pts, self.ball_boundary], dim=0)

    def _select_pts_outside(self):
        mask = torch.sum((self.interior[:, 0:self.ctx.ball_dim] - self.ctx.ball_centre.squeeze())**2, dim=1) > self.ctx.ball_r**2
        return self.interior[mask]

    def _gen_rand_ball_bnd(self):
        ball_pts = self._gen_rand_bnd_tensor(torch.ones(self.ctx.ball_dim, device=self.ctx.device),
                                             torch.full((self.ctx.ball_dim, ), -1, device=self.ctx.device),
                                             (self.ctx.ball_N, self.ctx.ball_dim))
        
        ball_pts_norm = torch.norm(ball_pts, p=2, dim=1).unsqueeze(1)
        ball_pts = (self.ctx.ball_r * ball_pts / ball_pts_norm) + self.ctx.ball_centre
        
        rest_pts = self._gen_rand_bnd_tensor(torch.tensor(self.ctx.l_bounds[self.ctx.ball_dim:], device=self.ctx.device),
                                             torch.tensor(self.ctx.u_bounds[self.ctx.ball_dim:], device=self.ctx.device),
                                             (self.ctx.ball_N, self.ctx.dim - self.ctx.ball_dim))
        
        return torch.column_stack((ball_pts, rest_pts))
