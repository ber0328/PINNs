import src.data.cube_domain as cb
from dataclasses import dataclass
import torch


@dataclass
class AquariumContext(cb.CubeContext):
    ball_centre: torch.Tensor = torch.tensor([])
    ball_r: float = 0.1
    ball_N: int = 100


class AquariumDomain(cb.CubeDomain):
    def __init__(self, ctx: AquariumContext):
        super().__init__(ctx)
        self.ball_boundary = self._gen_rand_ball_bnd()

    def generate_points(self):
        super().generate_points()
        self.ball_boundary = self._gen_rand_ball_bnd()
        self.interior = self._select_pts_outside()

    def get_all_points(self):
        square_pts = super().get_all_points()
        return torch.cat([square_pts, self.ball_boundary], dim=0)

    def _select_pts_outside(self):
        mask = torch.sum((self.interior - self.ctx.ball_centre.squeeze())**2, dim=1) > self.ctx.ball_r**2
        return self.interior[mask]

    def _gen_rand_ball_bnd(self):
        ball_pts = self._gen_rand_bnd_tensor(torch.ones(self.ctx.dim, device=self.ctx.device),
                                             torch.full((self.ctx.dim, ), -1, device=self.ctx.device),
                                             (self.ctx.ball_N, self.ctx.dim))
        ball_pts_norm = torch.norm(ball_pts, p=2, dim=1).unsqueeze(1)
        ball_pts = (self.ctx.ball_r * ball_pts / ball_pts_norm) + self.ctx.ball_centre
        return ball_pts
