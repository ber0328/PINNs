from src.data.square_domain import SquareDomain, SquareDomainConfig
from dataclasses import dataclass
import torch
from numpy import pi


@dataclass
class SquareWithCircleDomainConfig(SquareDomainConfig):
    ball_x: float = 0.5
    ball_y: float = 0.5
    ball_r: float = 0.1
    ball_N: int = 100


class SquareWithCircleDomain(SquareDomain):
    def __init__(self, config: SquareWithCircleDomainConfig):
        super().__init__(config)
        self.ball_boundary = self._gen_rand_ball_bnd()

    def generate_points(self):
        super().generate_points()
        self._gen_rand_ball_bnd()

    def get_all_points(self):
        square_pts = super().get_all_points()
        return torch.cat([square_pts, self.ball_boundary], dim=0)

    def _gen_rand_int(self):
        pts = super()._gen_rand_int()
        mask = ((pts[:, 0] - self.cfg.ball_x)**2 + (pts[:, 1] - self.cfg.ball_y)**2) > self.cfg.ball_r**2
        return pts[mask]

    def _gen_rand_ball_bnd(self):
        theta = 2 * pi * torch.rand(self.cfg.ball_N, 1, device=self.cfg.device)

        x = self.cfg.ball_x + self.cfg.ball_r * torch.cos(theta)
        y = self.cfg.ball_x + self.cfg.ball_r * torch.sin(theta)

        return torch.cat([x, y], dim=1)
