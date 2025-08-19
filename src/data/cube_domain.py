import torch
from src.data.abstract_domain import AbstractDomain
from typing import Callable, Tuple, List
from dataclasses import dataclass
from numpy import pi


@dataclass
class CubeContext:
    l_bounds: List
    u_bounds: List
    dim: int
    N_int: int
    N_sides: List[Tuple]
    device: str = 'cpu'
    mirror_left_right: List = None
    bnd_sampling: str = 'Uniform'
    int_sampling: str = 'Uniform'
    use_point_selection: bool = False
    residuum_fn: Callable = None
    model: torch.nn.Module = None
    loss_fn: Callable = None
    bias_pts: List[Tuple] = None


class CubeDomain(AbstractDomain):
    def __init__(self, ctx: CubeContext):
        self.ctx = ctx
        self.interior: torch.Tensor
        self.sides: List[Tuple[torch.Tensor, torch.Tensor]]

    def generate_points(self):
        # generate side_pts
        sides = []
        for i in range(self.ctx.dim):
            side = self._gen_side(i)
            sides.append(side)

        self.sides = sides
        if self.ctx.use_point_selection:
            int_pts = self._gen_int_pts().requires_grad_(True)
            worst = self.select_worst(int_pts, self.ctx.N_int // 10).detach()
            int_rand = self._gen_int_pts().requires_grad_(True)
            self.interior = torch.cat((worst, int_rand), dim=1)
        else:
            self.interior = self._gen_int_pts()

    def get_all_points(self):
        points = self.interior

        for left_pts, right_pts in self.sides:
            points = torch.cat((points, left_pts, right_pts), dim=0)

        return points

    def get_side_points(self, n: int) -> torch.Tensor:
        points = torch.empty((0, self.ctx.dim), device=self.ctx.device)

        for left_pts, right_pts in self.sides[:n]:
            points = torch.cat((points, left_pts, right_pts), dim=0)

        return points

    def select_worst(self, points, N):
        int_residuum = self.ctx.residuum_fn(points, self.ctx.model)
        _, worst_int = torch.topk(int_residuum.abs(), N, dim=0)
        return points[worst_int].squeeze(1)

    def _gen_int_pts(self):
        u_bound = torch.tensor(self.ctx.u_bounds, device=self.ctx.device)
        l_bound = torch.tensor(self.ctx.l_bounds, device=self.ctx.device)

        if self.ctx.int_sampling == 'Uniform':
            return (u_bound - l_bound) * torch.rand((self.ctx.N_int, self.ctx.dim), device=self.ctx.device) + l_bound
        elif self.ctx.int_sampling == 'Biased':
            base_pts = (u_bound - l_bound) * torch.rand((self.ctx.N_int, self.ctx.dim), device=self.ctx.device) + l_bound

            for point, n, radius in self.ctx.bias_pts:
                pts = 2 * torch.rand((n, self.ctx.dim), device=self.ctx.device)
                pts = torch.sub(pts, 1)
                mask = torch.sum(pts[:, :-1]**2, dim=1) <= 1
                pts = radius * pts[mask]
                pts[:, :-1] += point
                base_pts = torch.cat([base_pts, pts], dim=0)

            return base_pts

    def _gen_side(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        u_bound = torch.tensor(self.ctx.u_bounds, device=self.ctx.device)
        l_bound = torch.tensor(self.ctx.l_bounds, device=self.ctx.device)

        N_left, N_right = self.ctx.N_sides[i]
        pts_left = self._gen_rand_bnd_tensor(u_bound, l_bound, (N_left, self.ctx.dim))

        if self.ctx.mirror_left_right is not None and self.ctx.mirror_left_right[i]:
            pts_right = torch.clone(pts_left)
        else:
            pts_right = self._gen_rand_bnd_tensor(u_bound, l_bound, (N_right, self.ctx.dim))

        pts_left[:, i] = l_bound[i]
        pts_right[:, i] = u_bound[i]

        return pts_left, pts_right

    def _gen_rand_bnd_tensor(self, u_bound: torch.Tensor, l_bound: torch.Tensor, size: Tuple) -> torch.Tensor:
        if self.ctx.bnd_sampling == 'Uniform':        
            return (u_bound - l_bound) * torch.rand(size, device=self.ctx.device) + l_bound
