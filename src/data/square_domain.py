import torch
from src.data.abstract_domain import AbstractDomain
from typing import Tuple, List
from dataclasses import dataclass
from scipy.stats import qmc


@dataclass
class SquareDomainConfig:
    """
        Dataclass pro zabaleni dat potrebnych pro vytvoreni instance SquareDomain
    """
    # omezeni intervalu
    x_0: float = 0,
    x_1: float = 1,
    y_0: float = 0,
    y_1: float = 1,
    # pocet generovanych bodu na mnozinu
    N_int: int = 1000
    N_left: int = 100
    N_right: int = 100
    N_bottom: int = 100
    N_top: int = 100
    # strategie generovani bodu
    boundary_strategy: str = 'uniform'
    interior_strategy: str = 'uniform'
    # je-li tento atribut True, budou y-ove hodnoty na osach x_0 a x_1 stejne.
    mirror_right_left: bool = False
    # device
    device: str = 'cpu'


class SquareDomain(AbstractDomain):
    """
        Trida, ktera obstarava nahodnych dat ve 2d intervalu, a to zvlast vnitrnich bodu, a zvlast hranicnich bodu.
    """
    def __init__(self, config: SquareDomainConfig):
        self.cfg = config
        # vnitrni body
        self.interior = self._gen_rand_int()
        # leva hrana je pro y libovolna a x=x_0
        self.left_boundary = self._gen_rand_left_bnd()
        # horni hrana je pro x=x_1 a t libovolna
        self.top_boundary = self._gen_rand_top_bnd()
        # prava hrana je pro x livolna a y=y_1
        self.right_boundary = self._gen_rand_right_bnd()
        # leva hrana je pro x=x_0 a t libovolna
        self.bottom_boundary = self._gen_rand_bottom_bnd()

    def generate_points(self) -> None:
        self.interior = self._gen_rand_int()
        self.top_boundary = self._gen_rand_top_bnd()
        self.bottom_boundary = self._gen_rand_bottom_bnd()

        if self.cfg.mirror_right_left:
            self.right_boundary, self.left_boundary = self._gen_mirror_right_left()
        else:
            self.right_boundary = self._gen_rand_right_bnd()
            self.left_boundary = self._gen_rand_left_bnd()

    def get_all_points(self):
        return torch.cat([self.interior,
                          self.left_boundary,
                          self.top_boundary,
                          self.right_boundary,
                          self.bottom_boundary], dim=0)

    def _gen_rand_int(self) -> torch.Tensor:
        if self.cfg.interior_strategy == 'uniform':
            x = (self.cfg.x_1 - self.cfg.x_0) * torch.rand(self.cfg.N_int, 1, device=self.cfg.device) + self.cfg.x_0
            y = (self.cfg.y_1 - self.cfg.y_0) * torch.rand(self.cfg.N_int, 1, device=self.cfg.device) + self.cfg.y_0

            return torch.cat((x, y), 1)
        elif self.cfg.interior_strategy == 'latin':
            sampler = qmc.LatinHypercube(2)
            sample = sampler.random(self.cfg.N_int)
            sample = qmc.scale(sample, [self.cfg.x_0, self.cfg.x_0], [self.cfg.x_1, self.cfg.y_1])
            
            return torch.from_numpy(sample).to(self.cfg.device).float()

    def _gen_rand_left_bnd(self) -> torch.Tensor:
        """x=x_0 a y je libovolne"""
        x = torch.full((self.cfg.N_left, 1), self.cfg.x_0, device=self.cfg.device)

        if self.cfg.boundary_strategy == 'linear':
            y = torch.linspace(self.cfg.y_0, self.cfg.y_1, self.cfg.N_left, device=self.cfg.device).unsqueeze(1)
        elif self.cfg.boundary_strategy == 'uniform':
            y = (self.cfg.y_1 - self.cfg.y_0) * torch.rand(self.cfg.N_left, 1, device=self.cfg.device) + self.cfg.y_0

        return torch.cat((x, y), 1)

    def _gen_rand_top_bnd(self) -> torch.Tensor:
        """ x je libovlne a y=y_1 """
        if self.cfg.boundary_strategy == 'linear':
            x = torch.linspace(self.cfg.x_0, self.cfg.x_1, self.cfg.N_top, device=self.cfg.device).unsqueeze(1)
        elif self.cfg.boundary_strategy == 'uniform':
            x = (self.cfg.x_1 - self.cfg.x_0) * torch.rand(self.cfg.N_top, 1, device=self.cfg.device) + self.cfg.x_0

        y = torch.full((self.cfg.N_top, 1), self.cfg.y_1, device=self.cfg.device)

        return torch.cat((x, y), 1)

    def _gen_rand_right_bnd(self) -> torch.Tensor:
        """ x=x_1 a y je libovolne """
        x = torch.full((self.cfg.N_right, 1), self.cfg.x_1, device=self.cfg.device)

        if self.cfg.boundary_strategy == 'linear':
            y = torch.linspace(self.cfg.y_0, self.cfg.y_1, self.cfg.N_right, device=self.cfg.device).unsqueeze(1)
        elif self.cfg.boundary_strategy == 'uniform':
            y = (self.cfg.y_1 - self.cfg.y_0) * torch.rand(self.cfg.N_right, 1, device=self.cfg.device) + self.cfg.y_0

        return torch.cat((x, y), 1)

    def _gen_rand_bottom_bnd(self) -> torch.Tensor:
        """ x je libovolne a y=y_0 """
        if self.cfg.boundary_strategy == 'linear':
            x = torch.linspace(self.cfg.x_0, self.cfg.x_1, self.cfg.N_bottom, device=self.cfg.device).unsqueeze(1)
        elif self.cfg.boundary_strategy == 'uniform':
            x = (self.cfg.x_1 - self.cfg.x_0) * torch.rand(self.cfg.N_bottom, 1, device=self.cfg.device) + self.cfg.x_0

        y = torch.full((self.cfg.N_bottom, 1), self.cfg.y_0, device=self.cfg.device)

        return torch.cat((x, y), 1)

    def _gen_mirror_right_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ y je libovolne, ale stejne jak pro x_0, tak pro x_1 """
        y = (self.cfg.y_1 - self.cfg.y_0) * torch.rand(self.cfg.N_left, 1, device=self.cfg.device) + self.cfg.y_0
        x_0 = torch.full((self.cfg.N_left, 1), self.cfg.x_0, device=self.cfg.device)
        x_1 = torch.full((self.cfg.N_left, 1), self.cfg.x_1, device=self.cfg.device)

        return torch.cat((x_0, y), 1), torch.cat((x_1, y), 1)
