import torch
from src.data.abstract_domain import AbstractDomain
from typing import Tuple
from dataclasses import dataclass


@dataclass
class SquareDomainConfig:
    """
        Dataclass pro zabaleni dat potrebnych pro vytvoreni instance SquareDomain
    """
    # omezeni intervalu
    x_0: int = 0
    x_1: int = 1
    y_0: int = 0
    y_1: int = 1
    # je-li tento atribut True, budou y-ove hodnoty na osach x_0 a x_1 stejne.
    mirror_right_left: bool = False
    # pocet generovanych bodu na mnozinu
    N: int = 1000
    # device
    device: str = 'cpu'


class SquareDomain(AbstractDomain):
    """
        Trida, ktera obstarava nahodnych dat ve 2d intervalu, a to zvlast vnitrnich bodu, a zvlast hranicnich bodu.
    """
    def __init__(self, config: SquareDomainConfig):
        self.x_0 = config.x_0
        self.x_1 = config.x_1
        self.y_0 = config.y_0
        self.y_1 = config.y_1
        self.mirror_right_left = config.mirror_right_left
        self.device = torch.device(config.device)
        # vnitrni body
        self.interior = self._gen_rand_int(config.N)
        # leva hrana je pro y libovolna a x=x_0
        self.left_boundary = self._gen_rand_left_bnd(config.N)
        # horni hrana je pro x=x_1 a t libovolna
        self.top_boundary = self._gen_rand_top_bnd(config.N)
        # prava hrana je pro x livolna a y=y_1
        self.right_boundary = self._gen_rand_right_bnd(config.N)
        # leva hrana je pro x=x_0 a t libovolna
        self.bottom_boundary = self._gen_rand_bottom_bnd(config.N)

    def generate_points(self, N=1000) -> None:
        self.interior = self._gen_rand_int(N)
        self.left_boundary = self._gen_rand_left_bnd(N)
        self.top_boundary = self._gen_rand_top_bnd(N)

        if self.mirror_right_left:
            self.right_boundary, self.left_boundary = self._gen_mirror_right_left(N)
        else:
            self.right_boundary = self._gen_rand_right_bnd(N)
            self.bottom_boundary = self._gen_rand_bottom_bnd(N)

    def plot_points():
        pass

    def get_all_points(self):
        return torch.cat([self.interior,
                          self.left_boundary,
                          self.top_boundary,
                          self.right_boundary,
                          self.bottom_boundary], dim=0)

    def _gen_rand_int(self, N: int) -> torch.Tensor:
        x = (self.x_1 - self.x_0) * torch.rand(N, 1, device=self.device) + self.x_0
        y = (self.y_1 - self.y_0) * torch.rand(N, 1, device=self.device) + self.y_0

        return torch.cat((x, y), 1)

    def _gen_rand_left_bnd(self, N: int) -> torch.Tensor:
        """x=x_0 a y je libovolne"""
        x = torch.full((N, 1), self.x_0, device=self.device)
        y = (self.y_1 - self.y_0) * torch.rand(N, 1, device=self.device) + self.y_0

        return torch.cat((x, y), 1)

    def _gen_rand_top_bnd(self, N: int) -> torch.Tensor:
        """ x je libovlne a y=y_1 """
        x = (self.x_1 - self.x_0) * torch.rand(N, 1, device=self.device) + self.x_0
        y = torch.full((N, 1), self.y_1, device=self.device)

        return torch.cat((x, y), 1)

    def _gen_rand_right_bnd(self, N: int) -> torch.Tensor:
        """ x=x_1 a y je libovolne """
        x = torch.full((N, 1), self.x_1, device=self.device)
        y = (self.y_1 - self.y_0) * torch.rand(N, 1, device=self.device) + self.y_0

        return torch.cat((x, y), 1)

    def _gen_rand_bottom_bnd(self, N: int) -> torch.Tensor:
        """ x je libovolne a y=y_0 """
        x = (self.x_1 - self.x_0) * torch.rand(N, 1, device=self.device) + self.x_0
        y = torch.full((N, 1), self.y_0, device=self.device)

        return torch.cat((x, y), 1)

    def _gen_mirror_right_left(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ y je libovolne, ale stejne jak pro x_0, tak pro x_1 """
        y = (self.y_1 - self.y_0) * torch.rand(N, 1, device=self.device) + self.y_0
        x_0 = torch.full((N, 1), self.x_0, device=self.device)
        x_1 = torch.full((N, 1), self.x_1, device=self.device)

        return torch.cat((x_0, y), 1), torch.cat((x_1, y), 1)
