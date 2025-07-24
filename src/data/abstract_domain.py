import torch
from abc import ABC, abstractmethod


class AbstractDomain(ABC):
    """
    Abstraktni trida, ktera definuje jednotlive domeny, na nichz se resi
    jednotlive ulohy.
    Kazda domena:
        1) obsahuje jeden tesnor reprezentujici vnitrni body domeny.
        2) obsahuje jeden, ci vice tesnoru reprezentujici hranicni body domeny.
        3) je schopna generovat nahodna data pro vsechny polozky
           implementace -- implementuje funkci "generate_points".
        5) je schopna vracet vsechny body generovane touto tridou najednou
           -- implementuje funkci "get_all_points".

    Domeny mohou obsahovat i dalsi data, jako napriklad device, hodnoty
    reprezentujici omezeni intervalu a jine polozky nutne ke spravnemu
    generovani dat.
    """

    @abstractmethod
    def generate_points(self) -> torch.Tensor:
        """
        Vygenereuje nahodna data pro všechny položky implementace.
        :return: Tensor s nahodnymi daty.
        """
        pass

    @abstractmethod
    def get_all_points(self) -> torch.Tensor:
        """
        Vrati vsechny body generovane touto tridou.
        :return: Tensor se vsemi body.
        """
        pass
