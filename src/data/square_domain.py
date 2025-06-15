import torch


class SquareDomain2D:
    """
        Trida, ktera obstarava nahodnych dat ve 2d intervalu, a to zvlast vnitrnich bodu, a zvlast hranicnich bodu.
    """
    def __init__(self, x_0: float, x_1: float, t_0: float, t_1: float):
        self.x_0 = x_0
        self.x_1 = x_1
        self.t_0 = t_0
        self.t_1 = t_1

    def gen_rand_int(self, N):
        x = (self.x_1 - self.x_0) * torch.rand(N) + self.x_0
        t = (self.t_1 - self.t_0) * torch.rand(N) + self.t_0

        return torch.cat((x.unsqueeze(1), t.unsqueeze(1)), 1)

    def gen_rand_left_bnd(self, N):
        x = (self.x_1 - self.x_0) * torch.rand(N) + self.x_0
        t = torch.full((N, 1), self.t_0)

        return torch.cat((x.unsqueeze(1), t), 1)

    def gen_rand_top_bot_bnd(self, N):
        t = (self.t_1 - self.t_0) * torch.rand(N) + self.t_0
        x = torch.full((N, 1), self.x_0)

        bnd_bot = torch.cat((x, t.unsqueeze(1)), 1)

        x = torch.full((N, 1), self.x_1)

        bnd_top = torch.cat((x, t.unsqueeze(1)), 1)

        return (bnd_bot, bnd_top)
