import torch
import torch.nn as nn
from typing import List


class MLPModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List):
        super(MLPModel, self).__init__()
        self.device = torch.device("cpu")
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        previous_dim = hidden_dim[0]

        for dim in hidden_dim:
            layers.append(nn.Tanh())
            layers.append(nn.Linear(previous_dim, dim))
            previous_dim = dim

        layers.append(nn.Tanh())
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def to(self, device):
        super(MLPModel, self).to(device)
        self.device = device
        return self
