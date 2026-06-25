import torch
from torch import nn
from src.models.mlp_model import MLPModel


class MultiheadModel(nn.Module):
    def __init__(self, trunk: MLPModel, *args):
        super(MultiheadModel, self).__init__()
        self.trunk = trunk
        self.models = nn.ModuleList([*args])
        
    def forward(self, x):
        x = self.trunk(x)
        return torch.column_stack([model(x) for model in self.models])
        