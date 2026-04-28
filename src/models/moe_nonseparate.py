from typing import Tuple, List
from torch import nn, Tensor
from dataclasses import dataclass
from src.models.modules import Normalize, DiscTanh
import torch


@dataclass
class MoECtx:
    input_dim: int
    output_dims: List[int]

    hidden_layers_per_output: List[List[int]]
    discontinuous_per_layer: List[bool]
    # Flat tuple of expert indices whose outputs are summed element-wise into one
    # tensor before the constraint functions are applied.
    # e.g. sum_outputs=(0, 1) with 3 experts → [expert_0 + expert_1, expert_2]
    # Experts not listed are kept as individual outputs and concatenated alongside.
    # If empty, all expert outputs are concatenated as-is.
    l_bounds: Tuple = ()
    u_bounds: Tuple = ()

    # Hard-BC pair applied to the full concatenated output (N, sum(output_dims)):
    #   out = g_0(x) + g_1(x) * out
    # Both callables must accept (N, input_dim) and return (N, sum(output_dims)).
    constraint_functions: Tuple = ()
    normalize_input: bool = True
    hard_enforce: bool = False


class MoEModel(nn.Module):
    def __init__(self, model_ctx: MoECtx):
        super(MoEModel, self).__init__()
        self.ctx = model_ctx
        self.g_0 = self.ctx.constraint_functions[0] if self.ctx.hard_enforce else None
        self.g_1 = self.ctx.constraint_functions[1] if self.ctx.hard_enforce else None

        sequentials = []
        for i, output_dim in enumerate(self.ctx.output_dims):
            sequential = []
            prev_dim = self.ctx.hidden_layers_per_output[i][0]

            if self.ctx.normalize_input:
                sequential.append(Normalize(self.ctx.l_bounds, self.ctx.u_bounds))

            sequential.append(nn.Linear(self.ctx.input_dim, prev_dim))

            for layer_dim in self.ctx.hidden_layers_per_output[i]:
                if self.ctx.discontinuous_per_layer[i]:
                    sequential.append(DiscTanh(prev_dim))
                else:
                    sequential.append(nn.Tanh())

                sequential.append(nn.Linear(prev_dim, layer_dim))

            sequential.append(nn.Tanh())
            sequential.append(nn.Linear(prev_dim, output_dim))

            sequentials.append(nn.Sequential(*sequential))

        self.experts = nn.ModuleList(sequentials)

    def forward(self, x: Tensor) -> Tensor:
        # Concatenate all expert outputs: (N, sum(output_dims))
        out = torch.cat([ex(x) for ex in self.experts], dim=1)

        if self.g_0 is not None and self.g_1 is not None:
            out = self.g_0(x) + self.g_1(x) * out

        return out

    def to(self, device):
        super(MoEModel, self).to(device)
        return self
