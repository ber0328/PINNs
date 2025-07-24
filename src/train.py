"""
    Modul obsahujici ruzne trenovaci algoritmy.
"""

from src.data.abstract_domain import AbstractDomain
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
import torch
import torch.optim as opt
from typing import List, Callable
from torch.optim.lr_scheduler import _LRScheduler
from dataclasses import dataclass


LOSS_FN = Callable[[Module, AbstractDomain], Tensor]


@dataclass
class TrainingContext:
    model: Module = None
    optimizer: Optimizer = None
    domain: AbstractDomain = None
    loss_fn: LOSS_FN = None
    scheduler: _LRScheduler = None
    N: int = 1000
    epochs: int = 5000
    resample_freq: int = 50
    monitor_gradient: bool = False
    monitor_lr: bool = False


def simple_train(ctx: TrainingContext) -> List:
    """
    Jednoduchy trenovaci algoritmus, ktery generuje nahodna data v kazde
    epose.
    """
    loss_values = []

    for epoch in range(ctx.epochs):
        ctx.optimizer.zero_grad()

        if epoch % ctx.resample_freq == 0:
            ctx.domain.generate_points()

        loss = ctx.loss_fn(ctx.model, ctx.domain)
        loss.backward()
        ctx.optimizer.step()

        torch.cuda.empty_cache()

        if not (ctx.scheduler is None):
            ctx.scheduler.step(loss)

        if epoch % 100 == 99 or epoch == 0:
            print(f"Loss at epoch {epoch + 1} is: {loss.item()}.", end=' ')
            loss_values.append(loss.item())

            if ctx.monitor_gradient:
                total_norm = 0.0
                for p in ctx.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Total gradient norm: {total_norm}", end=' ')

            if ctx.monitor_lr:
                print(f"Current learing rate: {ctx.optimizer.param_groups[0]['lr']}", end=' ')

            print()

    return loss_values


# TODO: either create seperate context for lbfgs, or add lbfgs params into current context
def train_switch_to_lbfgs(ctx: TrainingContext, epochs_with_lbfgs=500,
                          lbfgs_lr=1e-3, max_iter=20, history_size=10) -> List:
    loss_values = simple_train(ctx)

    def closure():
        optimizer.zero_grad()
        loss = ctx.loss_fn(ctx.model, ctx.domain)
        loss.backward()
        return loss

    print("Switching to LBFGS")
    optimizer = opt.LBFGS(ctx.model.parameters(), lr=lbfgs_lr,
                          max_iter=max_iter, history_size=history_size,
                          line_search_fn='strong_wolfe')

    for epoch in range(epochs_with_lbfgs):
        loss = optimizer.step(closure)

        if epoch % 100 == 99:
            print(f"Loss at lbfgs-epoch {epoch + 1} is: {loss.item()}")
            ctx.domain.generate_points()
            loss_values.append(loss.item())

    return loss_values


def train_with_lbfgs(ctx: TrainingContext) -> List:
    loss_values = []

    def closure():
        optimizer.zero_grad()
        loss = ctx.loss_fn(ctx.model, ctx.domain)
        loss.backward()
        return loss

    optimizer = opt.LBFGS(ctx.model.parameters(), lr=0.001)

    for epoch in range(ctx.epochs):
        loss = optimizer.step(closure)

        if epoch % 100 == 99:
            print(f"Loss {epoch + 1} is: {loss.item()}")
            loss_values.append(loss.item())

    return loss_values
