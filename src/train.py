"""
    Modul obsahujici ruzne trenovaci algoritmy.
"""

from src.data.abstract_domain import AbstractDomain
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor
import torch
import torch.optim as opt
from typing import List, Callable, Tuple
from torch.optim.lr_scheduler import _LRScheduler
from dataclasses import dataclass
from torch.autograd import grad


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
    resample: bool = True
    resample_freq: int = 50
    monitor_gradient: bool = False
    monitor_lr: bool = False
    detection_metric: Callable = None
    reinit_loss: Callable = None
    autobalance_weights: bool = False
    loss_weights: List = None
    upadte_loss_freq: int = 50


def simple_train(ctx: TrainingContext) -> Tuple[List, List, List]:
    """
    Jednoduchy trenovaci algoritmus, ktery generuje nahodna data v kazde
    epose.
    """
    component_loss_values: List[List] = []
    component_grad_norm_values: List[List] = []
    total_loss_values = []

    for epoch in range(ctx.epochs):
        ctx.optimizer.zero_grad()

        loss_components = ctx.loss_fn(ctx.model, ctx.domain)
        loss = 0

        for i in range(0, len(ctx.loss_weights)):
            loss += ctx.loss_weights[i] * loss_components[i]
        
        if ctx.resample and epoch % ctx.resample_freq == 0:
            ctx.domain.generate_points()

        if not component_loss_values:
            component_loss_values = [[] for _ in range(len(loss_components))]
            component_grad_norm_values = [[] for _ in range(len(loss_components))]

        is_log_epoch = epoch % 100 == 99 or epoch == 0

        if ctx.monitor_gradient and is_log_epoch:
            component_grad_norms = []
            for lc in loss_components:
                ctx.optimizer.zero_grad()
                lc.backward(retain_graph=True)
                norm = 0.0
                for p in ctx.model.parameters():
                    if p.grad is not None:
                        norm += p.grad.data.norm(2).item() ** 2
                component_grad_norms.append(norm ** 0.5)
            ctx.optimizer.zero_grad()
            for i, cn in enumerate(component_grad_norms):
                component_grad_norm_values[i].append(cn)

        loss.backward()

        if not (ctx.scheduler is None):
            ctx.scheduler.step(loss.item())

        if is_log_epoch:
            print(f"Epoch: {epoch + 1}. Loss: {loss.item()}.", end=' ')

            for i, loss_component in enumerate(loss_components):
                component_loss_values[i].append(loss_component.item())

            total_loss_values.append(loss.item())

            if ctx.monitor_gradient:
                total_norm = 0.0
                for p in ctx.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Total grad: {total_norm}", end=' ')
                for i, cn in enumerate(component_grad_norms):
                    print(f"{i}-th grad: {cn}", end=' ')
                    print(f"{i}-th weight: {ctx.loss_weights[i]}", end=' ')

            if ctx.monitor_lr:
                print(f"Current learing rate: {ctx.optimizer.param_groups[0]['lr']}\n", end=' ')

        ctx.optimizer.step()

        ctx.loss_weights = update_weights_by_loss(ctx.loss_weights, loss_components)

    return total_loss_values, component_loss_values, component_grad_norm_values


# TODO: either create seperate context for lbfgs, or add lbfgs params into current context
def train_switch_to_lbfgs(ctx: TrainingContext, epochs_with_lbfgs=500,
                          lbfgs_lr=1e-3, max_iter=20, history_size=10) -> List:
    total_loss_values, component_loss_values, component_grad_norm_values = simple_train(ctx)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss_components = ctx.loss_fn(ctx.model, ctx.domain)
        loss = sum(loss_components)
        loss.backward()
        return loss

    print("Switching to LBFGS")
    optimizer = opt.LBFGS(ctx.model.parameters(), lr=lbfgs_lr,
                          max_iter=max_iter, history_size=history_size,
                          line_search_fn='strong_wolfe')

    for epoch in range(epochs_with_lbfgs):
        loss = optimizer.step(closure)

        if ctx.resample and epoch % 100 == 99:
            ctx.domain.generate_points()

        if epoch % 100 == 99:
            print(f"Loss at lbfgs-epoch {epoch + 1} is: {loss.item()}")
            # Konvence: loss_values[0] obsahuje totalni ztratu
            total_loss_values.append(loss.item())

    return total_loss_values, component_loss_values, component_grad_norm_values


def train_with_lbfgs(ctx: TrainingContext) -> List:
    loss_values = []

    def closure():
        optimizer.zero_grad()
        loss = ctx.loss_fn(ctx.model, ctx.domain)
        loss.backward()
        return loss

    optimizer = opt.LBFGS(ctx.model.parameters(), lr=0.001, line_search_fn=1)

    for epoch in range(ctx.epochs):
        loss = optimizer.step(closure)

        if epoch % 100 == 99:
            print(f"Loss {epoch + 1} is: {loss.item()}")
            loss_values.append(loss.item())

    return loss_values


def ri_loss(det_metric: torch.Tensor, model: torch.nn.Module, device: str) -> torch.Tensor:
    rand_det_metric = torch.rand((det_metric, model.output_dim), device=device)
    out = model(det_metric)
    return torch.mean((rand_det_metric - out)**2)


def update_weights_by_loss(weights, losses, beta=0.9, clip=(1.0e-3, 1.0e3)):
    losses = [l.item() for l in losses]
    
    mean_loss = sum(losses) / len(losses)
    
    new_weights = []
    
    for i in range(0, len(losses)):
        target = mean_loss / (losses[i] + 1.0e-12)
        target = max(clip[0], min(clip[1], target))
        new_weights.append(beta * weights[i] + (1.0 - beta) * target)
        
    avg = sum(new_weights) / len(new_weights)
    return [v / avg for v in new_weights]


def grad_norm(loss, params):
    loss_grads = grad(loss, params, torch.ones_like(loss), 
                      create_graph=False, retain_graph=True, allow_unused=True)
    
    total = 0
    for g in loss_grads:
        total += g.detach().pow(2).sum()
        
    return torch.sqrt(total + 1e-20).item()


def update_weights_grad(model, losses, weights, beta=0.9, clip=(1e-3, 1.0e3)):
    params = list(model.trunk.parameters()) if hasattr(model, "trunk") else list(model.parameters())

    grad_norms = [grad_norm(v, params) for v in losses]
    total_grad = sum(grad_norms) / len(grad_norms)
    
    new_weights = []
    
    for i in range(0, len(grad_norms)):
        target = total_grad / (grad_norms[i] + 1.0e-12)
        target = max(clip[0], min(clip[1], target))
        new_weights.append(beta * weights[i] + (1.0 - beta) * target)

    return new_weights