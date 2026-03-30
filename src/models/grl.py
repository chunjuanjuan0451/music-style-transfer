from __future__ import annotations
import torch
from torch import Tensor


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    return _GradReverseFn.apply(x, lambd)
