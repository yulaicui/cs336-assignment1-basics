import math
from typing import Callable, Optional, Union

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                # Gradient initialized as None until the first `.backward()` call
                if p.grad is None:
                    continue

                state: dict = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                theta: torch.Tensor = p.data
                m: torch.Tensor = state.get("m", torch.zeros_like(theta, requires_grad=False))
                v: torch.Tensor = state.get("v", torch.zeros_like(theta, requires_grad=False))

                # Update first-moment estimate
                m = beta1 * m + (1 - beta1) * grad

                # Update second-moment estimate
                v = beta2 * v + (1 - beta2) * torch.square(grad)

                # Compute adjusted learning rate for this iteration (t)
                lr_adjusted = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Compute weight decay delta
                weight_decay_delta = -lr * weight_decay * theta
                
                # Update parameters 
                p.data = theta - (lr_adjusted * m / (torch.sqrt(v) + eps)) + weight_decay_delta

                # Update states
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
