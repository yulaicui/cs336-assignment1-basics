import math
import os
from typing import BinaryIO, IO, Iterable

import numpy as np
import numpy.typing as npt
import torch


EPSILON = 1e-6 # a very small value for numerical stability


def get_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2 * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    norms: list[torch.Tensor] = [torch.linalg.vector_norm(g) for g in grads]
    total_norm: torch.Tensor = torch.linalg.vector_norm(torch.stack(norms)).item() 

    if total_norm >= max_l2_norm:
        downscale_multiplier = max_l2_norm / (total_norm + EPSILON)
        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(downscale_multiplier)


def load_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device_name: str, random_seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    device: torch.device = torch.device(device_name)

    # randomly sample B starting indices
    rng = np.random.default_rng(seed=random_seed)
    starting_indices: list[int] = list(rng.choice(a=dataset.shape[-1] - context_length, size=batch_size, replace=False))

    x_ranges = np.array([list(range(i, i+context_length)) for i in starting_indices])
    y_ranges = np.array([list(range(i+1, i+context_length+1)) for i in starting_indices])
    x: torch.Tensor = torch.from_numpy(dataset[x_ranges])
    x.to(device)
    y: torch.Tensor = torch.from_numpy(dataset[y_ranges])
    y.to(device)

    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save(
        {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },
        out
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint: dict = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']