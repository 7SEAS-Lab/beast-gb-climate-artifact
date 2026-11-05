import math

import torch
import torch.distributed as dist
import torch.nn.init as init
from torch.utils.flop_counter import FlopCounterMode


def count_flops(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: torch.device,
    dtp_group: dist.ProcessGroup,
    n_samples: int = 1,
    bayesian=True,
):
    """
    Count floating point operations [TFLOP] in forward and backward pass of the model.

    FLOPs are accumulated over the entire group
    """
    x, y, _, _ = next(iter(dataloader))
    x, y = x.to(device), y.to(device)
    x.requires_grad_(True)
    y.requires_grad_(True)

    with FlopCounterMode(
        display=False
    ) as flop_counter:  # display=True breaks down by op
        if bayesian:
            pred = model(x, samples=n_samples)
        else:
            pred = model(x)

    forward_flops = torch.tensor(flop_counter.get_total_flops(), device=device)
    dist.all_reduce(forward_flops, group=dtp_group)

    with FlopCounterMode(
        display=False
    ) as flop_counter:  # display=True breaks down by op
        loss = loss_fn(pred, y)
        loss.backward()

    backward_flops = torch.tensor(flop_counter.get_total_flops(), device=device)
    dist.all_reduce(backward_flops, group=dtp_group)

    total_flops = forward_flops + backward_flops

    if dist.get_rank(group=None) == 0:
        print(f"Forward flops per iteration is {forward_flops / 1e12} TFLOP")
        print(f"Backward flops per iteration is {backward_flops / 1e12} TFLOP")
        print(
            f"Total forward-backward flops per iteration is {total_flops / 1e12} TFLOP"
        )

    return forward_flops, backward_flops, total_flops


def _init_uniform_override(self, parameter, fan_in) -> None:
    bound = math.sqrt(6.0 / fan_in) if fan_in > 0 else 0
    init.uniform_(parameter, -bound, bound)
