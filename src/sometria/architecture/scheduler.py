
import torch as t

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup into cosine decay, stepped per optimizer step."""

    return SequentialLR(
        optimizer,
        milestones=[warmup_steps],
        schedulers=[
            LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            ),
            CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-5
            ),
        ],
    )