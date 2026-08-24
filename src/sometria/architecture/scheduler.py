
import math

import torch as t


def lr_schedule(optimizer, warmup_steps: int, total_steps: int, min_factor: float = 0.01):
    """Linear warmup into cosine decay, stepped per optimizer step.

    A *factor* rather than an absolute floor, because a downstream finetune runs two
    parameter groups at very different rates. ``CosineAnnealingLR``'s ``eta_min`` is one
    number for every group, so a backbone at ``lr=1e-5`` under a head at ``lr=1e-3``
    would be annealed toward a floor at or above its own starting rate -- and would spend
    training warming up instead of decaying.
    """

    def factor(step: int) -> float:
        if step < warmup_steps:
            return min_factor + (1.0 - min_factor) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_factor + (1.0 - min_factor) * cosine

    return t.optim.lr_scheduler.LambdaLR(optimizer, factor)
