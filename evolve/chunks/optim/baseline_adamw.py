"""Contract for any optim impl: expose make(params, steps, **p) -> (optimizer, scheduler_or_None).
The harness calls scheduler.step() after each opt.step() if the scheduler is not None. Batch size
is a genome field on this axis, not a constant in the impl."""
import torch
NAME = "baseline_adamw"
DESCRIPTION = "AdamW lr 3e-4, weight_decay 1e-4, constant LR."
def make(params, steps, lr=3e-4, wd=1e-4):
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd), None
