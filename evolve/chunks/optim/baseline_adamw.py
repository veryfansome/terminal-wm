"""optim chunk baseline: AdamW(lr 3e-4, wd 1e-4), constant LR (the R4 default).

Contract for any optim impl: expose make(params, steps) -> (optimizer, scheduler_or_None). The
harness calls scheduler.step() after each opt.step() if the scheduler is not None. Controls the
optimization STRATEGY (optimizer + LR schedule); batch size stays a harness/genome budget knob."""
import torch
NAME = "baseline_adamw"
DESCRIPTION = "AdamW lr 3e-4, weight_decay 1e-4, constant LR."
def make(params, steps, lr=3e-4, wd=1e-4):
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd), None
