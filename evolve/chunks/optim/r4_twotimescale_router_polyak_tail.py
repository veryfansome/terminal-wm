import math

import torch

from evolve.chunks.optim.r18_spectral_capped_transition_readout import _MuonKeys, _SpectralCap

NAME = "r4_twotimescale_router_polyak_tail"
DESCRIPTION = (
    "AdamW + Muon on the delta-rule addressing pairs + the soft spectral cap on (D,D) readouts, "
    "re-timed as a two-timescale scheme with a Polyak-averaged tail. Two-dimensional weights whose "
    "input width is D and whose output width is at most router_out_max form their own parameter "
    "group: they are held at a constant multiple of the base learning rate for the whole run after "
    "warmup and carry no weight decay, while every other parameter warms up, holds, cosine-decays "
    "to a floor and then sits at that floor for the last avg_frac of the step budget. Across that "
    "constant-step tail the optimizer keeps a uniform running mean of every parameter and copies "
    "it into the live parameters on the final step, re-applying the spectral cap afterwards; the "
    "copy is skipped if any averaged tensor is non-finite."
)

D = 768


class _PolyakTail:

    def __init__(self, params, total, window):
        self.params = list(params)
        self.total = int(total)
        self.start = max(0, self.total - int(window))
        self.count = 0
        self.n = 0
        self.avg = None

    @torch.no_grad()
    def observe(self):
        self.count += 1
        if self.count <= self.start:
            return False
        if self.avg is None:
            self.avg = [p.detach().clone() for p in self.params]
            self.n = 1
        else:
            self.n += 1
            w = 1.0 / float(self.n)
            for a, p in zip(self.avg, self.params):
                a.lerp_(p.detach().to(a.dtype), w)
        if self.count < self.total:
            return False
        for a in self.avg:
            if not torch.isfinite(a).all():
                return False
        for a, p in zip(self.avg, self.params):
            p.data.copy_(a)
        return True


class _TwoTimescaleOpt:

    def __init__(self, adamw, muon, cap, tail):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.tail = tail

    @property
    def param_groups(self):
        groups = list(self.adamw.param_groups)
        if self.muon is not None:
            groups = groups + list(self.muon.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        self.cap.project()
        if self.tail is not None and self.tail.observe():
            self.cap.project()

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict()}
        if self.muon is not None:
            d["muon"] = self.muon.state_dict()
        return d


class _MultiSched:

    def __init__(self, *scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]


class _NoCap:

    def project(self):
        return None


def _trunk_lambda(steps, warmup_frac, hold_frac, floor_ratio, tail):
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    decay_end = max(decay_start + 1, steps - int(tail))
    decay_len = max(1, decay_end - decay_start)
    floor = float(floor_ratio)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        if step >= decay_end:
            return floor
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor + (1.0 - floor) * cos

    return lr_lambda


def _router_lambda(steps, warmup_frac):
    warm = max(20, int(warmup_frac * steps))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 1.0

    return lr_lambda


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.10, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         router_out_max=128, router_lr_mult=2.0, router_wd=0.0, avg_frac=0.25):
    params = [p for p in params]
    steps = max(1, int(steps))
    window = min(steps, max(1, int(round(float(avg_frac) * steps))))

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    router = [p for p in params
              if p.ndim == 2 and p.shape[1] == D and p.shape[0] <= int(router_out_max)
              and id(p) not in key_ids]
    router_ids = {id(p) for p in router}

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    rest = [p for p in params if id(p) not in key_ids and id(p) not in router_ids]

    trunk_fn = _trunk_lambda(steps, warmup_frac, hold_frac, floor_ratio, window)
    router_fn = _router_lambda(steps, warmup_frac)

    groups = []
    lambdas = []
    if rest:
        groups.append({"params": rest, "lr": lr, "weight_decay": wd})
        lambdas.append(trunk_fn)
    if router:
        groups.append({"params": router, "lr": lr * float(router_lr_mult),
                       "weight_decay": float(router_wd)})
        lambdas.append(router_fn)
    if not groups:
        groups.append({"params": params, "lr": lr, "weight_decay": wd})
        lambdas.append(trunk_fn)

    adamw = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lambdas)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, trunk_fn))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()
    tail = _PolyakTail(params, steps, window) if params else None

    return _TwoTimescaleOpt(adamw, muon, cap, tail), _MultiSched(*scheds)
