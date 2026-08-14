import math

import torch

NAME = "r6_dualtimescale_slow_momentum_routing"
DESCRIPTION = (
    "Replaces the single-timescale Adam core with a dual-EMA (AdEMAMix-style) update: every "
    "parameter carries a fast gradient EMA (beta1) and a second, budget-adaptive SLOW EMA whose "
    "horizon is a fixed fraction of the run's step budget, and the step direction is "
    "(m_fast/bias_corr + alpha(t) * m_slow) / sqrt(v_hat), with alpha(t) and beta_slow(t) warmed "
    "in over the first 60% of the run so early training is plain Adam. The delta-rule addressing "
    "pairs (the identical-shape (key_d, d) read/write projections) take the SAME dual-timescale "
    "direction but with its singular values flattened by Newton-Schulz, so their step size is set "
    "by the orthogonalization and only the direction changes. A power-iteration soft spectral cap "
    "on (D, D) content readouts is retained as a norm guard. LR shape (warmup / hold / cosine to "
    "floor) is byte-identical to the parent so the measured difference isolates the update rule."
)

D = 768


def _ns_orth(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        s = x @ x.mT
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


def _slow_beta_at(k, warm, b_start, b_end):
    if warm <= 1:
        return b_end
    f = min(1.0, float(k) / float(warm))
    ls = math.log(b_start)
    le = math.log(b_end)
    den = (1.0 - f) * le + f * ls
    if abs(den) < 1e-12:
        return b_end
    val = math.exp(ls * le / den)
    if not math.isfinite(val):
        return b_end
    return min(max(val, b_start), b_end)


def _alpha_at(k, warm, alpha):
    if warm <= 1:
        return float(alpha)
    return float(alpha) * min(1.0, float(k) / float(warm))


def _resolve_slow_end(steps, slow_frac, lo, hi):
    horizon = max(2.0, float(slow_frac) * float(max(1, int(steps))))
    b = 1.0 - 1.0 / horizon
    return float(min(max(b, lo), hi))


class _DualTimescaleAdam(torch.optim.Optimizer):

    def __init__(self, params, lr, beta1=0.9, beta2=0.95, beta_slow_start=0.9,
                 beta_slow_end=0.999, alpha=2.0, alpha_warm=1, slow_warm=1,
                 eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, beta1=float(beta1), beta2=float(beta2),
                        beta_slow_start=float(beta_slow_start),
                        beta_slow_end=float(beta_slow_end), alpha=float(alpha),
                        alpha_warm=int(alpha_warm), slow_warm=int(slow_warm),
                        eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)
        self.global_k = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_k += 1
        k = self.global_k
        for group in self.param_groups:
            lr = float(group["lr"])
            b1 = group["beta1"]
            b2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]
            b3 = _slow_beta_at(k, group["slow_warm"], group["beta_slow_start"],
                               group["beta_slow_end"])
            a = _alpha_at(k, group["alpha_warm"], group["alpha"])
            bc1 = 1.0 - b1 ** k
            bc2 = 1.0 - b2 ** k
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["fast"] = torch.zeros_like(p)
                    st["slow"] = torch.zeros_like(p)
                    st["sq"] = torch.zeros_like(p)
                fast = st["fast"]
                slow = st["slow"]
                sq = st["sq"]
                fast.mul_(b1).add_(g, alpha=1.0 - b1)
                slow.mul_(b3).add_(g, alpha=1.0 - b3)
                sq.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                denom = sq.div(bc2).sqrt_().add_(eps)
                upd = fast.div(bc1).add_(slow, alpha=a).div_(denom)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


class _OrthDualTimescale(torch.optim.Optimizer):

    def __init__(self, params, lr, beta1=0.9, beta_slow_start=0.9, beta_slow_end=0.999,
                 alpha=2.0, alpha_warm=1, slow_warm=1, ns_steps=5, rms_match=0.2):
        defaults = dict(lr=lr, beta1=float(beta1), beta_slow_start=float(beta_slow_start),
                        beta_slow_end=float(beta_slow_end), alpha=float(alpha),
                        alpha_warm=int(alpha_warm), slow_warm=int(slow_warm),
                        ns_steps=int(ns_steps), rms_match=float(rms_match))
        super().__init__(params, defaults)
        self.global_k = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_k += 1
        k = self.global_k
        for group in self.param_groups:
            lr = float(group["lr"])
            b1 = group["beta1"]
            ns = group["ns_steps"]
            rms = group["rms_match"]
            b3 = _slow_beta_at(k, group["slow_warm"], group["beta_slow_start"],
                               group["beta_slow_end"])
            a = _alpha_at(k, group["alpha_warm"], group["alpha"])
            bc1 = 1.0 - b1 ** k
            for p in group["params"]:
                g = p.grad
                if g is None or p.ndim != 2:
                    continue
                if not torch.isfinite(g).all():
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["fast"] = torch.zeros_like(p)
                    st["slow"] = torch.zeros_like(p)
                fast = st["fast"]
                slow = st["slow"]
                fast.mul_(b1).add_(g, alpha=1.0 - b1)
                slow.mul_(b3).add_(g, alpha=1.0 - b3)
                u = fast.div(bc1).add_(slow, alpha=a)
                if not torch.isfinite(u).all():
                    continue
                o = _ns_orth(u, steps=ns)
                scale = rms * math.sqrt(max(p.shape[0], p.shape[1]))
                p.add_(o, alpha=-lr * scale)
        return loss


class _SpectralGuard:

    def __init__(self, params, cap=4.0, iters=2):
        self.params = [p for p in params if p is not None and p.ndim == 2]
        self.cap = float(cap)
        self.iters = max(1, int(iters))
        self.left = [None] * len(self.params)

    @torch.no_grad()
    def project(self):
        for i, p in enumerate(self.params):
            w = p.data
            if not torch.isfinite(w).all():
                continue
            u = self.left[i]
            if u is None or u.shape[0] != w.shape[0] or u.dtype != w.dtype or u.device != w.device:
                u = torch.nn.functional.normalize(w.new_ones(w.shape[0]), dim=0)
            v = None
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self.left[i] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _NoGuard:

    def project(self):
        return None


class _RoutingOpt:

    def __init__(self, core, orth, guard):
        self.core = core
        self.orth = orth
        self.guard = guard

    @property
    def param_groups(self):
        groups = list(self.core.param_groups)
        if self.orth is not None:
            groups = groups + list(self.orth.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        self.core.zero_grad(set_to_none=set_to_none)
        if self.orth is not None:
            self.orth.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.core.step()
        if self.orth is not None:
            self.orth.step()
        self.guard.project()

    def state_dict(self):
        d = {"core": self.core.state_dict()}
        if self.orth is not None:
            d["orth"] = self.orth.state_dict()
        return d


class _MultiSched:

    def __init__(self, *scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]


def _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio):
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    decay_len = max(1, steps - decay_start)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor_ratio + (1.0 - floor_ratio) * cos

    return lr_lambda


def make(params, steps, lr=4e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta1=0.9, beta2=0.95, alpha=2.0, alpha_warm_frac=0.6,
         slow_warm_frac=0.6, slow_horizon_frac=0.35, slow_start=0.9,
         slow_end_min=0.98, slow_end_max=0.9999, eps=1e-8, key_d=64,
         ns_steps=5, rms_match=0.2, spectral_cap=4.0, spectral_iters=2):
    params = [p for p in params]
    steps = max(1, int(steps))

    slow_end = _resolve_slow_end(steps, slow_horizon_frac, slow_end_min, slow_end_max)
    slow_warm = max(1, int(round(float(slow_warm_frac) * steps)))
    alpha_warm = max(1, int(round(float(alpha_warm_frac) * steps)))

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    core = _DualTimescaleAdam(rest if rest else params, lr=lr, beta1=beta1, beta2=beta2,
                              beta_slow_start=slow_start, beta_slow_end=slow_end,
                              alpha=alpha, alpha_warm=alpha_warm, slow_warm=slow_warm,
                              eps=eps, weight_decay=wd)
    scheds = [torch.optim.lr_scheduler.LambdaLR(core, lr_lambda)]

    orth = None
    if keys and rest:
        orth = _OrthDualTimescale(keys, lr=lr, beta1=beta1, beta_slow_start=slow_start,
                                  beta_slow_end=slow_end, alpha=alpha, alpha_warm=alpha_warm,
                                  slow_warm=slow_warm, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(orth, lr_lambda))

    guard = _SpectralGuard(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoGuard()

    return _RoutingOpt(core, orth, guard), _MultiSched(*scheds)
