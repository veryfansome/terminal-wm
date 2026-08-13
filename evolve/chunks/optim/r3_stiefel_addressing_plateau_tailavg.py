import math
import torch

NAME = "r3_stiefel_addressing_plateau_tailavg"
DESCRIPTION = (
    "AdamW + Muon on the (key_d x d) delta-rule addressing projections and a soft spectral-norm "
    "cap on the (D,D) latent-transition readout, plus two additions. (1) A soft Stiefel "
    "retraction: after every step each addressing projection is blended a fixed fraction of the "
    "way toward its orthogonal polar factor (Newton-Schulz), rescaled to its own Frobenius norm, "
    "so the addressing map stays a full-rank rotation onto a freely chosen key_d-dimensional "
    "subspace instead of drifting to low stable rank. (2) A constant-LR plateau over the closing "
    "fraction of training with a uniform tail average of every parameter across that plateau, "
    "written back into the live parameters on the final step and then re-retracted and re-capped "
    "so the averaged weights satisfy the same geometry the model trained under. Degrades to plain "
    "AdamW-with-schedule plus tail averaging on archs with neither surface."
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


class _MuonKeys(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, ns_steps=5, rms_match=0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps, rms_match=rms_match))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]; lr = group["lr"]
            ns = group["ns_steps"]; rms = group["rms_match"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)
                o = _ns_orth(u, steps=ns)
                scale = rms * math.sqrt(max(p.shape[0], p.shape[1]))
                p.add_(o, alpha=-lr * scale)
        return loss


class _StiefelRetraction:

    def __init__(self, params, rate=0.2, ns_steps=6):
        self.params = list(params)
        self.rate = float(rate)
        self.ns_steps = max(1, int(ns_steps))

    @torch.no_grad()
    def apply(self, rate=None):
        r = self.rate if rate is None else float(rate)
        if r <= 0.0:
            return
        r = min(1.0, r)
        for p in self.params:
            if p is None or p.ndim != 2:
                continue
            w = p.data
            if not torch.isfinite(w).all():
                continue
            fro = float(w.norm())
            if not math.isfinite(fro) or fro < 1e-8:
                continue
            o = _ns_orth(w, steps=self.ns_steps)
            if not torch.isfinite(o).all():
                continue
            onorm = float(o.norm())
            if not math.isfinite(onorm) or onorm < 1e-8:
                continue
            o = o * (fro / onorm)
            if not torch.isfinite(o).all():
                continue
            w.mul_(1.0 - r).add_(o, alpha=r)


class _SpectralCap:

    def __init__(self, params, cap=4.0, iters=2):
        self.params = list(params)
        self.cap = float(cap)
        self.iters = int(iters)
        self._u = {}

    @torch.no_grad()
    def project(self):
        for p in self.params:
            if p is None or p.ndim != 2 or not torch.isfinite(p).all():
                continue
            w = p.data
            n = w.shape[0]
            u = self._u.get(id(p))
            if u is None or u.shape[0] != n:
                u = torch.nn.functional.normalize(w.new_ones(n), dim=0)
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _TailAverage:

    def __init__(self, params, start_step, total_steps):
        self.params = list(params)
        self.start_step = int(start_step)
        self.total_steps = int(total_steps)
        self.count = 0
        self.acc = None
        self.done = False

    @torch.no_grad()
    def observe(self, t):
        if self.done or t < self.start_step:
            return
        if self.acc is None:
            self.acc = [p.detach().to(torch.float32).clone() for p in self.params]
            self.count = 1
            return
        self.count += 1
        inv = 1.0 / float(self.count)
        for a, p in zip(self.acc, self.params):
            a.add_(p.detach().to(torch.float32).sub(a), alpha=inv)

    @torch.no_grad()
    def write_back(self):
        if self.done:
            return False
        self.done = True
        if self.acc is None or self.count < 2:
            return False
        wrote = False
        for a, p in zip(self.acc, self.params):
            if a.shape != p.shape or not torch.isfinite(a).all():
                continue
            p.data.copy_(a.to(p.dtype))
            wrote = True
        self.acc = None
        return wrote


class _GeoOpt:

    def __init__(self, adamw, muon, retract, cap, avg, total_steps):
        self.adamw = adamw
        self.muon = muon
        self.retract = retract
        self.cap = cap
        self.avg = avg
        self.total_steps = int(total_steps)
        self._t = 0

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
        self.retract.apply()
        self.cap.project()
        self._t += 1
        self.avg.observe(self._t)
        if self._t >= self.total_steps and not self.avg.done:
            if self.avg.write_back():
                self.retract.apply(rate=1.0)
                self.cap.project()

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict(), "t": self._t}
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


def _plateau_start(steps, warmup_frac, hold_frac, avg_frac):
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    plateau = int(round((1.0 - float(avg_frac)) * steps))
    plateau = min(max(plateau, decay_start + 1), max(1, steps - 1))
    return warm, decay_start, plateau


def _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio, avg_frac):
    warm, decay_start, plateau = _plateau_start(steps, warmup_frac, hold_frac, avg_frac)
    decay_len = max(1, plateau - decay_start)
    fr = float(floor_ratio)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        if step >= plateau:
            return fr
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return fr + (1.0 - fr) * cos

    return lr_lambda


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.15, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         retract_rate=0.2, retract_ns=6, avg_frac=0.25):
    params = [p for p in params]
    steps = max(1, int(steps))

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

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio, avg_frac)
    _, _, plateau = _plateau_start(steps, warmup_frac, hold_frac, avg_frac)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    retract = _StiefelRetraction(keys, rate=retract_rate, ns_steps=retract_ns)
    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    avg = _TailAverage(params, plateau, steps)

    return _GeoOpt(adamw, muon, retract, cap, avg, steps), _MultiSched(*scheds)
