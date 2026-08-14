import math

import torch

NAME = "r5_relsam_spectral_transport_keys"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the (key_d x d) delta-rule addressing projections "
    "+ a soft spectral-norm cap on (D,D) latent-transition readouts, all wrapped in a "
    "one-backward sharpness-aware update: after each optimizer step every tensor is displaced "
    "along its own current gradient direction by a scale-invariant radius rho*||W||, so the next "
    "step's gradient is evaluated at the displaced point while the update itself is applied to "
    "the undisplaced weights. The radius is zero during warmup and annealed linearly to zero over "
    "the final fraction of the step budget, so training ends on undisplaced weights."
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


class _NoCap:

    def project(self):
        return None


class _RelativeSharpness:

    def __init__(self, params, rho, steps, warm_steps, tail_steps):
        self.params = list(params)
        self.rho = float(rho)
        self.steps = max(1, int(steps))
        self.warm = max(0, int(warm_steps))
        self.tail = max(1, int(tail_steps))
        self.displacement = [None] * len(self.params)
        self.t = 0

    def radius(self):
        if not math.isfinite(self.rho) or self.rho <= 0.0:
            return 0.0
        if self.t < self.warm:
            return 0.0
        start = self.steps - self.tail
        if self.t >= start:
            frac = float(self.steps - self.t) / float(self.tail)
            return self.rho * max(0.0, min(1.0, frac))
        return self.rho

    @torch.no_grad()
    def restore(self):
        for i, e in enumerate(self.displacement):
            if e is None:
                continue
            self.params[i].data.sub_(e)
            self.displacement[i] = None

    @torch.no_grad()
    def displace(self):
        r = self.radius()
        if r <= 0.0:
            return
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all():
                continue
            gn = float(g.norm())
            if not math.isfinite(gn) or gn <= 1e-12:
                continue
            wn = float(p.data.norm())
            if not math.isfinite(wn) or wn <= 1e-12:
                continue
            e = g.mul(r * wn / gn)
            if not torch.isfinite(e).all():
                continue
            p.data.add_(e)
            self.displacement[i] = e


class _SharpCapOpt:

    def __init__(self, adamw, muon, cap, sharp):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.sharp = sharp

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
        self.sharp.restore()
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        self.cap.project()
        self.sharp.t += 1
        self.sharp.displace()

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


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         sam_rho=0.02, sam_warm_frac=0.08, sam_tail_frac=0.12):
    params = [p for p in params]

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

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()

    warm_steps = int(float(sam_warm_frac) * steps)
    tail_steps = max(1, int(float(sam_tail_frac) * steps))
    sharp = _RelativeSharpness(params, sam_rho, steps, warm_steps, tail_steps)

    return _SharpCapOpt(adamw, muon, cap, sharp), _MultiSched(*scheds)
