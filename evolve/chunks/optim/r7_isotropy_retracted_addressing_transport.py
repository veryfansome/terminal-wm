import math

import torch

NAME = "r7_isotropy_retracted_addressing_transport"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the delta-rule addressing pairs + the soft "
    "spectral cap on (D,D) content readouts, PLUS a norm-preserving ISOTROPY RETRACTION applied "
    "after every step to the address projections (2-D weights whose output width is key_d) and "
    "the (D,D) content readouts. For each such weight W the retraction forms the Newton-Schulz "
    "polar factor Q of W, rescales Q to W's exact Frobenius norm, blends W <- (1-eta) W + eta T, "
    "then restores the Frobenius norm exactly. The operator therefore changes only the singular-"
    "value profile of W and never its magnitude: it moves mass out of the dominant singular "
    "directions into the starved ones, driving each address map toward a row-isometry and each "
    "content readout toward a scaled rotation. Magnitude control stays entirely with the spectral "
    "cap and weight decay. Unchanged on archs with neither an address projection nor a (D,D) "
    "readout."
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
        self.iters = max(1, int(iters))
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
            v = None
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            if v is None:
                continue
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _IsotropyRetraction:

    def __init__(self, params, eta=0.04, ns_steps=5, every=1):
        self.params = list(params)
        self.eta = float(eta)
        self.ns_steps = max(1, int(ns_steps))
        self.every = max(1, int(every))
        self.count = 0

    @torch.no_grad()
    def retract(self):
        if not self.params or self.eta <= 0.0:
            return
        self.count += 1
        if (self.count % self.every) != 0:
            return
        for p in self.params:
            if p is None or p.ndim != 2:
                continue
            w = p.data
            if not torch.isfinite(w).all():
                continue
            fro = float(w.norm())
            if not math.isfinite(fro) or fro <= 1e-12:
                continue
            q = _ns_orth(w, steps=self.ns_steps)
            if not torch.isfinite(q).all():
                continue
            qn = float(q.norm())
            if not math.isfinite(qn) or qn <= 1e-12:
                continue
            blend = w.mul(1.0 - self.eta).add_(q, alpha=self.eta * fro / qn)
            bn = float(blend.norm())
            if not math.isfinite(bn) or bn <= 1e-12:
                continue
            blend.mul_(fro / bn)
            if not torch.isfinite(blend).all():
                continue
            w.copy_(blend)


class _NoOp:

    def project(self):
        return None

    def retract(self):
        return None


class _IsoOpt:

    def __init__(self, adamw, muon, iso, cap):
        self.adamw = adamw
        self.muon = muon
        self.iso = iso
        self.cap = cap

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
        self.iso.retract()
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
         iso_eta=0.04, iso_ns_steps=5, iso_every=1, iso_min_dim=8):
    params = [p for p in params]
    steps = max(1, int(steps))
    kd = int(key_d)
    md = max(2, int(iso_min_dim))

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == kd
            and p.shape[1] != kd and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    iso = []
    for p in params:
        if p.ndim != 2:
            continue
        m, n = int(p.shape[0]), int(p.shape[1])
        if min(m, n) < md:
            continue
        if m == D and n == D:
            iso.append(p)
        elif m == kd and n != kd:
            iso.append(p)

    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not keys and not dd and not iso:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    retractor = _IsotropyRetraction(iso, eta=iso_eta, ns_steps=iso_ns_steps,
                                    every=iso_every) if iso else _NoOp()
    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoOp()

    return _IsoOpt(adamw, muon, retractor, cap), _MultiSched(*scheds)
