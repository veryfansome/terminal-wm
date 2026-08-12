import math

import torch

NAME = "r2_spherical_addressing_tail_avg"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the sibling-paired (key_d x d) addressing "
    "projections, with every BIAS-FREE addressing matrix rescaled back to its initial Frobenius "
    "norm after each Muon step, so the orthogonalized fixed-RMS update produces a constant "
    "angular step instead of one that shrinks as the matrix grows. Keeps the soft spectral-norm "
    "cap (power-iteration projection of the top singular value) on (D,D) readouts, and adds a "
    "uniform average of the parameter iterates over the final fraction of the step budget which "
    "is written into the live parameters at the last optimizer step and re-projected through the "
    "cap. Falls back to AdamW groups on archs without those shapes."
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


class _MuonSphere(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum=0.95, ns_steps=5, rms_match=0.2, radii=None):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps, rms_match=rms_match))
        self.radii = dict(radii or {})

    @torch.no_grad()
    def project_spheres(self):
        for group in self.param_groups:
            for p in group["params"]:
                r = self.radii.get(id(p))
                if r is None or not (r > 0.0):
                    continue
                n = p.data.norm()
                ok = torch.isfinite(n) & (n > 1e-8)
                scale = torch.where(ok, r / n.clamp_min(1e-8), torch.ones_like(n))
                p.data.mul_(scale)

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
        self.project_spheres()
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


class _TailAverage:

    def __init__(self, params, total_steps, tail_frac):
        self.params = list(params)
        self.total = max(1, int(total_steps))
        frac = min(max(float(tail_frac), 0.0), 1.0)
        start = int(round((1.0 - frac) * self.total))
        self.start = min(max(start, 0), self.total - 1)
        self.acc = None
        self.count = 0

    @torch.no_grad()
    def observe(self, step):
        if step <= self.start:
            return
        if self.acc is None:
            self.acc = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]
        for a, p in zip(self.acc, self.params):
            a.add_(p.data.float())
        self.count += 1

    @torch.no_grad()
    def finalize(self):
        if self.acc is None or self.count == 0:
            return
        for a, p in zip(self.acc, self.params):
            m = a / float(self.count)
            if torch.isfinite(m).all():
                p.data.copy_(m.to(p.data.dtype))


class _CapAvgOpt:

    def __init__(self, adamw, muon, cap, avg, total_steps):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.avg = avg
        self.total = max(1, int(total_steps))
        self._n = 0
        self._done = False

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
        self._n += 1
        self.avg.observe(self._n)
        if self._n >= self.total and not self._done:
            self._done = True
            self.avg.finalize()
            if self.muon is not None:
                self.muon.project_spheres()
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


def _addressing_indices(params, key_d):
    cand = [i for i, p in enumerate(params)
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    counts = {}
    for i in cand:
        s = tuple(params[i].shape)
        counts[s] = counts.get(s, 0) + 1
    keys = [i for i in cand if counts[tuple(params[i].shape)] >= 2]
    sphere = []
    for i in keys:
        j = i + 1
        biased = (j < len(params) and params[j].ndim == 1
                  and params[j].shape[0] == params[i].shape[0])
        if not biased:
            sphere.append(i)
    return keys, sphere


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.35,
         floor_ratio=0.20, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2, sphere_scale=1.0,
         tail_frac=0.25):
    params = [p for p in params]
    steps = max(1, int(steps))

    key_idx, sphere_idx = _addressing_indices(params, int(key_d))
    keys = [params[i] for i in key_idx]
    key_ids = {id(p) for p in keys}

    radii = {}
    for i in sphere_idx:
        p = params[i]
        n = float(p.detach().norm())
        if math.isfinite(n) and n > 1e-8:
            radii[id(p)] = n * float(sphere_scale)

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]
    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonSphere(keys, lr=lr, momentum=momentum, ns_steps=ns_steps,
                           rms_match=rms_match, radii=radii)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()
    avg = _TailAverage(params, steps, tail_frac)

    return _CapAvgOpt(adamw, muon, cap, avg, steps), _MultiSched(*scheds)
