import math

import torch

NAME = "tailcycle_snapshot_average_spectral_muon"
DESCRIPTION = (
    "AdamW on the bulk, Muon (Newton-Schulz orthogonalised momentum) on the (key_d x d) "
    "addressing projections, and a soft spectral-norm cap on the (D,D) latent-transition "
    "readout, driven by a warmup-hold-cosine envelope that, over the final avg_frac of the step "
    "budget, switches into avg_cycles cosine restarts running from the envelope value at that "
    "point down to floor_ratio. The full parameter vector is snapshotted at the end of every "
    "restart and at the last step; on the last step the arithmetic mean of those snapshots is "
    "written back into the live parameters and the spectral cap is re-projected, so the network "
    "the harness hands to the instrument is the averaged iterate rather than the final one. "
    "Averaging is skipped when fewer than two finite snapshots were collected."
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


class _NoCap:

    def project(self):
        return None


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
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _SnapshotAverage:

    def __init__(self, params, marks):
        self.params = list(params)
        self.marks = set(int(m) for m in marks)
        self.acc = None
        self.count = 0

    @torch.no_grad()
    def observe(self, t):
        if int(t) not in self.marks:
            return
        for p in self.params:
            if not torch.isfinite(p.data).all():
                return
        if self.acc is None:
            self.acc = [torch.zeros_like(p.data, dtype=torch.float32) for p in self.params]
        for a, p in zip(self.acc, self.params):
            a.add_(p.data.detach().to(torch.float32))
        self.count += 1

    @torch.no_grad()
    def finalize(self):
        if self.acc is None or self.count < 2:
            return False
        wrote = False
        for a, p in zip(self.acc, self.params):
            m = a / float(self.count)
            if torch.isfinite(m).all():
                p.data.copy_(m.to(p.data.dtype))
                wrote = True
        return wrote


class _CycleAvgOpt:

    def __init__(self, adamw, muon, cap, averager, total):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.averager = averager
        self.total = max(1, int(total))
        self.t = 0
        self.merged = False

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
        self.t += 1
        self.averager.observe(self.t)
        if not self.merged and self.t >= self.total:
            self.merged = True
            if self.averager.finalize():
                self.cap.project()

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict(), "t": self.t, "merged": self.merged}
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


def _plan(steps, warmup_frac, hold_frac, floor_ratio, avg_frac, avg_cycles):
    total = max(1, int(steps))
    warm = max(1, min(total, max(20, int(warmup_frac * total))))
    hold = max(0, int(hold_frac * total))
    decay_start = min(total, warm + hold)
    decay_len = max(1, total - decay_start)
    floor = float(floor_ratio)

    def envelope(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
        return floor + (1.0 - floor) * cos

    frac = min(0.9, max(0.0, float(avg_frac)))
    tail = int(frac * total)
    avg_start = max(decay_start + 1, total - tail)
    n_cyc = max(1, int(avg_cycles))
    span = total - avg_start
    if span < n_cyc:
        n_cyc = max(1, span) if span > 0 else 1
    cyc_len = max(1, int(math.ceil(span / float(n_cyc)))) if span > 0 else 1
    hi = envelope(min(avg_start, total - 1))

    def lr_lambda(step):
        if step < avg_start:
            return envelope(step)
        u = ((step - avg_start) % cyc_len) / float(cyc_len)
        cos = 0.5 * (1.0 + math.cos(math.pi * u))
        return floor + (hi - floor) * cos

    marks = {total}
    if span > 0:
        for c in range(1, n_cyc + 1):
            marks.add(min(total, avg_start + c * cyc_len))
    return lr_lambda, marks, total


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         avg_frac=0.35, avg_cycles=4):
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

    lr_lambda, marks, total = _plan(steps, warmup_frac, hold_frac, floor_ratio,
                                    avg_frac, avg_cycles)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()
    averager = _SnapshotAverage(params, marks)

    return _CycleAvgOpt(adamw, muon, cap, averager, total), _MultiSched(*scheds)
