import math

import torch

NAME = "r22_lookahead_slowweight_consolidation"
DESCRIPTION = (
    "Lookahead slow-weight consolidation (Zhang et al. 2019, Alg.1) wrapped around the carried optimizer "
    "inner optimizer (Muon addressing + AdamW warmup-hold-cosine-floor + spectral-capped (D,D) "
    "readout, all verbatim): every la_k inner steps phi += alpha*(theta_k - phi), fast reset to "
    "phi, final step returns phi. Prop.2 variance reduction (same mean, strictly smaller variance "
    "fixed point) targets the low-SNR content-delivery params. la_alpha=1.0 recovers the unwrapped optimizer "
    "optimizer bit-for-bit (verified strict superset)."
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
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]; lr = group["lr"]; ns = group["ns_steps"]; rms = group["rms_match"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]; buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)
                o = _ns_orth(u, steps=ns)
                scale = rms * math.sqrt(max(p.shape[0], p.shape[1]))
                p.add_(o, alpha=-lr * scale)
        return loss


class _SpectralCap:
    def __init__(self, params, cap=4.0, iters=2):
        self.params = list(params); self.cap = float(cap); self.iters = int(iters); self._u = {}

    @torch.no_grad()
    def project(self):
        for p in self.params:
            if p is None or p.ndim != 2 or not torch.isfinite(p).all():
                continue
            w = p.data; n = w.shape[0]; u = self._u.get(id(p))
            if u is None or u.shape[0] != n:
                u = torch.nn.functional.normalize(w.new_ones(n), dim=0)
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _CapOpt:
    def __init__(self, adamw, muon, cap):
        self.adamw = adamw; self.muon = muon; self.cap = cap

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
    warm = max(20, int(warmup_frac * steps)); hold = int(hold_frac * steps)
    decay_start = warm + hold; decay_len = max(1, steps - decay_start)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len
        return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return lr_lambda


class _Lookahead:

    def __init__(self, inner, params, total_steps, la_k=5, la_alpha=0.5):
        self.inner = inner
        self.params = [p for p in params]
        self.la_k = max(1, int(la_k))
        self.la_alpha = float(la_alpha)
        self.total_steps = int(total_steps)
        self._t = 0
        self._slow = [p.detach().clone() for p in self.params]

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.inner.state_dict() if hasattr(self.inner, "state_dict") else {}

    @torch.no_grad()
    def step(self, closure=None):
        self.inner.step(closure)
        self._t += 1
        if self.la_alpha >= 1.0:
            return
        if self._t % self.la_k != 0 and self._t != self.total_steps:
            return
        for p, slow in zip(self.params, self._slow):
            if not torch.isfinite(p).all():
                continue
            slow.add_(p.detach() - slow, alpha=self.la_alpha)
            p.copy_(slow)


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30, floor_ratio=0.05,
         beta2=0.95, key_d=64, momentum=0.95, ns_steps=5, rms_match=0.2, spectral_cap=4.0,
         spectral_iters=2, la_k=5, la_alpha=0.5):
    params = [p for p in params]

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d and p.shape[1] != key_d and p.shape[1] != D]
    # Sibling rule: addressing projections always come as identical-shape read/write pairs, so a
    # lone tensor matching the signature (Embedding(key_d, d)) is not addressing and must not route.
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]
    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    def _wrap(inner, sched):
        return _Lookahead(inner, params, steps, la_k=la_k, la_alpha=la_alpha), sched

    if not keys and not dd:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return _wrap(opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda))

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    if not dd:
        from types import SimpleNamespace
        cap = SimpleNamespace(project=lambda: None)
        return _wrap(_CapOpt(adamw, muon, cap), _MultiSched(*scheds))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    return _wrap(_CapOpt(adamw, muon, cap), _MultiSched(*scheds))
