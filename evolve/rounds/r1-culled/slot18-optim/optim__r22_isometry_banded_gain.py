import math

import torch

NAME = "r22_isometry_banded_gain"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the (key_d x d) addressing projections + the "
    "spectral cap on the (D,D) transition readout, extended with two per-hop retention controls: "
    "(1) scalar and vector parameters (the memory-retention logit, LayerNorm gains, biases) are "
    "moved to a weight-decay-free group so decay cannot drag the retention logit toward the middle "
    "of its range; (2) an ISOMETRY BAND on the multiplicative half of the transition's gain head - "
    "the first D rows of every (2D, h) head that is not zero-initialised at construction get a "
    "row-norm cap and their bias is clamped, which keeps the per-hop content gain (1+gamma) inside "
    "a narrow band around 1 so a file's content survives three or four silent hops."
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
            v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
            for _ in range(self.iters):
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _IsometryBand:

    def __init__(self, weights, biases, row_cap, bias_cap):
        self.weights = list(weights)
        self.biases = list(biases)
        self.row_cap = float(row_cap)
        self.bias_cap = float(bias_cap)

    @torch.no_grad()
    def project(self):
        for p in self.weights:
            if not torch.isfinite(p).all():
                continue
            gain = p.data[:D]
            norm = gain.norm(dim=1, keepdim=True)
            scale = (self.row_cap / norm.clamp_min(1e-8)).clamp(max=1.0)
            gain.mul_(scale)
        for b in self.biases:
            if not torch.isfinite(b).all():
                continue
            b.data[:D].clamp_(-self.bias_cap, self.bias_cap)


class _ProjOpt:

    def __init__(self, adamw, muon, projectors):
        self.adamw = adamw
        self.muon = muon
        self.projectors = list(projectors)

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
        for pr in self.projectors:
            pr.project()

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


def _adamw_groups(tensors, wd):
    decayed = [p for p in tensors if p.ndim >= 2]
    plain = [p for p in tensors if p.ndim < 2]
    groups = []
    if decayed:
        groups.append({"params": decayed, "weight_decay": wd})
    if plain:
        groups.append({"params": plain, "weight_decay": 0.0})
    if not groups:
        groups.append({"params": tensors, "weight_decay": wd})
    return groups


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2, gain_row_cap=1.0,
         gain_bias_cap=0.25):
    params = [p for p in params]

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    # Addressing projections come as identical-shape siblings; a lone match is not addressing.
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    # make() runs on a freshly built net, so the nonzero test excludes the zero-init FiLM heads.
    gain_w = [p for p in params
              if p.ndim == 2 and p.shape[0] == 2 * D
              and float(p.detach().abs().max()) > 0.0]
    gain_b = [p for p in params
              if p.ndim == 1 and p.shape[0] == 2 * D
              and float(p.detach().abs().max()) > 0.0]

    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    projectors = []
    if dd:
        projectors.append(_SpectralCap(dd, cap=spectral_cap, iters=spectral_iters))
    if gain_w or gain_b:
        projectors.append(_IsometryBand(gain_w, gain_b, gain_row_cap, gain_bias_cap))

    adamw = torch.optim.AdamW(_adamw_groups(rest, wd), lr=lr, weight_decay=wd,
                              betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    if muon is None and not projectors:
        return adamw, scheds[0]

    return _ProjOpt(adamw, muon, projectors), _MultiSched(*scheds)
