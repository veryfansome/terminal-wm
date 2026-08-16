import math

import torch

NAME = "r18_stiefel_address_isometry"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the delta-rule addressing pairs + the soft "
    "spectral cap on (D,D) latent-transition readouts, PLUS a scale-preserving Stiefel "
    "retraction applied after every step to the ADDRESSING GROUP. The addressing group is "
    "defined by function rather than by which optimizer touches it: every 2-D weight that maps "
    "a feature space onto a narrow address/score space and occurs as an identical-shape sibling "
    "pair, which covers both the (key_d, d) delta-rule read/write pairs and the (m, D) pairs "
    "that score content directly off the raw command embedding with m <= addr_out_max. The "
    "retraction is W <- (1 - tau) * W + tau * (||W||_F / ||Q||_F) * Q with Q the Newton-Schulz "
    "orthogonalization of W. Because the target is rescaled to the current Frobenius norm the "
    "step is norm-non-increasing and exactly norm-preserving at its fixed point, so it moves "
    "conditioning rather than scale and does not fight weight decay, while the singular values "
    "of W are driven toward equality. Address maps stay "
    "partial isometries, the bilinear content-matching form W_q^T W_k stays full rank inside "
    "its subspace, and rank-1 collapse of a scorer into a separable recency ranking is removed "
    "as a reachable solution. Unchanged on archs with no addressing pairs and no (D,D) readout."
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
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _NoCap:

    def project(self):
        return None


class _StiefelRetract:

    def __init__(self, params, tau=0.05, ns_steps=3, min_dim=4):
        self.params = list(params)
        self.tau = float(tau)
        self.ns_steps = max(1, int(ns_steps))
        self.min_dim = max(2, int(min_dim))

    @torch.no_grad()
    def retract(self):
        if self.tau <= 0.0:
            return
        for p in self.params:
            if p is None or p.ndim != 2:
                continue
            w = p.data
            if min(w.shape[0], w.shape[1]) < self.min_dim:
                continue
            if not torch.isfinite(w).all():
                continue
            fro = w.norm()
            if not torch.isfinite(fro) or float(fro) < 1e-8:
                continue
            q = _ns_orth(w, steps=self.ns_steps)
            if not torch.isfinite(q).all():
                continue
            qn = q.norm()
            if not torch.isfinite(qn) or float(qn) < 1e-8:
                continue
            q = q * (fro / qn)
            cand = w.mul(1.0 - self.tau).add_(q, alpha=self.tau)
            if not torch.isfinite(cand).all():
                continue
            w.copy_(cand)


class _NoRetract:

    def retract(self):
        return None


class _IsometryOpt:

    def __init__(self, adamw, muon, cap, retract):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.retract = retract

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
        self.retract.retract()

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


def _sibling_pairs(cands):
    counts = {}
    for p in cands:
        counts[tuple(p.shape)] = counts.get(tuple(p.shape), 0) + 1
    return [p for p in cands if counts[tuple(p.shape)] >= 2]


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         addr_out_max=128, stiefel_tau=0.05, stiefel_ns=3, stiefel_min_dim=4):
    params = [p for p in params]
    steps = max(1, int(steps))

    cand_keys = [p for p in params
                 if p.ndim == 2 and p.shape[0] == key_d
                 and p.shape[1] != key_d and p.shape[1] != D]
    keys = _sibling_pairs(cand_keys)
    key_ids = {id(p) for p in keys}

    cand_scorers = [p for p in params
                    if p.ndim == 2 and p.shape[1] == D
                    and p.shape[0] <= int(addr_out_max) and p.shape[0] != D
                    and id(p) not in key_ids]
    scorers = _sibling_pairs(cand_scorers)
    scorer_ids = {id(p) for p in scorers}

    addressing = keys + [p for p in scorers if id(p) not in key_ids]

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not keys and not dd and not scorer_ids:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()

    if addressing and float(stiefel_tau) > 0.0:
        retract = _StiefelRetract(addressing, tau=stiefel_tau, ns_steps=stiefel_ns,
                                  min_dim=stiefel_min_dim)
    else:
        retract = _NoRetract()

    return _IsometryOpt(adamw, muon, cap, retract), _MultiSched(*scheds)
