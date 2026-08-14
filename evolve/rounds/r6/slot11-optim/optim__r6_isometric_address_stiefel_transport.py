import math

import torch

NAME = "r6_isometric_address_stiefel_transport"
DESCRIPTION = (
    "Riemannian Muon on the ADDRESS geometry of every associative store in the genome, plus the "
    "soft spectral cap on (D,D) readouts and a dormant-gain group. Any 2-D weight whose narrow "
    "side lies in [addr_out_min, addr_out_max] and whose wide side is at least addr_aspect times "
    "the narrow side is treated as an address projection — the arch's delta-rule read/write pairs, "
    "the latent-transition path projection, the verb codebook, and a head's shared path-to-key "
    "encoder even when it is a lone tensor of a shape nothing else shares. Those matrices are "
    "polar-retracted onto a fixed-radius Stiefel manifold at construction (Frobenius norm exactly "
    "preserved, only the conditioning changes) and are thereafter optimised on it: the momentum "
    "direction is projected onto the tangent space W_tan = U - sym(W_hat U^T) W_hat, "
    "Newton-Schulz-orthogonalised, re-projected, applied, and followed by a damped polar "
    "retraction back to radius s0. Address matrices therefore stay isometries up to a frozen "
    "scale, so distinct command features cannot be collapsed onto the same memory address by a "
    "rank-deficient projection. AdamW keeps the rest with weight decay only on multi-dimensional "
    "weights, and exactly-zero D-length gain vectors (zero-init injection channels) form their own "
    "undecayed, LR-boosted group. Falls back to plain AdamW when no address matrices exist."
)

D = 768
_EPS = 1e-7
_QA, _QB, _QC = 3.4445, -4.7750, 2.0315


def _ns_orth(g, steps=5):
    x = g.float()
    flipped = x.shape[0] > x.shape[1]
    if flipped:
        x = x.mT
    x = x / x.norm().clamp_min(_EPS)
    for _ in range(steps):
        s = x @ x.mT
        x = _QA * x + (_QB * s + _QC * (s @ s)) @ x
    if flipped:
        x = x.mT
    return x.to(g.dtype)


def _polar(w, quintic=4, cubic=2):
    x = w.float()
    flipped = x.shape[0] > x.shape[1]
    if flipped:
        x = x.mT
    x = x / x.norm().clamp_min(_EPS)
    for _ in range(max(1, int(quintic))):
        s = x @ x.mT
        x = _QA * x + (_QB * s + _QC * (s @ s)) @ x
    for _ in range(max(0, int(cubic))):
        s = x @ x.mT
        x = 1.5 * x - 0.5 * (s @ x)
    if flipped:
        x = x.mT
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _tangent(u, w_hat):
    a = w_hat @ u.mT
    return u - (0.5 * (a + a.mT)) @ w_hat


def _radius(p):
    m = min(int(p.shape[0]), int(p.shape[1]))
    s = float(p.data.norm())
    if not math.isfinite(s) or s <= 0.0:
        return 1.0 / math.sqrt(float(max(p.shape)))
    return s / math.sqrt(float(m))


class _StiefelMuon(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum=0.95, ns_steps=5, rms_match=0.2,
                 iso_tau=0.30, iso_quintic=4, iso_cubic=2, iso_init=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                      rms_match=rms_match, iso_tau=iso_tau,
                                      iso_quintic=iso_quintic, iso_cubic=iso_cubic))
        self.radius = {}
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.ndim != 2 or not bool(torch.isfinite(p).all()):
                        continue
                    s = _radius(p)
                    self.radius[id(p)] = s
                    if iso_init:
                        q = _polar(p.data, int(group["iso_quintic"]) + 2,
                                   int(group["iso_cubic"]) + 1)
                        if bool(torch.isfinite(q).all()):
                            p.data.copy_((q * s).to(p.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = float(group["momentum"])
            lr = float(group["lr"])
            ns = int(group["ns_steps"])
            rms = float(group["rms_match"])
            tau = float(group["iso_tau"])
            qn = int(group["iso_quintic"])
            cb = int(group["iso_cubic"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not bool(torch.isfinite(g).all()):
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)

                s = self.radius.get(id(p))
                on_manifold = p.ndim == 2 and s is not None and s > 0.0
                if on_manifold:
                    wide = p.shape[0] <= p.shape[1]
                    w_hat = (p.data if wide else p.data.mT).float() / s
                    u32 = (u if wide else u.mT).float()
                    if bool(torch.isfinite(w_hat).all()):
                        d32 = _tangent(_ns_orth(_tangent(u32, w_hat), steps=ns), w_hat)
                    else:
                        d32 = _ns_orth(u32, steps=ns)
                    d32 = torch.nan_to_num(d32, nan=0.0, posinf=0.0, neginf=0.0)
                    o = (d32 if wide else d32.mT).to(p.dtype)
                else:
                    o = torch.nan_to_num(_ns_orth(u, steps=ns),
                                         nan=0.0, posinf=0.0, neginf=0.0)

                scale = rms * math.sqrt(float(max(p.shape[0], p.shape[1])))
                p.add_(o, alpha=-lr * scale)

                if on_manifold and tau > 0.0:
                    q = _polar(p.data, qn, cb)
                    p.data.lerp_((q * s).to(p.dtype), tau)
        return loss


class _SpectralCap:

    def __init__(self, params, cap=4.0, iters=2):
        self.params = list(params)
        self.cap = float(cap)
        self.iters = max(1, int(iters))
        self.vectors = {}

    @torch.no_grad()
    def project(self):
        for p in self.params:
            if p is None or p.ndim != 2 or not bool(torch.isfinite(p).all()):
                continue
            w = p.data
            n = w.shape[0]
            u = self.vectors.get(id(p))
            if u is None or u.shape[0] != n:
                u = torch.nn.functional.normalize(w.new_ones(n), dim=0)
            v = None
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self.vectors[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _NoCap:

    def project(self):
        return None


class _StiefelOpt:

    def __init__(self, adamw, muon, cap):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap

    @property
    def param_groups(self):
        groups = []
        if self.adamw is not None:
            groups = groups + list(self.adamw.param_groups)
        if self.muon is not None:
            groups = groups + list(self.muon.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if self.adamw is not None:
            self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        self.cap.project()

    def state_dict(self):
        d = {}
        if self.adamw is not None:
            d["adamw"] = self.adamw.state_dict()
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
    floor = float(floor_ratio)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor + (1.0 - floor) * cos

    return lr_lambda


def _is_address(p, out_min, out_max, aspect):
    if p.ndim != 2:
        return False
    a, b = int(p.shape[0]), int(p.shape[1])
    lo, hi = (a, b) if a <= b else (b, a)
    if lo < int(out_min) or lo > int(out_max):
        return False
    return float(hi) >= float(aspect) * float(lo)


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, momentum=0.95, ns_steps=5, rms_match=0.2,
         spectral_cap=4.0, spectral_iters=2, addr_out_min=8, addr_out_max=96,
         addr_aspect=2.0, iso_tau=0.30, iso_quintic=4, iso_cubic=2, iso_init=1.0,
         gain_lr_mult=2.0, gain_dim=D):
    params = [p for p in params]
    steps = max(1, int(steps))

    addr = [p for p in params if _is_address(p, addr_out_min, addr_out_max, addr_aspect)]
    addr_ids = {id(p) for p in addr}

    dd = [p for p in params
          if p.ndim == 2 and int(p.shape[0]) == D and int(p.shape[1]) == D]

    decay, nodecay, gain = [], [], []
    for p in params:
        if id(p) in addr_ids:
            continue
        if p.ndim >= 2:
            decay.append(p)
        elif p.numel() == int(gain_dim) and float(p.detach().abs().max()) == 0.0:
            gain.append(p)
        else:
            nodecay.append(p)

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not addr:
        flat = decay + nodecay + gain
        opt = torch.optim.AdamW(flat if flat else params, lr=lr, weight_decay=wd,
                                betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": wd})
    if nodecay:
        groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
    if gain:
        groups.append({"params": gain, "lr": lr * float(gain_lr_mult), "weight_decay": 0.0})

    adamw = None
    scheds = []
    if groups:
        adamw = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        scheds.append(torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda))

    muon = _StiefelMuon(addr, lr=lr, momentum=momentum, ns_steps=ns_steps,
                        rms_match=rms_match, iso_tau=iso_tau, iso_quintic=iso_quintic,
                        iso_cubic=iso_cubic, iso_init=float(iso_init) > 0.0)
    scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()
    return _StiefelOpt(adamw, muon, cap), _MultiSched(*scheds)
