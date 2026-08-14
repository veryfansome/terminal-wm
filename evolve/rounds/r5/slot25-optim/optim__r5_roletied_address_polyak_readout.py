import math

import torch

NAME = "r5_roletied_address_polyak_readout"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-to-a-high-floor) + Muon on the (key_d x d) addressing projections + "
    "a soft spectral-norm cap on (D,D) latent readouts, plus two post-step operators. (1) ROLE TIE: "
    "any shape group of EXACTLY two (key_d x m) addressing matrices is pulled, column by column, "
    "toward its best column-wise rank-one factorisation -- for each column j the 2x2 Gram of the "
    "two columns is diagonalised in closed form, its top eigenvector gives a shared unit direction "
    "p_j, and the two columns are replaced by (p_j . u_j) p_j and (p_j . v_j) p_j -- so the pair "
    "converges to W_a = P diag(s), W_b = P diag(t): one shared address projection with two "
    "per-feature role scalars. The pull is an interpolation with a ramped coefficient, so the pair "
    "starts free and is progressively constrained. (2) TAIL AVERAGE: from a fraction of the way "
    "through training an exponential moving average of every parameter is accumulated with a "
    "horizon set as a fraction of the step budget, and at the final optimizer step the "
    "bias-corrected average is written back into the live parameters, after which the role tie and "
    "the spectral cap are re-applied so the returned weights sit on the same constraints as the "
    "trajectory. Groups without a two-member addressing pair leave the tie inert; archs without a "
    "(D,D) matrix leave the cap inert."
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
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _RoleTie:

    def __init__(self, pairs, lam=0.15, ramp_steps=200):
        self.pairs = list(pairs)
        self.lam = float(lam)
        self.ramp = max(1, int(ramp_steps))

    @torch.no_grad()
    def project(self, t):
        if not self.pairs or self.lam <= 0.0:
            return
        frac = min(1.0, float(t) / float(self.ramp))
        lam = self.lam * frac
        if lam <= 0.0:
            return
        for pa, pb in self.pairs:
            u = pa.data
            v = pb.data
            if u.shape != v.shape or u.ndim != 2:
                continue
            uf = u.float()
            vf = v.float()
            if not (torch.isfinite(uf).all() and torch.isfinite(vf).all()):
                continue
            a = (uf * uf).sum(dim=0)
            c = (vf * vf).sum(dim=0)
            b = (uf * vf).sum(dim=0)
            half = 0.5 * (a - c)
            disc = torch.sqrt((half * half + b * b).clamp_min(0.0))
            w1 = half + disc
            w2 = b
            deg = (w1 * w1 + w2 * w2) < 1e-24
            w1 = torch.where(deg, torch.zeros_like(w1), w1)
            w2 = torch.where(deg, torch.ones_like(w2), w2)
            p = uf * w1.unsqueeze(0) + vf * w2.unsqueeze(0)
            pn = p.norm(dim=0)
            live = pn > 1e-8
            p = p / pn.clamp_min(1e-8).unsqueeze(0)
            ca = (p * uf).sum(dim=0)
            cb = (p * vf).sum(dim=0)
            m = (live.float() * lam).unsqueeze(0)
            un = uf * (1.0 - m) + (p * ca.unsqueeze(0)) * m
            vn = vf * (1.0 - m) + (p * cb.unsqueeze(0)) * m
            if torch.isfinite(un).all() and torch.isfinite(vn).all():
                u.copy_(un.to(u.dtype))
                v.copy_(vn.to(v.dtype))


class _TailAverage:

    def __init__(self, params, start_step, horizon):
        self.params = list(params)
        self.start = max(1, int(start_step))
        h = max(2, int(horizon))
        self.decay = 1.0 - 1.0 / float(h)
        self.shadow = None
        self.count = 0
        self.done = False

    @torch.no_grad()
    def accumulate(self, t):
        if self.done or t < self.start or not self.params:
            return
        if self.shadow is None:
            self.shadow = [torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                           for p in self.params]
            self.count = 0
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)
        self.count += 1

    @torch.no_grad()
    def commit(self):
        if self.done:
            return
        self.done = True
        if self.shadow is None or self.count < 2:
            return
        bias = 1.0 - self.decay ** self.count
        if not math.isfinite(bias) or bias <= 1e-6:
            return
        for s, p in zip(self.shadow, self.params):
            avg = s / bias
            if torch.isfinite(avg).all():
                p.data.copy_(avg.to(p.dtype))


class _TiedAvgOpt:

    def __init__(self, adamw, muon, cap, tie, avg, total_steps):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.tie = tie
        self.avg = avg
        self.total = max(1, int(total_steps))
        self.seen = 0

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
        self.seen += 1
        self.tie.project(self.seen)
        self.cap.project()
        self.avg.accumulate(self.seen)
        if self.seen >= self.total:
            self.avg.commit()
            self.tie.project(self.total)
            self.cap.project()

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict(), "seen": self.seen}
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
         floor_ratio=0.20, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         tie_lambda=0.15, tie_ramp_frac=0.05,
         avg_start_frac=0.45, avg_horizon_frac=0.12):
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

    pairs = []
    for shp, cnt in shape_counts.items():
        if cnt != 2:
            continue
        members = [p for p in cand if tuple(p.shape) == shp]
        pairs.append((members[0], members[1]))

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    rest = [p for p in params if id(p) not in key_ids]
    if not rest:
        rest = params
        keys = []

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    tie = _RoleTie(pairs, lam=tie_lambda,
                   ramp_steps=max(1, int(float(tie_ramp_frac) * steps)))
    avg = _TailAverage(params,
                       start_step=max(1, int(float(avg_start_frac) * steps)),
                       horizon=max(2, int(float(avg_horizon_frac) * steps)))

    return _TiedAvgOpt(adamw, muon, cap, tie, avg, steps), _MultiSched(*scheds)
