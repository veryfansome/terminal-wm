import math
import torch

NAME = "r23_powerlaw_consolidation_capped_readout"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the (key_d x d) delta-rule addressing "
    "projections + the soft spectral-norm cap on (D,D) content readouts, PLUS a Benna-Fusi "
    "style power-law consolidation cascade: every parameter carries a chain of exponential "
    "iterate averages with geometrically increasing windows (alpha_k = alpha * 4^-k), and in "
    "the final tail of training the live weight is pulled onto the capacity-weighted mixture "
    "of those averages, so the deployed weights are a power-law trailing average of the "
    "trajectory instead of one noisy iterate. The pull is confined to the tail, where the "
    "schedule already sits near its LR floor, so the closed-loop drag it introduces costs "
    "almost no optimisation progress."
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


class _PowerLawConsolidation:

    def __init__(self, params, steps, depth=4, alpha=0.25, ratio=4.0,
                 capacity=2.0, track_frac=0.30, tail_frac=0.30, pull_max=0.04):
        self.params = [p for p in params if p is not None and p.is_floating_point()]
        self.depth = max(1, int(depth))
        r = max(1.0001, float(ratio))
        a0 = min(0.9, max(1e-4, float(alpha)))
        self.alphas = [a0 * (r ** (-k)) for k in range(self.depth)]
        caps = [float(capacity) ** k for k in range(self.depth)]
        tot = sum(caps)
        self.weights = [c / tot for c in caps]
        total = max(1, int(steps))
        self.track_start = int(min(0.95, max(0.0, float(track_frac))) * total)
        self.pull_start = int((1.0 - min(1.0, max(0.0, float(tail_frac)))) * total)
        self.pull_start = max(self.pull_start, self.track_start)
        self.pull_len = max(1, total - self.pull_start)
        self.pull_max = max(0.0, float(pull_max))
        self.t = 0
        self.slots = {}

    @torch.no_grad()
    def step(self):
        self.t += 1
        if self.t < self.track_start or self.pull_max <= 0.0:
            return
        pull = 0.0
        if self.t >= self.pull_start:
            f = min(1.0, (self.t - self.pull_start) / float(self.pull_len))
            pull = self.pull_max * (f * f * (3.0 - 2.0 * f))
        for p in self.params:
            key = id(p)
            slot = self.slots.get(key)
            if slot is None:
                slot = ([p.detach().clone() for _ in range(self.depth)],
                        torch.zeros_like(p))
                self.slots[key] = slot
            us, buf = slot
            for u, a in zip(us, self.alphas):
                u.mul_(1.0 - a).add_(p, alpha=a)
            if pull <= 0.0:
                continue
            buf.zero_()
            for u, w in zip(us, self.weights):
                buf.add_(u, alpha=w)
            if not torch.isfinite(buf).all():
                continue
            p.mul_(1.0 - pull).add_(buf, alpha=pull)


class _ConsolidatingOpt:

    def __init__(self, adamw, muon, consol, cap):
        self.adamw = adamw
        self.muon = muon
        self.consol = consol
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
        if self.consol is not None:
            self.consol.step()
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


class _NoCap:
    def project(self):
        return None


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
         consol_depth=4, consol_alpha=0.25, consol_ratio=4.0, consol_capacity=2.0,
         consol_track_frac=0.30, consol_tail_frac=0.30, consol_pull=0.04):
    params = [p for p in params]

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
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

    consol = _PowerLawConsolidation(
        params, steps, depth=consol_depth, alpha=consol_alpha, ratio=consol_ratio,
        capacity=consol_capacity, track_frac=consol_track_frac,
        tail_frac=consol_tail_frac, pull_max=consol_pull,
    )

    if not keys:
        adamw = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
        muon = None
    else:
        adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()
    return _ConsolidatingOpt(adamw, muon, consol, cap), _MultiSched(*scheds)
