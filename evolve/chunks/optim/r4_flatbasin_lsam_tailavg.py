import math

import torch

NAME = "r4_flatbasin_lsam_tailavg"
DESCRIPTION = (
    "AdamW + Muon on the (key_d x d) addressing projections + a soft spectral-norm cap on the "
    "(D,D) latent-transition readout, run under a flat-basin regime with two added mechanisms. "
    "(1) Carried layerwise sharpness-aware perturbation: after each update every 2-D parameter is "
    "displaced by e_l = rho * (||w_l|| / (||g_l|| + eps)) * g_l, so the next step's single backward "
    "pass measures the gradient at a nearby higher-loss point; the displacement is subtracted again "
    "before the update is applied, so the parameters that are updated, decayed, capped and averaged "
    "are always the undisplaced ones. Displacement magnitude is a fixed fraction of each layer's own "
    "weight norm and is passed through nan_to_num, so no host synchronisation and no unbounded step "
    "is possible. (2) Constant-LR tail averaging: the schedule warms up, holds, cosine-decays to a "
    "floor reached at the start of the tail, then stays constant, and the undisplaced parameters of "
    "every tail step are accumulated into a running mean that is written into the parameters on the "
    "final optimiser step (spectral cap re-applied afterwards). Batch statistics need no refresh "
    "because the trunk normalises per token. Both mechanisms are inert on any genome that sets "
    "rho or the tail fraction to zero."
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
            mu = group["momentum"]
            lr = group["lr"]
            ns = group["ns_steps"]
            rms = group["rms_match"]
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
            for _ in range(self.iters - 1):
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
            u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _NoCap:

    def project(self):
        return None


class _FlatBasinOpt:

    def __init__(self, adamw, muon, cap, all_params, sam_params, steps,
                 rho, sam_start, avg_start, sam_eps):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.all_params = list(all_params)
        self.sam_params = list(sam_params)
        self.steps = max(1, int(steps))
        self.rho = float(rho)
        self.sam_eps = float(sam_eps)
        self.sam_start = int(sam_start)
        self.avg_start = int(avg_start)
        self.t = 0
        self.n_avg = 0
        self.avg = None
        self.disp = [None] * len(self.sam_params)
        self.displaced = False
        self.finalized = False

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

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict(), "t": self.t, "n_avg": self.n_avg}
        if self.muon is not None:
            d["muon"] = self.muon.state_dict()
        return d

    @torch.no_grad()
    def _undisplace(self):
        if not self.displaced:
            return
        for i, p in enumerate(self.sam_params):
            e = self.disp[i]
            if e is not None:
                p.data.sub_(e)
                self.disp[i] = None
        self.displaced = False

    @torch.no_grad()
    def _displace(self):
        applied = False
        for i, p in enumerate(self.sam_params):
            g = p.grad
            if g is None:
                continue
            wn = p.data.pow(2).sum().sqrt()
            gn = g.pow(2).sum().sqrt()
            coef = (self.rho * wn) / (gn + self.sam_eps)
            e = torch.nan_to_num(g.mul(coef), nan=0.0, posinf=0.0, neginf=0.0)
            self.disp[i] = e
            p.data.add_(e)
            applied = True
        self.displaced = applied

    @torch.no_grad()
    def _accumulate(self):
        if self.avg is None:
            self.avg = [p.data.detach().to(torch.float32).clone() for p in self.all_params]
            self.n_avg = 1
            return
        self.n_avg += 1
        w = 1.0 / float(self.n_avg)
        for a, p in zip(self.avg, self.all_params):
            a.lerp_(p.data.to(a.dtype), w)

    @torch.no_grad()
    def _finalize(self):
        self.finalized = True
        if self.avg is None or self.n_avg < 2:
            self.avg = None
            return
        for a, p in zip(self.avg, self.all_params):
            if torch.isfinite(a).all():
                p.data.copy_(a.to(p.dtype))
        self.cap.project()
        self.avg = None

    def step(self, closure=None):
        self._undisplace()
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        self.cap.project()
        self.t += 1
        if not self.finalized and self.t >= self.avg_start:
            self._accumulate()
        if not self.finalized and self.t >= self.steps:
            self._finalize()
            return None
        if self.rho > 0.0 and self.t >= self.sam_start and self.sam_params:
            self._displace()
        return None


class _MultiSched:
    def __init__(self, *scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]


def _phase_points(steps, warmup_frac, hold_frac, avg_frac):
    steps = max(1, int(steps))
    warm = min(max(20, int(warmup_frac * steps)), max(1, steps - 1))
    hold = int(hold_frac * steps)
    decay_start = min(warm + hold, max(1, steps - 1))
    avg_start = int((1.0 - max(0.0, min(0.9, float(avg_frac)))) * steps)
    avg_start = max(avg_start, decay_start + 1)
    avg_start = min(avg_start, max(1, steps - 1))
    return warm, decay_start, avg_start


def _schedule_lambda(warm, decay_start, avg_start, floor_ratio):
    decay_len = max(1, avg_start - decay_start)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        if step >= avg_start:
            return floor_ratio
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor_ratio + (1.0 - floor_ratio) * cos

    return lr_lambda


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.12, avg_frac=0.25, beta2=0.95, key_d=64, momentum=0.95,
         ns_steps=5, rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         rho=0.05, sam_eps=1e-12):
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

    warm, decay_start, avg_start = _phase_points(steps, warmup_frac, hold_frac, avg_frac)
    lr_lambda = _schedule_lambda(warm, decay_start, avg_start, float(floor_ratio))

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()

    sam_params = [p for p in params if p.ndim == 2 and p.requires_grad]

    opt = _FlatBasinOpt(adamw, muon, cap, params, sam_params, steps,
                        rho=float(rho), sam_start=warm, avg_start=avg_start,
                        sam_eps=float(sam_eps))
    return opt, _MultiSched(*scheds)
