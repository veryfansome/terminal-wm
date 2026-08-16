import math

import torch

NAME = "r7_address_isometry_retraction"
DESCRIPTION = (
    "The two-timescale AdamW + Muon + spectral-cap + Polyak-tail optimizer with one mechanism "
    "added: after every step the projections whose job is to SEPARATE IDENTITIES are retracted "
    "toward the nearest row-isometry, at exactly constant Frobenius norm. Two disjoint families "
    "are selected by shape. The address family is every (key_d x w) two-dimensional weight whose "
    "input width is neither key_d nor D — the shared read/write address map applied to the raw "
    "command token's coordinate regions, and the delta-rule file/path read and write maps. The "
    "closure family is every (r x D) weight with 2 <= r <= router_out_max — the reference-closure "
    "query, key and probe maps. For each such W the retraction computes A = W / ||W||_F, whose "
    "largest singular value is bounded by 1 by construction, runs ortho_iters Bjorck-Schulz steps "
    "A <- 1.5 A - 0.5 (A A^T) A, which map every singular value monotonically toward 1 without "
    "overshoot from that starting bound, rescales the result BY ITS OWN MEASURED Frobenius norm "
    "so the target carries ||W||_F whether or not the iteration has converged, blends W a fixed "
    "fraction of the way there, and renormalizes the blend back to ||W||_F. The Frobenius norm is "
    "therefore an exact invariant of the retraction for any iteration count and any conditioning, "
    "so the only fixed points are the scaled row-isometries, weight decay and the learning-rate "
    "schedule keep sole authority over magnitude, and the closure attention temperature is carried "
    "through unchanged. Weights with min(rows, cols) == 1 have a single singular value, are already "
    "on the fixed set, and are skipped as a proven no-op rather than as an exclusion. The same "
    "retraction is re-applied at full strength after the Polyak tail copies its averaged weights "
    "in, because a uniform average of successive near-isometries is itself ill-conditioned. No new "
    "trainable parameter is introduced; every affected weight keeps receiving gradient from the "
    "loss on every step."
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


class _IsometryRetraction:

    def __init__(self, groups, iters):
        self.batches = []
        for ps, rate in groups:
            r = min(1.0, max(0.0, float(rate)))
            if r <= 0.0:
                continue
            by_shape = {}
            for p in ps:
                if p.ndim != 2 or min(p.shape[0], p.shape[1]) < 2:
                    continue
                by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape in sorted(by_shape):
                self.batches.append((by_shape[shape], r))
        self.iters = max(1, int(iters))

    @torch.no_grad()
    def apply(self, full=False):
        for params, rate in self.batches:
            r = 1.0 if full else rate
            w = torch.stack([p.data for p in params]).float()
            fro = w.flatten(1).norm(dim=1).view(-1, 1, 1)
            a = w / fro.clamp_min(1e-12)
            for _ in range(self.iters):
                g = torch.bmm(a, a.transpose(1, 2))
                a = torch.baddbmm(a, g, a, beta=1.5, alpha=-0.5)
            a = a * (fro / a.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-12))
            ok = torch.isfinite(a).flatten(1).all(dim=1).view(-1, 1, 1) & (fro > 1e-8)
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            blend = ok.to(a.dtype) * r
            b = w * (1.0 - blend) + a * blend
            bn = b.flatten(1).norm(dim=1).view(-1, 1, 1)
            b = b * torch.where(bn > 1e-12, fro / bn.clamp_min(1e-12), torch.ones_like(bn))
            for i, p in enumerate(params):
                p.data.copy_(b[i])


class _NoOrtho:

    def apply(self, full=False):
        return None


class _PolyakTail:

    def __init__(self, params, total, window):
        self.params = list(params)
        self.total = int(total)
        self.start = max(0, self.total - int(window))
        self.count = 0
        self.n = 0
        self.avg = None

    @torch.no_grad()
    def observe(self):
        self.count += 1
        if self.count <= self.start:
            return False
        if self.avg is None:
            self.avg = [p.detach().clone() for p in self.params]
            self.n = 1
        else:
            self.n += 1
            w = 1.0 / float(self.n)
            for a, p in zip(self.avg, self.params):
                a.lerp_(p.detach().to(a.dtype), w)
        if self.count < self.total:
            return False
        for a in self.avg:
            if not torch.isfinite(a).all():
                return False
        for a, p in zip(self.avg, self.params):
            p.data.copy_(a)
        return True


class _IsometryOpt:

    def __init__(self, adamw, muon, cap, retraction, tail):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.retraction = retraction
        self.tail = tail

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
        self.retraction.apply()
        if self.tail is not None and self.tail.observe():
            self.cap.project()
            self.retraction.apply(full=True)

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


def _trunk_lambda(steps, warmup_frac, hold_frac, floor_ratio, tail):
    warm = max(20, int(warmup_frac * steps))
    hold = int(hold_frac * steps)
    decay_start = warm + hold
    decay_end = max(decay_start + 1, steps - int(tail))
    decay_len = max(1, decay_end - decay_start)
    floor = float(floor_ratio)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        if step >= decay_end:
            return floor
        p = (step - decay_start) / decay_len
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
        return floor + (1.0 - floor) * cos

    return lr_lambda


def _router_lambda(steps, warmup_frac):
    warm = max(20, int(warmup_frac * steps))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 1.0

    return lr_lambda


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.10, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         router_out_max=128, router_lr_mult=2.0, router_wd=0.0, avg_frac=0.25,
         ortho_rate=0.5, router_ortho_rate=0.35, ortho_iters=12):
    params = [p for p in params]
    steps = max(1, int(steps))
    window = min(steps, max(1, int(round(float(avg_frac) * steps))))

    address = [p for p in params
               if p.ndim == 2 and p.shape[0] == key_d
               and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in address:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in address if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    router = [p for p in params
              if p.ndim == 2 and p.shape[1] == D and p.shape[0] <= int(router_out_max)
              and id(p) not in key_ids]
    router_ids = {id(p) for p in router}

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    rest = [p for p in params if id(p) not in key_ids and id(p) not in router_ids]

    trunk_fn = _trunk_lambda(steps, warmup_frac, hold_frac, floor_ratio, window)
    router_fn = _router_lambda(steps, warmup_frac)

    groups = []
    lambdas = []
    if rest:
        groups.append({"params": rest, "lr": lr, "weight_decay": wd})
        lambdas.append(trunk_fn)
    if router:
        groups.append({"params": router, "lr": lr * float(router_lr_mult),
                       "weight_decay": float(router_wd)})
        lambdas.append(router_fn)
    if not groups:
        groups.append({"params": params, "lr": lr, "weight_decay": wd})
        lambdas.append(trunk_fn)

    adamw = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lambdas)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, trunk_fn))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()

    retraction = _IsometryRetraction(
        [(address, ortho_rate), (router, router_ortho_rate)], ortho_iters)
    if not retraction.batches:
        retraction = _NoOrtho()

    tail = _PolyakTail(params, steps, window) if params else None

    return _IsometryOpt(adamw, muon, cap, retraction, tail), _MultiSched(*scheds)
