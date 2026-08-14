import math
import torch

NAME = "r6_gradmetric_address_whitening"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the delta-rule addressing projections + the soft "
    "spectral cap on (D,D) latent-transition readouts, PLUS a gradient-metric ADDRESS WHITENING "
    "projection on the same addressing pairs. For each (key_d, in) addressing weight the optimizer "
    "keeps a trace-normalized exponential average of grad^T grad, which is the KFAC input factor of "
    "that layer up to a per-sample output-gradient weighting, shrinks it toward the identity, and "
    "then takes a capped relative descent step on ||W A W^T / c - I||_F^2 after every optimizer "
    "step, where c is the mean diagonal of W A W^T so the constraint fixes only the conditioning of "
    "the map and never its scale. The realized address cloud is thereby pushed toward isotropic "
    "covariance on the command distribution the layer actually sees, so low-variance directions "
    "that carry file identity are not swamped by the high-variance verb/format directions, and two "
    "distinct contents receive addresses that a delta-rule memory can keep separate across a long "
    "chain of writes. The strength is held through warmup and hold and released to zero across the "
    "cosine decay, so the code geometry is shaped early and the loss owns the weights at the end. "
    "Inactive on archs with no identical-shape addressing pair."
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


class _AddressWhitening:

    def __init__(self, params, total_steps, rho=0.02, ema=0.98, shrink=0.05,
                 min_obs=25, max_in=1024, release_start=0.55, release_end=0.85):
        self.params = [p for p in params
                       if p.ndim == 2 and p.shape[0] <= p.shape[1]]
        self.rho = max(0.0, float(rho))
        self.ema = min(0.9999, max(0.0, float(ema)))
        self.shrink = min(1.0, max(0.0, float(shrink)))
        self.min_obs = max(1, int(min_obs))
        self.max_in = max(1, int(max_in))
        self.total = max(1, int(total_steps))
        rs = min(1.0, max(0.0, float(release_start)))
        re = min(1.0, max(rs + 1e-6, float(release_end)))
        self.hold_until = rs * self.total
        self.free_at = re * self.total
        self.seen = 0
        self._metric = {}
        self._count = {}

    def _strength(self):
        t = float(self.seen)
        if t <= self.hold_until:
            return self.rho
        if t >= self.free_at:
            return 0.0
        x = (t - self.hold_until) / max(1e-6, self.free_at - self.hold_until)
        return self.rho * 0.5 * (1.0 + math.cos(math.pi * x))

    @torch.no_grad()
    def observe(self):
        for p in self.params:
            g = p.grad
            if g is None or g.ndim != 2:
                continue
            n_in = int(g.shape[1])
            if n_in > self.max_in:
                continue
            if not torch.isfinite(g).all():
                continue
            gf = g.detach().float()
            mom = gf.t() @ gf
            tr = float(torch.diagonal(mom).sum())
            if not math.isfinite(tr) or tr <= 1e-20:
                continue
            mom = mom * (float(n_in) / tr)
            if not torch.isfinite(mom).all():
                continue
            key = id(p)
            cur = self._metric.get(key)
            if cur is None:
                self._metric[key] = mom
                self._count[key] = 1
            else:
                cur.mul_(self.ema).add_(mom, alpha=1.0 - self.ema)
                self._count[key] = self._count[key] + 1

    @torch.no_grad()
    def project(self):
        self.seen += 1
        rho = self._strength()
        if rho <= 0.0:
            return
        for p in self.params:
            w = p.data
            if w.ndim != 2 or not torch.isfinite(w).all():
                continue
            k, n_in = int(w.shape[0]), int(w.shape[1])
            wf = w.float()
            key = id(p)
            mom = self._metric.get(key)
            if mom is not None and self._count.get(key, 0) >= self.min_obs:
                eye_in = torch.eye(n_in, device=wf.device, dtype=wf.dtype)
                metric = mom * (1.0 - self.shrink) + eye_in * self.shrink
                wm = wf @ metric
            else:
                wm = wf
            gram = wm @ wf.t()
            c = float(torch.diagonal(gram).sum()) / float(k)
            if not math.isfinite(c) or c <= 1e-12:
                continue
            resid = gram / c - torch.eye(k, device=wf.device, dtype=wf.dtype)
            upd = (resid @ wm) / c
            un = float(upd.norm())
            wn = float(wf.norm())
            if not math.isfinite(un) or not math.isfinite(wn):
                continue
            if un <= 1e-12 or wn <= 1e-12:
                continue
            rel = un / wn
            step = rho if rel <= 1.0 else rho / rel
            upd = upd * step
            if not torch.isfinite(upd).all():
                continue
            w.add_(upd.to(w.dtype), alpha=-1.0)


class _WhitenOpt:

    def __init__(self, adamw, muon, cap, whiten):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.whiten = whiten

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
        if self.whiten is not None:
            self.whiten.observe()
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        self.cap.project()
        if self.whiten is not None:
            self.whiten.project()

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


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         whiten_rho=0.02, whiten_ema=0.98, whiten_shrink=0.05, whiten_min_obs=25,
         whiten_max_in=1024, whiten_release_start=0.55, whiten_release_end=0.85):
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

    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]

    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not keys and not dd:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    whiten = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))
        whiten = _AddressWhitening(
            keys, steps, rho=whiten_rho, ema=whiten_ema, shrink=whiten_shrink,
            min_obs=whiten_min_obs, max_in=whiten_max_in,
            release_start=whiten_release_start, release_end=whiten_release_end)
        if not whiten.params or whiten.rho <= 0.0:
            whiten = None

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else _NoCap()

    return _WhitenOpt(adamw, muon, cap, whiten), _MultiSched(*scheds)
