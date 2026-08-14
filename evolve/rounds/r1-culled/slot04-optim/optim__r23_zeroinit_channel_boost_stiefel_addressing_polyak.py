import math
import torch

NAME = "r23_zeroinit_channel_boost_stiefel_addressing_polyak"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine, raised floor) + Muon on the (key_d x d) delta-rule addressing "
    "projections + a sync-free spectral cap on (D,D) readouts, PLUS three new mechanisms: "
    "(1) tensors that are exactly zero at construction are detected and split off — zero matrices "
    "get weight_decay 0 and a learning-rate multiplier, zero vectors get weight_decay 0 — because "
    "every gated injection channel of this arch family starts at exact zero and must grow inside "
    "a fixed step budget while decay pulls it back; (2) a Frobenius-preserving "
    "relaxed Bjorck pull of the addressing projections toward the Stiefel manifold, equalizing "
    "their singular values without changing their scale; (3) tail Polyak averaging merged into "
    "the live weights at the final step, paired with the raised LR floor."
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
                g = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
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


class _StiefelPull:

    def __init__(self, params, eta=0.08):
        self.params = [p for p in params if p.ndim == 2]
        self.eta = float(eta)

    @torch.no_grad()
    def project(self, strength):
        if strength <= 0.0 or self.eta <= 0.0:
            return
        step = 0.5 * self.eta * float(strength)
        for p in self.params:
            w = p.data
            flip = w.shape[0] > w.shape[1]
            m = w.mT if flip else w
            rows = m.shape[0]
            gram = m @ m.mT
            scale = torch.diagonal(gram).sum().div(rows).clamp_min(1e-12)
            pull = m - (gram / scale) @ m
            pull = torch.nan_to_num(pull, nan=0.0, posinf=0.0, neginf=0.0)
            before = m.norm()
            m.add_(pull, alpha=step)
            ratio = before / m.norm().clamp_min(1e-12)
            m.mul_(torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0))


class _SpectralCap:

    def __init__(self, params, cap=4.0, iters=2):
        self.params = list(params)
        self.cap = float(cap)
        self.iters = int(iters)
        self._u = {}

    @torch.no_grad()
    def project(self):
        for p in self.params:
            if p is None or p.ndim != 2:
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
            self._u[id(p)] = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = torch.dot(u, w.mv(v)).clamp_min(1e-8)
            factor = torch.clamp(self.cap / sigma, max=1.0)
            w.mul_(torch.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0))


class _CompositeOpt:

    def __init__(self, adamw, muon, stiefel, cap, warm):
        self.adamw = adamw
        self.muon = muon
        self.stiefel = stiefel
        self.cap = cap
        self.warm = int(warm)
        self._n = 0

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
        self._n += 1
        if self.stiefel is not None:
            self.stiefel.project(1.0 if self._n > self.warm else 0.0)
        if self.cap is not None:
            self.cap.project()

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict(), "n": self._n}
        if self.muon is not None:
            d["muon"] = self.muon.state_dict()
        return d


class _TailPolyak:

    def __init__(self, params, steps, start_frac, horizon_div):
        self.params = list(params)
        self.steps = max(1, int(steps))
        self.start = int(max(0.0, min(0.95, float(start_frac))) * self.steps)
        window = max(1, self.steps - self.start)
        self.beta = min(0.9999, max(0.9, 1.0 - float(horizon_div) / window))
        self.count = 0
        self.avg = None
        self.done = False

    @torch.no_grad()
    def step(self):
        if self.done:
            return
        self.count += 1
        if self.count <= self.start:
            return
        if self.avg is None:
            self.avg = [p.detach().clone() for p in self.params]
        else:
            for a, p in zip(self.avg, self.params):
                a.mul_(self.beta).add_(p.detach(), alpha=1.0 - self.beta)
        if self.count >= self.steps:
            for a, p in zip(self.avg, self.params):
                p.data.copy_(torch.where(torch.isfinite(a), a, p.data))
            self.avg = None
            self.done = True


class _MultiSched:
    def __init__(self, scheds, polyak):
        self.scheds = list(scheds)
        self.polyak = polyak

    def step(self):
        for s in self.scheds:
            s.step()
        if self.polyak is not None:
            self.polyak.step()

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


@torch.no_grad()
def _is_exactly_zero(p):
    return bool(p.numel() > 0 and torch.all(p == 0).item())


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.15, beta2=0.95, key_d=64, momentum=0.95, ns_steps=5,
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2, stiefel_eta=0.08,
         zero_lr_mult=2.0, polyak_start_frac=0.60, polyak_horizon_div=3.0,
         **unused):
    params = [p for p in params]
    steps = max(1, int(steps))

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
    zeroed = [p for p in rest if _is_exactly_zero(p)]
    grown = [p for p in rest if not _is_exactly_zero(p)]
    sprouting = [p for p in zeroed if p.ndim >= 2]
    flat_zero = [p for p in zeroed if p.ndim < 2]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)
    warm = max(20, int(warmup_frac * steps))

    groups = []
    if grown:
        groups.append({"params": grown, "lr": lr, "weight_decay": wd})
    if flat_zero:
        groups.append({"params": flat_zero, "lr": lr, "weight_decay": 0.0})
    if sprouting:
        groups.append({"params": sprouting, "lr": lr * float(zero_lr_mult),
                       "weight_decay": 0.0})
    if not groups:
        groups.append({"params": rest, "lr": lr, "weight_decay": wd})

    adamw = torch.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps,
                         rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    stiefel = _StiefelPull(keys, eta=stiefel_eta) if keys else None
    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else None
    polyak = _TailPolyak(params, steps, polyak_start_frac, polyak_horizon_div)

    if muon is None and stiefel is None and cap is None:
        return adamw, _MultiSched(scheds, polyak)

    return _CompositeOpt(adamw, muon, stiefel, cap, warm), _MultiSched(scheds, polyak)
