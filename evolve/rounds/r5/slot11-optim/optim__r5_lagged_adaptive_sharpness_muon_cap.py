import math
import torch

NAME = "r5_lagged_adaptive_sharpness_muon_cap"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on identical-shape (key_d x d) addressing projection "
    "pairs + a soft spectral-norm cap on (D,D) readouts, wrapped in a ZERO-EXTRA-BACKWARD adaptive "
    "sharpness step. After each update the parameters are displaced along an elementwise "
    "|theta|-scaled, globally normalized ascent direction (an exponential moving average of the "
    "gradient), so the NEXT minibatch's forward and backward are evaluated at the displaced point; "
    "the displacement is subtracted again before that gradient is applied, making the update a "
    "descent step at the base point using a gradient taken at a nearby ascent point. The "
    "displacement is ramped in after warmup and is skipped on the final step, so training ends and "
    "evaluation runs on undisplaced weights. Reduces to the plain AdamW+Muon+cap optimizer when "
    "rho is 0."
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


class _AdaptiveSharpnessShift:

    def __init__(self, params, rho=0.5, eta=0.01, beta=0.6):
        self.params = [p for p in params if p.requires_grad]
        self.rho = float(rho)
        self.eta = float(eta)
        self.beta = min(max(float(beta), 0.0), 0.999)
        self.direction = {}
        self.displacement = {}

    @torch.no_grad()
    def revert(self):
        if not self.displacement:
            return
        for p in self.params:
            e = self.displacement.pop(id(p), None)
            if e is not None:
                p.data.sub_(e)
        self.displacement = {}

    @torch.no_grad()
    def displace(self, magnitude):
        if magnitude <= 0.0 or self.rho <= 0.0:
            return
        staged = []
        total = None
        for p in self.params:
            g = p.grad
            if g is None:
                continue
            if not torch.isfinite(g).all():
                continue
            d = self.direction.get(id(p))
            if d is None or d.shape != g.shape:
                d = torch.zeros_like(p.data)
                self.direction[id(p)] = d
            d.mul_(self.beta).add_(g, alpha=1.0 - self.beta)
            t = p.data.abs().add(self.eta)
            scaled = t * d
            staged.append((p, t * scaled))
            piece = scaled.pow(2).sum()
            total = piece if total is None else total + piece
        if total is None or not staged:
            return
        norm = float(total.sqrt())
        if not math.isfinite(norm) or norm <= 1e-12:
            return
        coef = magnitude / norm
        for p, raw in staged:
            e = raw.mul_(coef)
            if not torch.isfinite(e).all():
                continue
            p.data.add_(e)
            self.displacement[id(p)] = e


class _SharpCapOpt:

    def __init__(self, adamw, muon, cap, shift, total_steps, warm, ramp_len, rho):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
        self.shift = shift
        self.total_steps = max(1, int(total_steps))
        self.warm = int(warm)
        self.ramp_len = max(1, int(ramp_len))
        self.rho = float(rho)
        self.taken = 0

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

    def _magnitude(self):
        x = (self.taken - self.warm) / float(self.ramp_len)
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return self.rho
        return self.rho * (x * x * (3.0 - 2.0 * x))

    def step(self, closure=None):
        self.shift.revert()
        self.adamw.step()
        if self.muon is not None:
            self.muon.step()
        self.cap.project()
        self.taken += 1
        if self.taken < self.total_steps:
            self.shift.displace(self._magnitude())

    def state_dict(self):
        d = {"adamw": self.adamw.state_dict(), "taken": self.taken}
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
         rho=0.5, sharp_eta=0.01, sharp_beta=0.6, sharp_ramp_frac=0.10):
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
    warm = max(20, int(warmup_frac * steps))
    ramp_len = max(1, int(sharp_ramp_frac * steps))

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]

    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    if dd:
        cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    else:
        from types import SimpleNamespace
        cap = SimpleNamespace(project=lambda: None)

    shift = _AdaptiveSharpnessShift(params, rho=rho, eta=sharp_eta, beta=sharp_beta)
    opt = _SharpCapOpt(adamw, muon, cap, shift, steps, warm, ramp_len, rho)
    return opt, _MultiSched(*scheds)
