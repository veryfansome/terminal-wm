import math

import torch

NAME = "r5_persistent_gradient_mixture_channel_opener"
DESCRIPTION = (
    "AdEMAMix-style two-timescale momentum in place of AdamW: each parameter carries a fast "
    "beta1 EMA and a second, much slower EMA whose horizon is scheduled from beta1 up to a "
    "beta3 end value capped at a quarter of the run length, and the update direction is "
    "(bias-corrected fast EMA + alpha * slow EMA) / sqrt(v), with alpha ramped linearly from "
    "zero. A gradient component that persists over hundreds of steps therefore receives up to "
    "(1 + alpha) times the step a transient component of the same instantaneous size receives. "
    "The same fast/slow mixture is formed before Newton-Schulz orthogonalization in the Muon "
    "branch that drives identical-shape (key_d x d) addressing projection pairs, where it "
    "rotates the orthogonalized direction toward the persistent component without changing the "
    "step size. Parameters that are exactly zero when the optimizer is built are split into "
    "their own groups with weight decay switched off and a cosine-decaying learning-rate "
    "multiplier (full multiplier for 2-D matrices, its square root for 1-D vectors) that "
    "returns to 1 after a configurable opening fraction of the run, so zero-initialised "
    "injection projections and gates leave zero early and the modules upstream of them receive "
    "gradient for most of the step budget. Post-step power-iteration spectral-norm capping of "
    "(D,D) matrices is retained. Warmup-hold-cosine-floor learning-rate schedule."
)

D = 768

_EPS_ORTH = 1e-7


def _ns_orth(g, steps=5, eps=_EPS_ORTH):
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


def _beta_slow_at(frac, b_start, b_end):
    ls = math.log(b_start)
    le = math.log(b_end)
    den = (1.0 - frac) * le + frac * ls
    if abs(den) < 1e-12:
        return b_end
    val = math.exp(ls * le / den)
    if not math.isfinite(val):
        return b_end
    return min(max(val, b_start), b_end)


class _MixtureAdam(torch.optim.Optimizer):

    def __init__(self, groups, lr, betas, eps, weight_decay, total_steps, alpha,
                 mix_warmup_frac, beta_slow_end):
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self.total_steps = max(1, int(total_steps))
        self.alpha = float(alpha)
        self.mix_warm = max(1, int(max(0.02, float(mix_warmup_frac)) * self.total_steps))
        self.beta_slow_end = float(beta_slow_end)
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_step += 1
        frac = min(1.0, self.global_step / float(self.mix_warm))
        alpha_t = self.alpha * frac
        for group in self.param_groups:
            b1, b2 = group["betas"]
            b3 = _beta_slow_at(frac, b1, self.beta_slow_end)
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["k"] = 0
                    st["m_fast"] = torch.zeros_like(p)
                    st["m_slow"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["k"] += 1
                k = st["k"]
                m_fast = st["m_fast"]
                m_slow = st["m_slow"]
                v = st["v"]
                m_fast.mul_(b1).add_(g, alpha=1.0 - b1)
                m_slow.mul_(b3).add_(g, alpha=1.0 - b3)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                bc1 = 1.0 - b1 ** k
                bc2 = 1.0 - b2 ** k
                denom = v.div(bc2).sqrt_().add_(eps)
                upd = m_fast.div(bc1).add_(m_slow, alpha=alpha_t)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(upd, denom, value=-lr)
        return loss


class _MixtureMuon(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum, ns_steps, rms_match, total_steps, alpha,
                 mix_warmup_frac, beta_slow_start, beta_slow_end):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                                      rms_match=rms_match))
        self.total_steps = max(1, int(total_steps))
        self.alpha = float(alpha)
        self.mix_warm = max(1, int(max(0.02, float(mix_warmup_frac)) * self.total_steps))
        self.beta_slow_start = float(beta_slow_start)
        self.beta_slow_end = float(beta_slow_end)
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_step += 1
        frac = min(1.0, self.global_step / float(self.mix_warm))
        alpha_t = self.alpha * frac
        b3 = _beta_slow_at(frac, self.beta_slow_start, self.beta_slow_end)
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
                    st["slow"] = torch.zeros_like(g)
                buf = st["buf"]
                slow = st["slow"]
                buf.mul_(mu).add_(g)
                slow.mul_(b3).add_(g, alpha=1.0 - b3)
                u = g.add(buf, alpha=mu).mul_(1.0 - mu).add_(slow, alpha=alpha_t)
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
            v = None
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            if v is None:
                continue
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _ComboOpt:

    def __init__(self, core, muon, cap):
        self.core = core
        self.muon = muon
        self.cap = cap

    @property
    def param_groups(self):
        groups = list(self.core.param_groups)
        if self.muon is not None:
            groups = groups + list(self.muon.param_groups)
        return groups

    def zero_grad(self, set_to_none=True):
        self.core.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.core.step()
        if self.muon is not None:
            self.muon.step()
        if self.cap is not None:
            self.cap.project()


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


def _opener_lambda(base_lambda, steps, open_frac, mult):
    n = max(1, int(max(0.0, float(open_frac)) * steps))
    m = max(1.0, float(mult))

    def lr_lambda(step):
        if step >= n or m <= 1.0:
            return base_lambda(step)
        p = step / float(n)
        boost = 1.0 + (m - 1.0) * 0.5 * (1.0 + math.cos(math.pi * p))
        return base_lambda(step) * boost

    return lr_lambda


@torch.no_grad()
def _all_zero(p):
    return bool(torch.count_nonzero(p.detach()).item() == 0)


def make(params, steps, lr=3e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30,
         floor_ratio=0.05, beta1=0.9, beta2=0.95, eps=1e-8, alpha=4.0,
         mix_warmup_frac=0.4, beta_slow_end=0.999, key_d=64, momentum=0.95,
         ns_steps=5, rms_match=0.2, spectral_cap=4.0, spectral_iters=2,
         open_mult=6.0, open_frac=0.15):
    params = [p for p in params]
    steps = max(1, int(steps))

    beta1 = min(max(float(beta1), 0.5), 0.99)
    beta_slow_end = float(beta_slow_end)
    beta_slow_end = min(beta_slow_end, 1.0 - 1.0 / max(50.0, 0.25 * steps))
    beta_slow_end = min(max(beta_slow_end, beta1 + 1e-4), 0.99999)

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
    cold2 = [p for p in rest if p.ndim == 2 and _all_zero(p)]
    cold1 = [p for p in rest if p.ndim == 1 and _all_zero(p)]
    cold_ids = {id(p) for p in cold2} | {id(p) for p in cold1}
    warm = [p for p in rest if id(p) not in cold_ids]

    base_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)
    open2 = _opener_lambda(base_lambda, steps, open_frac, open_mult)
    open1 = _opener_lambda(base_lambda, steps, open_frac, math.sqrt(max(1.0, open_mult)))

    groups = []
    lambdas = []
    if warm:
        groups.append({"params": warm, "weight_decay": wd})
        lambdas.append(base_lambda)
    if cold2:
        groups.append({"params": cold2, "weight_decay": 0.0})
        lambdas.append(open2)
    if cold1:
        groups.append({"params": cold1, "weight_decay": 0.0})
        lambdas.append(open1)
    if not groups:
        groups.append({"params": rest if rest else params, "weight_decay": wd})
        lambdas.append(base_lambda)

    core = _MixtureAdam(groups, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd,
                        total_steps=steps, alpha=alpha, mix_warmup_frac=mix_warmup_frac,
                        beta_slow_end=beta_slow_end)
    scheds = [torch.optim.lr_scheduler.LambdaLR(core, lambdas)]

    muon = None
    if keys:
        muon = _MixtureMuon(keys, lr=lr, momentum=momentum, ns_steps=ns_steps,
                            rms_match=rms_match, total_steps=steps, alpha=alpha,
                            mix_warmup_frac=mix_warmup_frac, beta_slow_start=beta1,
                            beta_slow_end=beta_slow_end)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, base_lambda))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters) if dd else None

    return _ComboOpt(core, muon, cap), _MultiSched(*scheds)
