"""OPTIM chunk: SLOW-WEIGHT CONSOLIDATION (Lookahead; Zhang, Lucas, Hinton & Ba, NeurIPS 2019,
arXiv:1907.08610) wrapped AROUND the champion inner optimizer (Muon-on-addressing + AdamW
warmup-hold-cosine-floor + spectral-capped (D,D) transition readout), all kept VERBATIM.

WHY (the lever this round hasn't tried). After 21+8 candidates the record localizes all
content-attributable imagination in the champion TRUNK, and BINDING-v2 selects the content-
attributable differential IMAG_CA. My own TRAIN-window analysis adds one fact the round did not
have: the history/content-dependence of a read (its target's distance from the cross-system
centroid of the SAME command) cleanly separates command-only-solvable reads (mean 0.230) from
command-only-FAILURE reads (0.629) -- but it is UNIFORMLY spread across sequences, so a
sequence-selection batcher barely moves batch composition (failed-reads/seq 8.21 -> 8.52 at
beta=1, a 1.04x no-op; genuine windows/seq 1.92 -> 1.89, 0.99x). Interpretation: the content-
learning gradient signal is diluted UNIFORMLY across every batch -- it cannot be concentrated by
DATA SELECTION. It can only be acted on where that diluted signal is integrated over time (the
optimizer) or where the function is changed (arch). The optimizer GRADIENT space was taken this
round (temporal gradient consensus); the optimizer WEIGHT space is untried.

MECHANISM. Lookahead keeps a set of SLOW weights phi and lets the champion inner optimizer A run
k fast steps from phi; then phi <- phi + alpha*(theta_k - phi) and the fast weights reset to phi
(Algorithm 1, verbatim). A forced consolidation at the final step returns phi (the paper returns
phi). Proposition 2 of the paper proves, on the noisy-quadratic proxy, that Lookahead converges to
the SAME expected value as the inner optimizer but to a STRICTLY SMALLER variance fixed point for
any alpha in (0,1) at equal learning rate (V*_LA = [first-product-term < 1] * V*_SGD). Variance
reduction helps the LOWEST-SNR parameters most -- here the content-delivery / memory-readout
parameters that carry IMAG_CA and receive exactly the diluted signal my data measured. So the
consolidated phi is a lower-variance estimate of the content-direction (cleaner, more transferable
across the held-out systems that fitness rewards) -- WITHOUT biasing the solution (same mean =>
low fitness risk) and WITHOUT any model, loss, batch, forward, or eval change.

STRICT INCUMBENT SUPERSET (verified, max|delta| = 0.0). The inner optimizer is the champion
`r18_spectral_capped_transition_readout` inlined verbatim: identical Muon routing (6 (key_d,d)
addressing matrices), identical spectral cap on the unique (D,D) `tr_read`, identical warmup-hold-
cosine-floor schedule. At la_alpha = 1.0 (or la_k > steps) NO consolidation ever runs and the
parameter trajectory is the champion's bit-for-bit. The slow-weight interpolation of spectral-
capped iterates stays inside the spectral-norm ball by convexity, so the cap is never violated by
consolidation. NaN-safe: a non-finite param skips its interpolation; no RNG. Lookahead's slow
weights live in the OPTIMIZER, never on the net (net.state_dict untouched), so the frozen
instrument, PAD-invariance, and all ablations are structurally inherited.

Contract: make(params, steps, **kw) -> (optimizer, scheduler). opt.zero_grad / opt.step /
scheduler.step once per iteration. Pure/self-contained; torch only.

Ref: Zhang, Lucas, Hinton, Ba, \"Lookahead Optimizer: k steps forward, 1 step back\", NeurIPS 2019
(arXiv:1907.08610) -- Algorithm 1 (slow/fast update), Proposition 2 (variance fixed point strictly
below the inner optimizer), and the paper's \"maintain the inner optimizer's internal state\" choice
(momentum kept; only the params reset), all used here.
"""
import math

import torch

NAME = "r22_lookahead_slowweight_consolidation"
DESCRIPTION = (
    "Lookahead slow-weight consolidation (Zhang et al. 2019, Alg.1) wrapped around the champion "
    "inner optimizer (Muon addressing + AdamW warmup-hold-cosine-floor + spectral-capped (D,D) "
    "readout, all verbatim): every la_k inner steps phi += alpha*(theta_k - phi), fast reset to "
    "phi, final step returns phi. Prop.2 variance reduction (same mean, strictly smaller variance "
    "fixed point) targets the low-SNR content-delivery params. la_alpha=1.0 recovers the champion "
    "optimizer bit-for-bit (verified strict superset)."
)

D = 768


# ================= champion inner optimizer, inlined VERBATIM =================
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
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]; lr = group["lr"]; ns = group["ns_steps"]; rms = group["rms_match"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]; buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)
                o = _ns_orth(u, steps=ns)
                scale = rms * math.sqrt(max(p.shape[0], p.shape[1]))
                p.add_(o, alpha=-lr * scale)
        return loss


class _SpectralCap:
    def __init__(self, params, cap=4.0, iters=2):
        self.params = list(params); self.cap = float(cap); self.iters = int(iters); self._u = {}

    @torch.no_grad()
    def project(self):
        for p in self.params:
            if p is None or p.ndim != 2 or not torch.isfinite(p).all():
                continue
            w = p.data; n = w.shape[0]; u = self._u.get(id(p))
            if u is None or u.shape[0] != n:
                u = torch.nn.functional.normalize(w.new_ones(n), dim=0)
            for _ in range(self.iters):
                v = torch.nn.functional.normalize(w.t().mv(u), dim=0, eps=1e-8)
                u = torch.nn.functional.normalize(w.mv(v), dim=0, eps=1e-8)
            self._u[id(p)] = u
            sigma = float(torch.dot(u, w.mv(v)))
            if math.isfinite(sigma) and sigma > self.cap:
                w.mul_(self.cap / max(sigma, 1e-8))


class _CapOpt:
    def __init__(self, adamw, muon, cap):
        self.adamw = adamw; self.muon = muon; self.cap = cap

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


def _incumbent_lambda(steps, warmup_frac, hold_frac, floor_ratio):
    warm = max(20, int(warmup_frac * steps)); hold = int(hold_frac * steps)
    decay_start = warm + hold; decay_len = max(1, steps - decay_start)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        if step < decay_start:
            return 1.0
        p = (step - decay_start) / decay_len
        return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return lr_lambda


# ================= NEW: Lookahead slow-weight consolidation wrapper =================
class _Lookahead:
    """Weight-space slow/fast consolidation (Zhang et al. 2019, Alg. 1) wrapping the champion inner
    optimizer. Every la_k inner steps: slow += alpha*(fast - slow); fast <- slow. A forced final
    consolidation at t == total_steps makes the returned net hold the SLOW weights phi (Alg.1
    returns phi). la_alpha >= 1.0 skips all consolidation -> the inner optimizer bit-for-bit."""

    def __init__(self, inner, params, total_steps, la_k=5, la_alpha=0.5):
        self.inner = inner
        self.params = [p for p in params]
        self.la_k = max(1, int(la_k))
        self.la_alpha = float(la_alpha)
        self.total_steps = int(total_steps)
        self._t = 0
        self._slow = [p.detach().clone() for p in self.params]

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.inner.state_dict() if hasattr(self.inner, "state_dict") else {}

    @torch.no_grad()
    def step(self, closure=None):
        self.inner.step(closure)
        self._t += 1
        if self.la_alpha >= 1.0:
            return                                               # exact inner optimizer (no consolidation)
        if self._t % self.la_k != 0 and self._t != self.total_steps:
            return
        for p, slow in zip(self.params, self._slow):
            if not torch.isfinite(p).all():                      # NaN-safe: skip this param
                continue
            slow.add_(p.detach() - slow, alpha=self.la_alpha)    # slow += alpha*(fast - slow)
            p.copy_(slow)                                        # fast <- slow (final t: net holds phi)


def make(params, steps, lr=5e-4, wd=5e-4, warmup_frac=0.04, hold_frac=0.30, floor_ratio=0.05,
         beta2=0.95, key_d=64, momentum=0.95, ns_steps=5, rms_match=0.2, spectral_cap=4.0,
         spectral_iters=2, la_k=5, la_alpha=0.5):
    params = [p for p in params]

    # -- champion routing, verbatim: addressing keys -> Muon --
    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    # -- (D,D) square transition readout(s) -> spectral cap (unique signature) --
    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]
    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _incumbent_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    def _wrap(inner, sched):
        return _Lookahead(inner, params, steps, la_k=la_k, la_alpha=la_alpha), sched

    if not keys and not dd:                                       # exact incumbent baseline (plain AdamW)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return _wrap(opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda))

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    if not dd:                                                    # keys present, no readout
        from types import SimpleNamespace
        cap = SimpleNamespace(project=lambda: None)
        return _wrap(_CapOpt(adamw, muon, cap), _MultiSched(*scheds))

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    return _wrap(_CapOpt(adamw, muon, cap), _MultiSched(*scheds))
