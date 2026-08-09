"""OPTIM chunk: the r8 Muon-on-addressing + AdamW/warmup-hold-cosine-floor stack, PLUS a soft
SPECTRAL-NORM CAP on the co-designed r18 arch's (D,D) latent-transition content READOUT
(`tr_read`), applied as a post-step projection.

CO-DESIGN (the epistasis stack this round tests). The arch
`r18_pathstate_latent_transition_worldmodel` is the ONLY channel that leaves the observation
manifold: it injects a per-path latent-transition correction into the prediction through a
square (D=768, D=768) readout `tr_read` (zero-init). Contrastive row-softmax is invariant to a
prediction's own norm, so this correction can grow off-manifold — the measured R10/R11 defect
(‖pred‖²/‖true‖² ≈ 4.7 on imagined listings). Capping the LARGEST SINGULAR VALUE of `tr_read`
bounds the gain of the injected correction, keeping predictions norm-calibrated and the per-path
recurrence dynamically stable, without touching the direction the operator learned.

WHY DISTINCT from the other optim proposals / the r8 stack:
  * Muon (r8) orthogonalizes the momentum of the (key_d×d) ADDRESSING matrices — equalizes
    ALL singular values of the key map for pattern separation. Kept here VERBATIM.
  * Shampoo (in-round #5) preconditions the ADDRESSING-matrix gradient with Kronecker curvature.
  * THIS caps only the TOP singular value of a DIFFERENT matrix (the content readout, not the
    addressing keys) toward a target, for norm-calibration / stability — a spectral-norm CONSTRAINT
    (Miyato et al., arXiv:1802.05957), not an orthogonalization or a curvature preconditioner.

STRICT SUPERSET OF THE r8 STACK. The cap group is routed by the UNIQUE (D,D) square signature — no
registered arch has a 768×768 weight except the co-designed arch's `tr_read`. On every other arch
the cap group is EMPTY and make() returns the exact r8 stack (Muon on addressing when present,
else plain AdamW), bit-identically. NaN-safe: non-finite weights skip the projection; the power
iteration is normalized by a clamped norm; no RNG.

Contract: make(params, steps, **kw) -> (optimizer, scheduler). opt.zero_grad / opt.step /
scheduler.step once per iteration. Pure/self-contained; torch only.
"""
import math
import torch

NAME = "r18_spectral_capped_transition_readout"
DESCRIPTION = (
    "AdamW(warmup-hold-cosine-floor) + Muon on the (key_d×d) delta-rule addressing "
    "projections, PLUS a soft spectral-norm cap on the co-designed r18 arch's (D,D) latent-"
    "transition content readout (post-step power-iteration projection of the top singular value "
    "to a target), for on-manifold norm calibration and recurrence stability. Unchanged on "
    "archs without a (D,D) readout."
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
    """Soft spectral-norm cap applied post-step to the routed (D,D) matrices. Persistent left
    singular estimate per param via 2 power iterations; if sigma > cap, scale the whole matrix by
    cap/sigma (a projection back onto the spectral-norm ball). NaN-safe; no RNG (u is deterministic
    from the first non-finite-free weight)."""

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


class _CapOpt:
    """Composite: AdamW (+ optional Muon) step, then the post-step spectral projection."""

    def __init__(self, adamw, muon, cap):
        self.adamw = adamw
        self.muon = muon
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
         rms_match=0.2, spectral_cap=4.0, spectral_iters=2):
    params = [p for p in params]

    # -- addressing keys -> Muon (exactly the r8 routing) --
    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}

    # -- (D,D) square transition readout(s) -> spectral cap (unique signature) --
    dd = [p for p in params if p.ndim == 2 and p.shape[0] == D and p.shape[1] == D]
    dd_ids = {id(p) for p in dd}

    rest = [p for p in params if id(p) not in key_ids]  # (D,D) stays trained by AdamW too

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not keys and not dd:  # exact carried baseline (plain AdamW)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    if not dd:  # keys present, no readout -> exact r8 Muon path
        from types import SimpleNamespace
        cap = SimpleNamespace(project=lambda: None)
        return _CapOpt(adamw, muon, cap), _MultiSched(*scheds)

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    return _CapOpt(adamw, muon, cap), _MultiSched(*scheds)
