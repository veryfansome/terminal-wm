TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

THE CONTRACT — axis 'optim': Expose make(params, steps, **p) -> (optimizer, scheduler_or_None). The harness calls scheduler.step() after every optimizer step. Batch size is a genome field on this axis, not a constant in the impl.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any optim impl: expose make(params, steps, **p) -> (optimizer, scheduler_or_None).
The harness calls scheduler.step() after each opt.step() if the scheduler is not None. Batch size
is a genome field on this axis, not a constant in the impl."""
import torch
NAME = "baseline_adamw"
DESCRIPTION = "AdamW lr 3e-4, weight_decay 1e-4, constant LR."
def make(params, steps, lr=3e-4, wd=1e-4):
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd), None
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-14-pathstate-rssm
  its fitness       -0.0075   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r18_transition_forwardmodel_consistency

YOUR PARENT'S CURRENT optim IMPL — r18_spectral_capped_transition_readout (this is the code you are mutating):
--------------------------------------------------------------------------------
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
    dd_ids = {id(p) for p in dd}

    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _schedule_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not keys and not dd:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    scheds = [torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)]
    muon = None
    if keys:
        muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps, rms_match=rms_match)
        scheds.append(torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))

    if not dd:
        from types import SimpleNamespace
        cap = SimpleNamespace(project=lambda: None)
        return _CapOpt(adamw, muon, cap), _MultiSched(*scheds)

    cap = _SpectralCap(dd, cap=spectral_cap, iters=spectral_iters)
    return _CapOpt(adamw, muon, cap), _MultiSched(*scheds)
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca +0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca -0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r4_twotimescale_router_polyak_tail (axis optim)
import math

import torch

from evolve.chunks.optim.r18_spectral_capped_transition_readout import _MuonKeys, _SpectralCap

NAME = "r4_twotimescale_router_polyak_tail"
DESCRIPTION = (
    "AdamW + Muon on the delta-rule addressing pairs + the soft spectral cap on (D,D) readouts, "
    "re-timed as a two-timescale scheme with a Polyak-averaged tail. Two-dimensional weights whose "
    "input width is D and whose output width is at most router_out_max form their own parameter "
    "group: they are held at a constant multiple of the base learning rate for the whole run after "
    "warmup and carry no weight decay, while every other parameter warms up, holds, cosine-decays "
    "to a floor and then sits at that floor for the last avg_frac of the step budget. Across that "
    "constant-step tail the optimizer keeps a uniform running mean of every parameter and copies "
    "it into the live parameters on the final step, re-applying the spectral cap afterwards; the "
    "copy is skipped if any averaged tensor is non-finite."
)

D = 768


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


class _TwoTimescaleOpt:

    def __init__(self, adamw, muon, cap, tail):
        self.adamw = adamw
        self.muon = muon
        self.cap = cap
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
        if self.tail is not None and self.tail.observe():
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
         router_out_max=128, router_lr_mult=2.0, router_wd=0.0, avg_frac=0.25):
    params = [p for p in params]
    steps = max(1, int(steps))
    window = min(steps, max(1, int(round(float(avg_frac) * steps))))

    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
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
    tail = _PolyakTail(params, steps, window) if params else None

    return _TwoTimescaleOpt(adamw, muon, cap, tail), _MultiSched(*scheds)

--- r5_persistent_gradient_mixture_channel_opener (axis optim)
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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of -0.0075 (g0-14-pathstate-rssm, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
