TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

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
  id                r4-11-flatbasin-lsam-tailavg
  its fitness       +0.0225   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r22_observation_occlusion_denoising
  optim               r4_flatbasin_lsam_tailavg   params {"avg_frac": 0.25, "beta2": 0.95, "floor_ratio": 0.12, "hold_frac": 0.3, "key_d": 64, "lr": 0.0005, "momentum": 0.95, "ns_steps": 5, "rho": 0.05, "rms_match": 0.2, "spectral_cap": 4.0, "spectral_iters": 2, "warmup_frac": 0.04, "wd": 0.0005}
  target              identity
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r4_registry_transport_composite

YOUR PARENT'S CURRENT optim IMPL — r4_flatbasin_lsam_tailavg (this is the code you are mutating):
--------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

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

--- r18_spectral_capped_transition_readout (axis optim)
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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0225 (r4-11-flatbasin-lsam-tailavg, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
