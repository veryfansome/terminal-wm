"""OPTIM chunk: incumbent warmup-hold-cosine-floor AdamW everywhere, EXCEPT the champion
fastweight arch's four delta-rule ADDRESSING projections (content_read/content_write/
path_read/path_write: the only (key_d=64, d) matrices in the net), which get a Muon-style
orthogonalized-momentum update (Newton-Schulz polar factor of the momentum, RMS-matched to
AdamW so the incumbent peak LR and schedule carry over unchanged).

WHY THIS CAN RAISE THE MARGIN (fastweights-aware, not generic)
--------------------------------------------------------------
The champion arch (r7_path_delta_fastweights_codex) is an online delta-rule associative
memory: it stores each (cmd, obs) pair as  mem <- decay*mem + beta * k (v - k^T mem)^T  with
unit key k = unit(W x), and reads target-space predictions back as k_q^T mem. A linear
associative memory's capacity and crosstalk are governed entirely by the GEOMETRY OF THE
KEYS: retrieval interference between stored pairs i,j scales with |k_i . k_j|, and the
memory can hold at most key_d well-separated directions (Schlag, Irie & Schmidhuber 2021,
"Linear Transformers Are Secretly Fast Weight Programmers", arXiv:2102.11174 - the delta
rule exists precisely to manage this interference). All keys live in the row space of the
four addressing matrices W in R^{key_d x d}. If W's spectrum collapses toward a few
dominant directions - which elementwise AdamW does nothing to prevent on a small over-
epoched dataset - distinct commands/paths hash to overlapping addresses, past outcomes
smear together, and exactly the mechanism that carries the held-out margin (the target-
space read) degrades on unseen systems.

Muon (Jordan et al. 2024; Liu et al. 2025, arXiv:2502.16982) replaces the update for a
matrix param with the polar factor of its momentum: every singular direction of the update
gets equal magnitude. Applied ONLY to the addressing matrices, this is pattern separation
implemented in the optimizer: learning pressure is spread across ALL key_d addressing
directions instead of amplifying the dominant few, keeping the key map well-conditioned /
full-rank so the two memories keep key_d usable slots. The trunk, gates, memory-decay
scalar, and head keep the exact incumbent AdamW + schedule (the proven config) - so this
complements, rather than perturbs, everything already selected.

MECHANISM (per addressing matrix, per step)
  buf   <- mu*buf + g                 (momentum, mu=0.95)
  u     <- g + mu*buf                 (nesterov)
  O     <- NewtonSchulz5(u)           (approx polar factor: semi-orthogonal, sv ~ 1)
  W     <- W - lr_t * 0.2*sqrt(max(n,m)) * O
The 0.2*sqrt(max(n,m)) factor RMS-matches AdamW's update (arXiv:2502.16982), so the SAME
peak lr (5e-4) and the SAME warmup(4%)-hold(30%)-cosine-to-floor(peak/20) schedule drive
both groups (one LambdaLR shape, two optimizers). Weight decay on the addressing group is
0: keys are unit-normalized after projection, so the function is invariant to the scale of
W - decay there would only shrink the matrix and silently inflate the relative step size.

ROUTING + SAFETY
  Params are routed by shape signature: ndim==2, shape==(key_d, in) with in != key_d and
  in != 768, AND at least 2 candidate tensors share that exact shape - delta-rule addressing
  projections always come as read/write SIBLINGS of identical shape, while the one shape
  collision in the registry (baseline transformer's pos_emb.weight, an Embedding(64, d)
  table) is a singleton. Verified against every arch impl in the registry: on the champion
  (d=176, key_d=64) this matches EXACTLY {content_read, content_write, path_read,
  path_write}.weight and nothing else (GRU cells are (528,176), FFN (352,176)/(176,352),
  projections (176,768), head (768,176), gates (1,352)/(2,176), norms/biases 1-D, scalars
  0-D); on every non-fastweights arch (baseline/hippo/mv/recency/...) the Muon group is
  EMPTY and make() returns the EXACT incumbent (torch.optim.AdamW + the same LambdaLR) -
  bit-identical behavior, strictly generalizing go_warmup_holdcos_floor.
  NaN-safe: non-finite grads skip that tensor's update; Newton-Schulz normalizes by a
  clamped Frobenius norm. No RNG anywhere -> deterministic given the harness seed. The
  optimizer only consumes gradients - it cannot touch the eval, causality, or the loss, and
  a constant prediction still cannot minimize the contrastive objective (anti-collapse
  unaffected).

Contract: make(params, steps, **kw) -> (optimizer, scheduler). The harness calls
opt.zero_grad(set_to_none=True), opt.step(), then scheduler.step() once per iteration.
Pure/self-contained; torch only; no file/network/global state.
"""
import math
import torch

NAME = "r8_muon_key_orthogonal_addressing"
DESCRIPTION = ("Incumbent AdamW(5e-4, wd 5e-4, b2 .95; 4% warmup, 30% hold, cos-to-floor) "
               "plus a Muon (Newton-Schulz orthogonalized-momentum, RMS-matched, wd 0) group "
               "scoped by shape signature to the fastweight arch's (key_d x d) delta-rule "
               "addressing projections; exact incumbent on archs without them.")

D = 768  # frozen encoder dim; addressing matrices never touch it on either side


def _ns_orth(g, steps=5, eps=1e-7):
    """Approximate polar factor of g via the quintic Newton-Schulz iteration (Muon's
    coefficients). Works for either orientation; returns a semi-orthogonal matrix with
    singular values ~1. Float32 internally; NaN-safe via the clamped norm."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        s = x @ x.mT                      # (n, n) with n = min(rows, cols)
        y = b * s + c * (s @ s)
        x = a * x + y @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class _MuonKeys(torch.optim.Optimizer):
    """Muon for the addressing matrices only. group['lr'] is driven by LambdaLR on the
    shared incumbent schedule; the update is RMS-matched to AdamW via 0.2*sqrt(max(n,m))
    (arXiv:2502.16982), so peak lr 5e-4 transfers. wd=0 (keys are scale-invariant)."""

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
                    continue  # NaN-safe: skip this tensor this step
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu)               # nesterov momentum
                o = _ns_orth(u, steps=ns)
                scale = rms * math.sqrt(max(p.shape[0], p.shape[1]))
                p.add_(o, alpha=-lr * scale)
        return loss


class _TwoOpt:
    """Minimal composite exposing the surface the harness uses (zero_grad / step /
    param_groups). Only instantiated when a Muon group exists."""

    def __init__(self, adamw, muon):
        self.adamw = adamw
        self.muon = muon

    @property
    def param_groups(self):
        return list(self.adamw.param_groups) + list(self.muon.param_groups)

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.adamw.step()
        self.muon.step()

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}


class _TwoSched:
    """Steps both LambdaLRs (identical schedule shape, per-optimizer peak lr)."""

    def __init__(self, *scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]


def _incumbent_lambda(steps, warmup_frac, hold_frac, floor_ratio):
    """Exactly go_warmup_holdcos_floor's multiplier: warmup(4%, min 20) -> hold(30%) ->
    cosine to floor_ratio (never 0)."""
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
         rms_match=0.2):
    params = [p for p in params]
    cand = [p for p in params
            if p.ndim == 2 and p.shape[0] == key_d
            and p.shape[1] != key_d and p.shape[1] != D]
    # addressing projections come as identical-shape read/write siblings (>=2); a lone
    # matching tensor (e.g. an Embedding(64, d) positional table) is NOT addressing.
    shape_counts = {}
    for p in cand:
        shape_counts[tuple(p.shape)] = shape_counts.get(tuple(p.shape), 0) + 1
    keys = [p for p in cand if shape_counts[tuple(p.shape)] >= 2]
    key_ids = {id(p) for p in keys}
    rest = [p for p in params if id(p) not in key_ids]

    lr_lambda = _incumbent_lambda(steps, warmup_frac, hold_frac, floor_ratio)

    if not keys:  # no addressing matrices (other archs) -> EXACT incumbent
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, beta2))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    adamw = torch.optim.AdamW(rest, lr=lr, weight_decay=wd, betas=(0.9, beta2))
    muon = _MuonKeys(keys, lr=lr, momentum=momentum, ns_steps=ns_steps,
                     rms_match=rms_match)
    sched = _TwoSched(torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda),
                      torch.optim.lr_scheduler.LambdaLR(muon, lr_lambda))
    return _TwoOpt(adamw, muon), sched
