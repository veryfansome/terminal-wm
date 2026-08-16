import torch
import torch.nn.functional as F

NAME = "r7_command_stratified_conditional_likelihood"
DESCRIPTION = (
    "Conditional (command-stratified) partial likelihood. Rows are partitioned into COHORTS of "
    "numerically equal causal command embeddings; the primary term is a listwise likelihood whose "
    "denominator is restricted to the anchor's own cohort, so every candidate in it answers the "
    "identical question and any predictor that is a function of the command alone emits the same "
    "logit row for every member of the cohort and is therefore tied at the cohort floor — the "
    "training-side analogue of the way the metric cancels location-keying. Inside a cohort the "
    "distinct-target candidates that come from the anchor's OWN trajectory carry an additive logit "
    "bonus, so where a command was issued twice in one trajectory and returned different content "
    "the conditional problem collapses onto exactly that pair; rows whose answer already occurred "
    "earlier in their own trajectory, and rows that have such a same-trajectory alternative, carry "
    "extra mass. Exact-target equivalence classes are pooled multi-positively with inverse-class "
    "mass. A reduced-weight full-batch listwise term keeps the global geometry calibrated for the "
    "N-way forced choice, a cohort-confusable repulsion hinge enforces a margin in the squared-L2 "
    "decision variable, and an untouched raw squared-error anchor keeps a constant prediction from "
    "minimizing the loss. Temperature, margin and all equivalence thresholds are fractions of the "
    "batch's mean off-diagonal target distance, so the objective is invariant to the scale the "
    "target axis imposes."
)

WANTS_CTX = True

_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0
_NUM_EPS = 1e-12
_NEG = -1.0e4

_TEMP_FRAC = 0.125
_EQ_FRAC = 1e-3
_COH_FRAC = 1e-3

_SEG_LOGIT = 4.0
_RHO_RECOVER = 1.0
_RHO_MOVE = 4.0

_GAMMA = 1.0
_W_COND = 1.0
_W_GLOBAL = 0.3
_ANCHOR = 0.05
_W_REP = 0.1
_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125
_GEPS = 1e-3


def _pair_sq(x):
    sq = (x * x).sum(dim=1, keepdim=True)
    return (sq + sq.t() - 2.0 * (x @ x.t())).clamp_min(0.0) / float(x.shape[1])


def _cohort_mask(ctx, n, eye, dtype):
    if ctx is None or not hasattr(ctx, "get"):
        return None
    cmd = ctx.get("cmd")
    if cmd is None or cmd.dim() != 2 or cmd.shape[0] != n or cmd.shape[1] < 1:
        return None
    cmd = cmd.detach().to(dtype=torch.float32)
    if not torch.isfinite(cmd).all():
        return None
    cc = _pair_sq(cmd)
    pair_n = float(n * (n - 1))
    mean_c = (cc.masked_fill(eye, 0.0).sum() / pair_n).clamp_min(_NUM_EPS)
    return (cc <= _COH_FRAC * mean_c) | eye


def _segment_mask(ctx, n, device):
    if ctx is None or not hasattr(ctx, "get"):
        return torch.ones(n, n, dtype=torch.bool, device=device)
    prev = ctx.get("prev")
    if prev is None or prev.dim() != 2 or prev.shape[0] != n:
        return torch.ones(n, n, dtype=torch.bool, device=device)
    starts = prev.detach().abs().sum(dim=1) == 0
    if not bool(starts.any()):
        return torch.ones(n, n, dtype=torch.bool, device=device)
    seg = (torch.cumsum(starts.long(), dim=0) - 1).clamp_min(0)
    return seg.unsqueeze(1) == seg.unsqueeze(0)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    raw_anchor = ((pred - tgt) ** 2).mean()
    if n < 4:
        return raw_anchor

    device = pred.device
    dtype = pred.dtype

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(_NUM_EPS)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(_NUM_EPS)
        sw = w.sqrt().unsqueeze(0)

    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=device)
        pair_n = float(n * (n - 1))

        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.masked_fill(eye, 0.0).sum() / pair_n).clamp_min(_NUM_EPS)
        temp = (_TEMP_FRAC * mean_off).clamp_min(1e-8)
        margin = _MARGIN_FRAC * mean_off
        tau = (_TAU_FRAC * mean_off).clamp_min(1e-8)

        equivalent = (tt <= _EQ_FRAC * mean_off) | eye
        distinct = ~equivalent
        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=dtype)
        inv_class = class_size.reciprocal()
        log_class = -class_size.log().unsqueeze(0)

        cohort = _cohort_mask(ctx, n, eye, dtype)
        if cohort is None:
            cohort = eye.clone()
        same_seg = _segment_mask(ctx, n, device)

        cohort_neg = cohort & distinct
        has_contrast = cohort_neg.any(dim=1)
        seg_neg = cohort_neg & same_seg
        has_move_alt = seg_neg.any(dim=1)

        order = torch.arange(n, device=device)
        earlier = order.unsqueeze(1) > order.unsqueeze(0)
        recoverable = (equivalent & same_seg & earlier & (~eye)).any(dim=1)

        neg_bonus = torch.where(
            seg_neg, torch.full((), _SEG_LOGIT, device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype))

        row_cond = inv_class * has_contrast.to(dtype) * (
            1.0
            + _RHO_RECOVER * recoverable.to(dtype)
            + _RHO_MOVE * has_move_alt.to(dtype)
        )
        row_cond = row_cond / row_cond.sum().clamp_min(_NUM_EPS)

        row_glob = inv_class / inv_class.sum().clamp_min(_NUM_EPS)

        rep_mass = cohort_neg.to(dtype) * (1.0 + _SEG_LOGIT * seg_neg.to(dtype))
        rep_total = rep_mass.sum(dim=1)
        q = rep_mass / rep_total.clamp_min(1e-6).unsqueeze(1)
        rep_gate = rep_total / (rep_total + _GEPS)

    logits = -dist2 / temp + log_class

    cohort_logits = (logits + neg_bonus).masked_fill(~cohort, _NEG)
    log_den_c = torch.logsumexp(cohort_logits, dim=1)
    log_num_c = torch.logsumexp(cohort_logits.masked_fill(distinct, _NEG), dim=1)
    nll_c = (log_den_c - log_num_c).clamp_min(0.0)
    conditional = (row_cond * nll_c).sum()

    log_den_g = torch.logsumexp(logits, dim=1)
    log_num_g = torch.logsumexp(logits.masked_fill(distinct, _NEG), dim=1)
    nll_g = (log_den_g - log_num_g).clamp_min(0.0)
    with torch.no_grad():
        focal_g = (1.0 - (-nll_g).exp().clamp(0.0, 1.0)).pow(_GAMMA)
    global_listwise = (row_glob * focal_g * nll_g).sum()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep_row = rep_gate * F.softplus((d_true + margin - d_conf) / tau)
    repulsion = (row_glob * rep_row).sum()

    return (
        _W_COND * conditional
        + _W_GLOBAL * global_listwise
        + _ANCHOR * raw_anchor
        + _W_REP * repulsion
    )
