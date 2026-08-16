import torch
import torch.nn.functional as F

NAME = "r7_matched_command_conditional_likelihood"
DESCRIPTION = (
    "Stratified conditional likelihood on matched command sets. The command embedding is used as "
    "a STRATIFICATION variable rather than a regression basis: rows whose causal command embedding "
    "is equal up to a relative floor of the batch's mean command distance form one anchor's risk "
    "set, and a second listwise term is formed whose softmax denominator is restricted to that risk "
    "set. Inside a risk set the read location is held fixed, so any predictor that is a function of "
    "the command alone scores every candidate identically and sits exactly at the conditional "
    "chance level; all gradient from that term is therefore carried by which content the chain of "
    "moves actually routed to the location. Exact-duplicate targets are pooled into one "
    "multi-positive class with inverse-class-size mass, both globally and inside each risk set, so "
    "a stratum whose contents happen to coincide asks for no discrimination, and the conditional "
    "temperature is a fraction of the mean WITHIN-risk-set target distance rather than of the "
    "global one, since same-location contents are far closer to each other than the batch average. "
    "The global confusability-ring listwise term, the gated repulsion margin in the eval's own "
    "squared-L2 decision variable, and an untouched raw squared-error anchor that keeps a constant "
    "prediction from minimizing the loss are retained unchanged."
)

WANTS_CTX = True

_TEMP_FRAC = 0.125
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_KAPPA = 4.0
_LAM_FRAC = 0.5
_DUP_FRAC = 0.025
_EQ_FRAC = 1e-3
_LAMBDA_REP = 0.1
_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125
_GEPS = 1e-3

_LAMBDA_COND = 1.0
_CMD_EQ_FRAC = 1e-4
_COND_TEMP_FRAC = 0.25
_MIN_MATCHED_ROWS = 8

_NUM_EPS = 1e-12


@torch.no_grad()
def _matched_command_sets(cmd, n, eye):
    if not torch.is_tensor(cmd) or cmd.dim() != 2 or cmd.shape[0] != n:
        return None
    c = cmd.detach().to(dtype=torch.float32)
    if not torch.isfinite(c).all():
        return None
    csq = (c * c).sum(dim=1, keepdim=True)
    cd2 = (csq + csq.t() - 2.0 * (c @ c.t())).clamp_min(0.0)
    mean_cd = cd2.sum() / float(n * (n - 1))
    if not bool(torch.isfinite(mean_cd)):
        return None
    thr = (_CMD_EQ_FRAC * mean_cd).clamp_min(_NUM_EPS)
    same = (cd2 <= thr) | eye
    del cd2
    return same


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    raw_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return raw_anchor

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
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        pair_n = float(n * (n - 1))

        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / pair_n).clamp_min(_EPS)

        temp = (_TEMP_FRAC * mean_off).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        dup_scale = (_DUP_FRAC * mean_off).clamp_min(_NUM_EPS)
        margin = _MARGIN_FRAC * mean_off
        tau = (_TAU_FRAC * mean_off).clamp_min(_EPS)

        equivalent = (tt <= _EQ_FRAC * mean_off) | eye
        distinct = ~equivalent
        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()

        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / dup_scale))
        ring = ring.masked_fill(equivalent, 0.0)

        neg_mass = distinct.to(dtype=pred.dtype)
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_mass).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log()
        log_class = -class_size.clamp_min(1.0).log().unsqueeze(0)

        mass = ring.sum(dim=1)
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GEPS)

        row_w = inv_class / inv_class.mean().clamp_min(_NUM_EPS)

    logits = -dist2 / temp + log_importance + log_class
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(distinct, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    listwise = (row_w * focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (row_w * gate * F.softplus((d_true + margin - d_conf) / tau)).mean()

    total = listwise + _ANCHOR * raw_anchor + _LAMBDA_REP * rep

    same = None
    if ctx is not None and hasattr(ctx, "get"):
        same = _matched_command_sets(ctx.get("cmd"), n, eye)
    if same is None:
        return total

    with torch.no_grad():
        matched_neg = same & distinct
        eligible = matched_neg.any(dim=1)
        n_eligible = int(eligible.sum())
    if n_eligible < _MIN_MATCHED_ROWS:
        return total

    with torch.no_grad():
        neg_pairs = matched_neg.to(dtype=pred.dtype)
        mean_in = (tt * neg_pairs).sum() / neg_pairs.sum().clamp_min(1.0)
        temp_c = (_COND_TEMP_FRAC * mean_in).clamp_min(_EPS)
        matched_pos = same & equivalent
        row_mass = inv_class * eligible.to(dtype=pred.dtype)
        row_measure = row_mass.sum().clamp_min(_NUM_EPS)

    logits_c = (-dist2 / temp_c + log_class).masked_fill(~same, float("-inf"))
    log_den_c = torch.logsumexp(logits_c, dim=1)
    log_num_c = torch.logsumexp(logits_c.masked_fill(~matched_pos, float("-inf")), dim=1)
    nll_c = (log_den_c - log_num_c).clamp_min(0.0)

    with torch.no_grad():
        focal_c = (1.0 - (-nll_c).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    conditional = (row_mass * focal_c * nll_c).sum() / row_measure

    return total + _LAMBDA_COND * conditional
