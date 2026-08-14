import torch
import torch.nn.functional as F

NAME = "r4_dupclass_ring_negatives"
DESCRIPTION = (
    "The precision-weighted focal listwise L2 contrastive with confusability-ring negative "
    "reweighting and the gated repulsion hinge, computed over EQUIVALENCE CLASSES of coincident "
    "targets instead of over rows: targets whose pairwise weighted distance is below a tiny "
    "numerical threshold form one class, every column's logit is divided by its class "
    "multiplicity, and each row's numerator is the log-sum-exp over its whole class. A batch "
    "region of exactly-repeated next-observations therefore counts once as a negative and is "
    "fully attainable as a positive. Reduces term-by-term to the un-classed loss when every "
    "target in the batch is distinct."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_KAPPA = 4.0
_DELTA = 0.05
_LAM_FRAC = 0.5
_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25
_GEPS = 1e-3

_DUP_TT = 1e-3
_NEG_INF = -1e9


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        sw = w.sqrt().unsqueeze(0)

    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())
    dist2 = dist2.clamp_min(0.0) / float(d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)
        dupmask = 1.0 - torch.exp(-tt / _DELTA)
        ring = (confus * dupmask).masked_fill(eye, 0.0)

        a_raw = 1.0 + _KAPPA * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0)
        log_a = a.clamp_min(1e-6).log()

        mass = ring.sum(dim=1)
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GEPS)

        coincident = (tt < _DUP_TT) | eye
        log_a = log_a.masked_fill(coincident, 0.0)
        mult = coincident.sum(dim=0).clamp_min(1).to(dist2.dtype)
        log_mult = mult.log().unsqueeze(0)
        log_coincident = torch.zeros_like(tt).masked_fill(~coincident, _NEG_INF)

    logits = -dist2 / _TEMP + log_a - log_mult
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits + log_coincident, dim=1)
    nll = (log_den - log_num).clamp_min(0.0)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep
