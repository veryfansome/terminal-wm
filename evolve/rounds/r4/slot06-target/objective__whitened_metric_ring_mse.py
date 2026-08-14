import torch
import torch.nn.functional as F

NAME = "whitened_metric_ring_mse"
DESCRIPTION = (
    "A single-metric recombination of the confusability-ring listwise contrastive and plain mean "
    "squared error. The listwise term keeps the ring re-weighting of in-batch negatives on "
    "target-target distances — near-duplicate targets get about zero weight, close-but-distinct "
    "targets get up to (1+kappa)x weight — together with the focal down-weighting of already-solved "
    "rows and the gated repulsion hinge on the squared-L2 decision variable. What it drops is the "
    "per-dimension precision re-weighting by inverse residual error: every distance in this loss is "
    "the plain isotropic squared L2 of the target space it is handed, so the geometry of the loss is "
    "whatever metric the target axis defines and nothing else re-weights it. The squared-error term "
    "is a co-equal summand rather than a small anchor, so the minimiser stays the conditional mean "
    "in that same metric."
)

_TEMP = 0.25
_GAMMA = 1.0
_MSE_W = 1.0
_LIST_W = 1.0
_EPS = 1e-2

_KAPPA = 4.0
_DELTA = 0.05
_LAM_FRAC = 0.5
_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25
_GEPS = 1e-3


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    p_sq = (pred * pred).sum(dim=1, keepdim=True)
    t_sq = (tgt * tgt).sum(dim=1, keepdim=True)
    dist2 = (p_sq + t_sq.t() - 2.0 * (pred @ tgt.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (t_sq + t_sq.t() - 2.0 * (tgt @ tgt.t())).clamp_min(0.0) / float(d)
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

    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    return _LIST_W * listwise + _MSE_W * mse_anchor + _LAMBDA_REP * rep
