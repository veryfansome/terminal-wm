import torch
import torch.nn.functional as F

NAME = "free_energy_precision_l2_contrastive"
DESCRIPTION = (
    "Row-only (metric-direction) listwise L2 contrastive loss whose per-dim-mean squared distances "
    "are PRECISION-WEIGHTED: each embedding dimension is scaled by a detached, tempered, mean-1 "
    "precision Pi_d = 1/MSE_d (Friston/Rao-Ballard inverse-error-variance), denoising the same-verb "
    "ranking by down-weighting unpredictable dimensions. Detached focal top-1 reweighting + small MSE "
    "anchor for absolute placement and anti-collapse."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        prec = 1.0 / (mse_d + _EPS)
        w = prec.pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        sw = w.sqrt()
        sw = sw.unsqueeze(0)

    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())
    dist2 = dist2.clamp_min(0.0) / float(d)

    logits = -dist2 / _TEMP
    labels = torch.arange(n, device=pred.device)

    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)

    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)

    listwise = (focal * nll).mean()
    return listwise + _ANCHOR * mse_anchor
