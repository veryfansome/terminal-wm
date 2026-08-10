import torch
import torch.nn.functional as F

NAME = "r17_mutation_counterfactual_twin_margin"
DESCRIPTION = (
    "The free-energy precision-weighted focal-listwise L2 contrastive backbone PLUS a "
    "mutation-counterfactual twin margin: per row, mine the single nearest NON-DUPLICATE target "
    "(the pre-mutation / cross-system counterfactual twin the v3 retrieval eval decides against), "
    "gate by a band-pass mutation-presence weight so the pressure lands only on rows that HAVE a "
    "close-but-distinct twin, and apply a hardest-foil relative margin in the eval's squared-L2 "
    "decision variable requiring the prediction to prefer the CURRENT (post-mutation) content over "
    "its stale twin. Replaces r12's diffuse soft-ring/expected-distance with a sharp single-twin "
    "commitment matched to the dynamical world's concentrated confusable mass."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_DELTA = 0.05
_LAM_FRAC = 0.5
_MARGIN = 0.5
_TAU_R = 0.25
_LAMBDA_TWIN = 0.2


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

    logits = -dist2 / _TEMP
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)

        big = torch.finfo(tt.dtype).max
        tt_masked = tt.masked_fill(eye, big)
        tt_masked = tt_masked.masked_fill(tt <= _DELTA, big)
        hf = tt_masked.min(dim=1)
        hf_val = hf.values
        hf_idx = hf.indices

        confus = torch.exp(-hf_val / lam)
        dupmask = 1.0 - torch.exp(-hf_val / _DELTA)
        gate = (confus * dupmask).clamp(0.0, 1.0)

    d_true = dist2.diagonal()
    d_twin = dist2.gather(1, hf_idx[:, None]).squeeze(1)
    twin = (gate * F.softplus((d_true + _MARGIN - d_twin) / _TAU_R)).mean()

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_TWIN * twin
