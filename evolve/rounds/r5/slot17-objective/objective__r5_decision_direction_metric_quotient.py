import torch
import torch.nn.functional as F

NAME = "r5_decision_direction_metric_quotient"
DESCRIPTION = (
    "Exact-target-equivalence quotient contrastive (numerically equal targets form one "
    "multi-positive class carrying inverse-class-size occurrence mass) over the close-but-distinct "
    "confusability ring, with two pieces of geometry derived from the retrieval decision rule "
    "instead of from prediction error. (1) The per-coordinate metric used inside the listwise "
    "softmax is the ring-weighted second moment of the unit target-difference directions: "
    "w_d proportional to (sum over confusable pairs of (t_id - t_jd)^2 / ||t_i - t_j||^2) ^ alpha, "
    "clamped and renormalised to mean one, so coordinates on which confusable observations "
    "actually differ carry the distance and coordinates on which they agree do not; the weights "
    "collapse to uniform when those directions are isotropic. This replaces the inverse-per-"
    "dimension-error precision weighting. (2) The repulsion term is a scale-free pairwise hinge in "
    "the raw unweighted squared-L2 the eval ranks in: for a confusable pair the quantity "
    "(||pred - t_j||^2 - ||pred - t_i||^2) / ||t_i - t_j||^2 is formed algebraically as "
    "1 + 2 e_i.(t_i - t_j)/||t_i - t_j||^2 with e_i = pred_i - t_i, and a softplus requires it to "
    "stay above a fixed fraction of one, aggregated over each row's ring by a temperature-"
    "controlled soft maximum so the hardest confusable partner dominates rather than the ring "
    "mean. Ring, duplicate, equivalence, temperature and gap-floor scales are all fractions of the "
    "class-measured mean off-diagonal target distance, and a class-measured raw squared-error "
    "anchor keeps a constant prediction from minimising the loss."
)

_EQ_FRAC = 1e-3
_LAM_FRAC = 0.5
_DUP_FRAC = 0.025
_KAPPA = 4.0
_TEMP_FRAC = 0.125
_GAMMA = 1.0
_ANCHOR = 0.05

_ALPHA = 0.75
_WMIN = 0.4
_WMAX = 3.0

_LAMBDA_REP = 0.25
_RATIO = 0.5
_TAU = 0.2
_TAU_AGG = 0.5
_GAP_FLOOR_FRAC = 0.1

_GEPS = 1e-3
_EPS = 1e-6
_NUM_EPS = 1e-12
_NEG = -1e30


def loss(pred, tgt):
    n, d = pred.shape
    per_row_mse = (pred - tgt).pow(2).mean(dim=-1)
    if n < 2:
        return per_row_mse.mean()

    with torch.no_grad():
        tf = tgt.detach().float()
        t_sq = (tf * tf).sum(dim=1, keepdim=True)
        tt = (t_sq + t_sq.t() - 2.0 * (tf @ tf.t())).clamp_min(0.0) / float(d)
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        mean_all = (tt * (~eye).to(tt.dtype)).sum() / float(n * (n - 1))
        equivalent = (tt <= _EQ_FRAC * mean_all) | eye
        distinct = ~equivalent
        usable = bool(distinct.any())

    if not usable:
        return per_row_mse.mean()

    with torch.no_grad():
        class_size = equivalent.sum(dim=1).clamp_min(1).float()
        inv_class = class_size.reciprocal()
        row_measure = inv_class.sum().clamp_min(_NUM_EPS)
        pair_measure = inv_class.unsqueeze(1) * inv_class.unsqueeze(0) * distinct.float()
        pair_mass = pair_measure.sum().clamp_min(_NUM_EPS)
        mean_off = ((tt * pair_measure).sum() / pair_mass).clamp_min(_EPS)

        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        dup = (_DUP_FRAC * mean_off).clamp_min(_EPS)
        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / dup))
        ring = ring.masked_fill(equivalent, 0.0)
        ringw = ring * inv_class.unsqueeze(1) * inv_class.unsqueeze(0)

        gap_floor = (_GAP_FLOOR_FRAC * mean_off * float(d)).clamp_min(_EPS)
        gap_eff = (tt * float(d)).clamp_min(gap_floor)

        rn = ringw / gap_eff
        row_rn = rn.sum(dim=1, keepdim=True)
        v = 2.0 * ((row_rn * tf * tf).sum(dim=0) - ((rn @ tf) * tf).sum(dim=0))
        v = v.clamp_min(0.0)
        v = v / v.mean().clamp_min(_NUM_EPS)
        dim_w = v.clamp_min(_NUM_EPS).pow(_ALPHA).clamp(_WMIN, _WMAX)
        dim_w = dim_w / dim_w.mean().clamp_min(_NUM_EPS)
        if not torch.isfinite(dim_w).all():
            dim_w = torch.ones_like(dim_w)
        sqrt_w = dim_w.sqrt().unsqueeze(0).to(dtype=pred.dtype)

    pw = pred * sqrt_w
    tw = tgt * sqrt_w
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        ttw = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0).float() / float(d)
        mean_offw = ((ttw * pair_measure).sum() / pair_mass).clamp_min(_EPS)
        temp = (_TEMP_FRAC * mean_offw).clamp_min(_EPS).to(dtype=pred.dtype)

        neg_measure = inv_class.unsqueeze(0) * distinct.float()
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_measure).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_measure.sum(dim=1, keepdim=True).clamp_min(1e-6)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log().to(dtype=pred.dtype)
        log_class = (-class_size.log()).unsqueeze(0).to(dtype=pred.dtype)

    logits = -dist2 / temp + log_importance + log_class
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(distinct, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)
        row_w = inv_class.to(dtype=pred.dtype)
        measure = row_measure.to(dtype=pred.dtype)

    listwise = (row_w * focal.to(dtype=pred.dtype) * nll).sum() / measure

    err = pred - tgt
    align = (err * tgt).sum(dim=1, keepdim=True) - err @ tgt.t()

    with torch.no_grad():
        coef = (2.0 / (_TAU * gap_eff)).to(dtype=pred.dtype)
        ring_mass = ringw.sum(dim=1)
        q = ringw / ring_mass.clamp_min(_NUM_EPS).unsqueeze(1)
        log_q = q.clamp_min(_NUM_EPS).log().to(dtype=pred.dtype)
        no_ring = ringw <= 0.0
        gate = (ring_mass / (ring_mass + _GEPS)).to(dtype=pred.dtype)

    arg = float((_RATIO - 1.0) / _TAU) - align * coef
    pen = F.softplus(arg)
    z = (pen / _TAU_AGG + log_q).masked_fill(no_ring, _NEG)
    hardest = _TAU_AGG * torch.logsumexp(z, dim=1)
    repulsion = (row_w * gate * hardest).sum() / measure

    mse_anchor = (row_w * per_row_mse).sum() / measure

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion
