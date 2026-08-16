import torch
import torch.nn.functional as F

NAME = "r7_paired_difference_alignment"
DESCRIPTION = (
    "The r22 exact-target equivalence quotient, plus a PAIRED DIFFERENCE-ALIGNMENT hinge over "
    "near-in-stream row pairs. For two rows i, j with numerically distinct targets and |i-j| "
    "inside a band (rows of one trajectory are contiguous in the flattened cmd-position order), "
    "the matched-minus-swapped assignment cost d2(i,i)+d2(j,j)-d2(i,j)-d2(j,i) reduces exactly to "
    "-2 (p_i - p_j) . (t_i - t_j), so the pair term is the regression coefficient of the "
    "prediction difference on the target difference, alpha_ij = (p_i-p_j).(t_i-t_j) / "
    "max(||t_i-t_j||^2, floor), penalised by a smooth hinge below a target coefficient. Any "
    "component shared by the two predictions cancels from alpha, so the term spends gradient only "
    "on the direction that separates the two contents and never on the common conditional mean; it "
    "is the training-time analogue of a matched-vs-swapped assignment test and it fights the "
    "shrinkage that collapses two in-window reads onto the same answer. Weights are the r22 "
    "occurrence measure, so the large equal-target class of silent commands contributes almost "
    "nothing. Saturating below the target coefficient, so scaling the prediction up buys nothing; "
    "a constant prediction drives alpha to zero and is maximally penalised."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_KAPPA = 4.0
_DUP_DELTA = 0.05
_LAM_FRAC = 0.5
_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25
_GATE_EPS = 1e-3

_EQ_EPS = 1e-5
_NUM_EPS = 1e-12

_BAND = 16
_ALIGN_TARGET = 0.75
_TAU_A = 0.25
_FLOOR_FRAC = 0.1
_LAMBDA_ALIGN = 0.5


def loss(pred, tgt):
    n, d = pred.shape
    per_row_mse = (pred - tgt).pow(2).mean(dim=-1)
    if n < 2:
        return per_row_mse.mean()

    with torch.no_grad():
        tf = tgt.float()
        t0_sq = (tf * tf).sum(dim=1, keepdim=True)
        tt_raw = (t0_sq + t0_sq.t() - 2.0 * (tf @ tf.t())).clamp_min(0.0)
        tt_raw = tt_raw / float(d)
        equivalent = tt_raw <= _EQ_EPS

        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()
        row_measure = inv_class.sum().clamp_min(_NUM_EPS)

        err2 = (pred.detach() - tgt).pow(2)
        mse_d = (inv_class.unsqueeze(1) * err2).sum(dim=0) / row_measure
        precision = (1.0 / (mse_d + _EPS)).pow(_BETA)
        precision = precision / precision.mean().clamp_min(_NUM_EPS)
        precision = precision.clamp(_WMIN, _WMAX)
        precision = precision / precision.mean().clamp_min(_NUM_EPS)
        sqrt_precision = precision.sqrt().unsqueeze(0)

    mse_anchor = (inv_class * per_row_mse).sum() / row_measure

    pw = pred * sqrt_precision
    tw = tgt * sqrt_precision
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    cross = pw @ tw.t()
    dist2 = (pw_sq + tw_sq.t() - 2.0 * cross).clamp_min(0.0)
    dist2 = dist2 / float(d)

    cross_diag = cross.diagonal()
    align_num = (cross_diag.unsqueeze(1) + cross_diag.unsqueeze(0) - cross - cross.t())
    align_num = align_num / float(d)

    with torch.no_grad():
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0)
        tt = tt / float(d)
        distinct = ~equivalent

        pair_measure = (
            inv_class.unsqueeze(1)
            * inv_class.unsqueeze(0)
            * distinct.to(inv_class.dtype)
        )
        mean_off = (tt * pair_measure).sum() / pair_measure.sum().clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)

        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / _DUP_DELTA))
        ring = ring.masked_fill(equivalent, 0.0)

        candidate_measure = inv_class.unsqueeze(0)
        neg_measure = candidate_measure * distinct.to(inv_class.dtype)
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_measure).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_measure.sum(dim=1, keepdim=True).clamp_min(1e-6)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log()

        log_class_measure = -class_size.log().unsqueeze(0)

        row_pos = torch.arange(n, device=pred.device)
        banded = (row_pos.unsqueeze(1) - row_pos.unsqueeze(0)).abs() <= _BAND
        upper = row_pos.unsqueeze(1) < row_pos.unsqueeze(0)
        pair_keep = banded & distinct & upper
        pair_weight = pair_measure * pair_keep.to(inv_class.dtype)
        pair_weight_sum = pair_weight.sum().clamp_min(_NUM_EPS)
        align_floor = (_FLOOR_FRAC * mean_off).clamp_min(_EPS)
        align_denom = torch.maximum(tt, align_floor)

    logits = -dist2 / _TEMP + log_importance + log_class_measure
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~equivalent, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        p_class = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_class).pow(_GAMMA)

    listwise = (inv_class * focal * nll).sum() / row_measure

    with torch.no_grad():
        ring_measure = ring * candidate_measure
        mass = ring_measure.sum(dim=1)
        q = ring_measure / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GATE_EPS)

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep_row = gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)
    repulsion = (inv_class * rep_row).sum() / row_measure

    align_coef = align_num / align_denom
    align_hinge = _TAU_A * F.softplus((_ALIGN_TARGET - align_coef) / _TAU_A)
    alignment = (pair_weight * align_hinge).sum() / pair_weight_sum

    return (listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion
            + _LAMBDA_ALIGN * alignment)
