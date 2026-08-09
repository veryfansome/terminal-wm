"""R22 objective: EXACT-TARGET EQUIVALENCE QUOTIENT.

The identity target admits exact equivalence classes: two occurrences with the same
standardized target vector are the same answer. The squared-L2 eval agrees—an exactly
equal foil ties the truth and therefore cannot beat it. The r12 loss does not:
its duplicate guard only removes the EXTRA ring emphasis, while duplicate occurrences
remain separate one-hot classes, separate candidate columns, and separate focal anchors.
For an exact class of size k, diagonal probability is at most about 1/k even when the
TOTAL correct-class probability is approximately one, so the focal term calls a solved
answer hard forever. Repeated empty stdout is the largest class in this corpus.

This loss takes the quotient by exact target equality. If k_j is target j's class size:
  * candidate logits receive -log(k_j), integrating k identical columns to one class;
  * the positive numerator is logsumexp over every target equivalent to t_i;
  * focal hardness uses total positive-class probability;
  * query rows, precision estimates, MSE, and repulsion use measure 1/k_i.
Thus duplicating any training example leaves the complete loss invariant. The r12
precision-weighted L2 geometry and close-BUT-DISTINCT ring remain, so mutation twins and
other genuinely different hard targets are still separated.

Measured TRAIN fact: 55.22% of rows in reference-stack batches have an exact sibling. On
the three archived 4000-step reference-stack checkpoints their aggregate class probability is
0.994-0.996, but diagonal focal hardness remains 0.800-0.812. Quotient weighting shifts
effective batch mass from repeated intervention stdout toward diverse read/revisit targets.

Pure pred/tgt function; no metadata, state, RNG, forward change, or eval visibility.
Anti-collapse: a constant prediction cannot identify all distinct target classes, and the
class-balanced MSE anchor remains strictly positive for varying targets.
"""

import torch
import torch.nn.functional as F

NAME = "r22_exact_target_equivalence_quotient"
DESCRIPTION = (
    "Exact-target quotient of the champion precision/ring L2 objective: numerically equal "
    "identity targets form one multi-positive class, candidate and anchor occurrence mass is "
    "divided by class size, and focal hardness uses aggregate class probability. Retains the "
    "close-distinct ring, repulsion margin, and class-balanced MSE anchor; invariant to exact "
    "sample duplication and anti-collapse-safe."
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

# Float32 Gram-matrix equality tolerance. On 20 real reference-stack batches:
# same-class max 3.34e-6; distinct-class min 3.31e-4; zero FP/FN at 1e-5.
_EQ_EPS = 1e-5
_NUM_EPS = 1e-12


def loss(pred, tgt):
    n, d = pred.shape
    per_row_mse = (pred - tgt).pow(2).mean(dim=-1)
    if n < 2:
        return per_row_mse.mean()

    # Define the target quotient and all detached measures. Equality is computed in
    # unweighted identity-target geometry; near-but-distinct vectors remain negatives.
    with torch.no_grad():
        tf = tgt.float()
        t0_sq = (tf * tf).sum(dim=1, keepdim=True)
        tt_raw = (t0_sq + t0_sq.t() - 2.0 * (tf @ tf.t())).clamp_min(0.0)
        tt_raw = tt_raw / float(d)
        equivalent = tt_raw <= _EQ_EPS

        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()
        row_measure = inv_class.sum().clamp_min(_NUM_EPS)

        # Quotient-weighted free-energy precision: repeating an exact target does not
        # change the error statistics or any dimension's reliability.
        err2 = (pred.detach() - tgt).pow(2)
        mse_d = (inv_class.unsqueeze(1) * err2).sum(dim=0) / row_measure
        precision = (1.0 / (mse_d + _EPS)).pow(_BETA)
        precision = precision / precision.mean().clamp_min(_NUM_EPS)
        precision = precision.clamp(_WMIN, _WMAX)
        precision = precision / precision.mean().clamp_min(_NUM_EPS)
        sqrt_precision = precision.sqrt().unsqueeze(0)

    mse_anchor = (inv_class * per_row_mse).sum() / row_measure

    # r12 precision-weighted per-dimension squared-L2 geometry.
    pw = pred * sqrt_precision
    tw = tgt * sqrt_precision
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0)
    dist2 = dist2 / float(d)

    with torch.no_grad():
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0)
        tt = tt / float(d)
        distinct = ~equivalent

        # Class-pair measure makes the ring scale invariant to duplicating either class.
        pair_measure = (
            inv_class.unsqueeze(1)
            * inv_class.unsqueeze(0)
            * distinct.to(inv_class.dtype)
        )
        mean_off = (tt * pair_measure).sum() / pair_measure.sum().clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)

        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / _DUP_DELTA))
        ring = ring.masked_fill(equivalent, 0.0)

        # Importance normalization is over distinct TARGET CLASSES, not occurrences.
        candidate_measure = inv_class.unsqueeze(0)
        neg_measure = candidate_measure * distinct.to(inv_class.dtype)
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_measure).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_measure.sum(dim=1, keepdim=True).clamp_min(1e-6)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log()

        # k identical candidate columns integrate to one unit of class measure.
        log_class_measure = -class_size.log().unsqueeze(0)

    logits = -dist2 / _TEMP + log_importance + log_class_measure
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~equivalent, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        p_class = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_class).pow(_GAMMA)

    listwise = (inv_class * focal * nll).sum() / row_measure

    # r12 close-distinct repulsion, integrated once per candidate class and once
    # per anchor class. Exact-equivalent targets carry zero repulsion by construction.
    with torch.no_grad():
        ring_measure = ring * candidate_measure
        mass = ring_measure.sum(dim=1)
        q = ring_measure / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GATE_EPS)

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep_row = gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)
    repulsion = (inv_class * rep_row).sum() / row_measure

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion
