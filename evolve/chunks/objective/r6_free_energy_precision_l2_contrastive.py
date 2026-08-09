"""objective chunk: Free-Energy PRECISION-WEIGHTED L2 contrastive (Rao-Ballard / Friston).

Cross-domain principle (predictive coding & the free-energy principle):
  Perception minimizes VARIATIONAL FREE ENERGY, which for a Gaussian generative model is a
  sum of squared prediction errors each scaled by its PRECISION Pi_d = 1 / Var(error_d) -- the
  inverse variance (reliability) of that error channel (Millidge et al. 2021, arXiv:2107.12979).
  Precisions multiplicatively modulate the influence of each error signal, down-weighting
  unreliable channels; formally this is natural-gradient / Fisher-information preconditioning
  (Millidge et al., arXiv:2111.06942). We translate this ONE principle into the retrieval loss.

WHY this might raise the held-out content-verb margin:
  The eval ranks the true next-obs vs same-verb foils by UNIFORM per-dim-mean squared L2. Per
  dimension: a PREDICTABLE dim has small error to the true target and large error to a foil, so it
  votes reliably for truth (signal). An UNPREDICTABLE dim (model outputs ~mean, targets ~N(0,1))
  contributes a zero-mean f_d - e_d -> pure NOISE in the ranking decision. Precision Pi_d = 1/MSE_d
  is high on signal dims, low on noise dims. Weighting the contrastive squared-distance by Pi_d
  DENOISES the ranking -- it concentrates the margin on the dimensions that actually separate
  same-verb observations, the JEPA "abstract away unpredictable detail" bet applied per-dimension.
  We keep the SAME metric-matched geometry as the L2 contrastive champion (row-only direction,
  per-dim-mean squared L2, temperature 0.25, focal top-1 reweighting, small MSE anchor) and add
  ONLY the precision tilt, NORMALIZED to mean 1 so overall scale/temperature match the uniform
  eval geometry (avoiding plain-InfoNCE's metric-mismatch penalty) while the tilt aligns training
  with the signal-bearing channels.

Contract / safety:
  * Pure function of (pred, tgt); precision weights are DETACHED (an empirical-Bayes M-step estimate
    of Var(error_d)) so gradients flow only through the distances, not the preconditioner. No file /
    network / global state; no in-place edits of inputs. All ops are elementwise + matmul (fast MPS).
  * NaN-safe: MSE floored by eps inside the precision, dist2 clamped >= 0, weights clamped to a finite
    band and re-normalized. n<2 -> MSE anchor only.
  * Anti-collapse: a constant prediction makes the precision-weighted logit vector identical across
    ROWS (pred_i is the same for all i), so the row softmax cannot put mass on the diagonal -> the
    contrastive NLL is pinned away from its minimum; and MSE(const, varying tgt) is strictly positive.
    Collapse cannot minimize the loss.
"""

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

_TEMP = 0.25       # on the mean-1-normalized per-dim-mean sqL2; gap d_true(~0.1-1) vs foil(~2) is O(1)
_GAMMA = 1.0       # focal focus on not-yet-#1 examples (the ones that decide top-1)
_ANCHOR = 0.05     # absolute-placement anchor (metric is shift-sensitive); also anti-collapse
_BETA = 0.5        # precision temper: w_d = Pi_d^BETA (0 -> uniform champion geometry; 1 -> full precision)
_EPS = 1e-2        # MSE floor inside precision (standardized dims have MSE ~0.1-2; caps 1/MSE blow-up)
_WMIN, _WMAX = 0.25, 4.0  # band on the mean-1 weights so no single dim dominates the geometry


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee (positive for constant pred vs varying tgt).
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # --- Free-energy precision: Pi_d = 1 / Var(error_d), estimated from the batch, DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                 # [d] per-dim mean squared error
        prec = 1.0 / (mse_d + _EPS)                            # [d] precision = inverse error variance
        w = prec.pow(_BETA)                                    # temper the tilt
        w = w / w.mean().clamp_min(1e-12)                      # normalize -> mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                              # band: no single dim dominates
        w = w / w.mean().clamp_min(1e-12)                      # re-normalize to mean 1
        sw = w.sqrt()                                          # [d]; scaling both sides by sqrt(w)
        sw = sw.unsqueeze(0)                                   # [1, d] broadcast over batch rows

    # Precision-weighted per-dim-mean squared L2 via the (a-b)^2 expansion on sqrt(w)-scaled vectors:
    #   dist2[i,j] = mean_d w_d (pred_i,d - tgt_j,d)^2  == ||sqrt(w)*pred_i - sqrt(w)*tgt_j||^2 / d
    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())           # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)                    # per-dim mean (metric normalization)

    # Metric direction only: pred_i ranked against candidate TRUE targets by (negative) weighted L2.
    logits = -dist2 / _TEMP                                    # closer (in precision geometry) => higher
    labels = torch.arange(n, device=pred.device)

    logp = F.log_softmax(logits, dim=1)                       # [n, n]
    nll = -logp.gather(1, labels[:, None]).squeeze(1)         # [n]

    # Detached focal weight: concentrate gradient on rows whose true target is not yet closest.
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)

    listwise = (focal * nll).mean()
    return listwise + _ANCHOR * mse_anchor
