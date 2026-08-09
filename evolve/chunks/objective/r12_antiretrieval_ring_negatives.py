"""objective chunk: ANTI-RETRIEVAL ring-negative contrastive — champion free-energy
precision geometry with in-batch negatives re-weighted toward RETRIEVAL-CONFUSABLE
target pairs (close-but-distinct observations), plus a small repulsion hinge away
from the confusable set.

THE R12 TARGET (measured, v2 inner): the fitness margin now subtracts
max(retrieve-by-cmd, within-trajectory retrieval). The two binding baselines are
  * cat (+.293, rbc .530): the corpus supplies a near-duplicate observation — the same
    config file on another system. The wrong-but-close candidate is an OBSERVATION
    EMBEDDING NEAR THE TRUTH.
  * grep (+.326, within-traj .513): the trajectory supplies the file's own earlier
    content; the grep answer (matching lines) embeds NEAR that earlier observation.
Both failure modes share one geometric signature: retrieval's candidate is a target
vector CLOSE TO (but distinct from) the true target. Under the champion's uniform
in-batch softmax, such pairs are a vanishing fraction of the negatives, so almost all
contrastive gradient is spent separating easy far negatives — precisely the pairs
retrieval already gets right, which the margin cancels out.

MECHANISM — weight the negatives by a detached CONFUSABILITY RING on target-target
distances, computed in the champion's own precision-weighted per-dim-mean L2 geometry:
  tt[i,j]   = precision-weighted per-dim-mean sqL2 between TRUE targets t_i, t_j
  confus    = exp(-tt / lambda),   lambda = 0.5 * mean off-diag tt  (batch-adaptive)
  dupmask   = 1 - exp(-tt / delta), delta = 0.05  (false-negative guard)
  ring      = confus * dupmask                    (band-pass: peak on close-but-distinct)
  a[i,j]    = (1 + kappa * ring) / row-mean       (off-diag, mean-1 per row; a[i,i] = 1)
and run the champion listwise term as an IMPORTANCE-WEIGHTED softmax:
  logits = -dist2 / tau + log a
This is the hard-negative reweighting of Robinson et al. (arXiv:2010.04592) with two
deliberate inversions: hardness is defined by TARGET-TARGET geometry, not
prediction-negative similarity (so the weighting is exactly "what a retrieval baseline
would return", independent of the model's current errors and stable from step 0), and
it is band-passed a la conditional / ring negative sampling (Wu et al.,
arXiv:2010.02037; Chuang et al. debiased contrastive, arXiv:2007.00224): NEAR-IDENTICAL
targets are down-weighted to ~0 because in this corpus they are FALSE negatives — the
same config file catted in two trajectories is the same right answer, and pushing away
from it is unsatisfiable noise (the champion's uniform softmax pays that noise; the
ring removes it, a second, separate win).

The batcher (sysblock hard-negative) makes this bite: every trajectory contributes all
its cmd positions to the flattened batch, so a grep step's own earlier observations sit
in the tgt rows (the within-traj candidate set), and the same-system block + uniform
remainder supply same-command same/other-system pairs (the retrieve-by-cmd candidate
set). The ring finds both without any metadata, which the loss(pred, tgt) contract
does not provide.

A small REPULSION HINGE makes the anti-retrieval push explicit and margin-bearing:
  q[i,:]    ∝ ring[i,:]                      (the confusable-candidate distribution)
  d_conf_i  = sum_j q[i,j] * dist2[i,j]      (expected distance to the confusable set)
  L_rep     = gate_i * softplus((d_true_i + m - d_conf_i) / tau_r)
i.e. "be closer to YOUR target than to what retrieval would supply, by margin m",
smoothly gated off (detached) for rows with no confusable mass in the batch. This is
the eval's own top-1 decision variable, restricted to the foils that actually decide
the margin against the retrieval baselines.

WHY the margin can rise: the margin only grows where the WM beats BOTH retrievals; a
loss that concentrates its ranking gradient on retrieval-confusable foils optimizes
that residual directly (system-specific config content vs the cross-distro template;
which lines match vs the whole earlier file), instead of re-earning separations the
baselines get for free.

Contract / safety:
  * Pure function of (pred, tgt); ring weights, q, gate, precision all DETACHED; no
    state, no in-place edits of inputs; two extra [n,n] ops beyond the champion (fast).
  * NaN-safe: eps floors in precision, lambda, row means, ring mass; dist2/tt
    clamp_min(0); weights bounded in [1/(1+kappa), 1+kappa] before normalization; log
    of a clamped strictly-positive tensor; n < 2 -> MSE anchor only.
  * Anti-collapse: constant pred -> dist2 rows identical, and log a (detached, computed
    from TARGETS only) cannot be gamed by pred, so the weighted row-softmax still
    cannot favor the diagonal -> NLL pinned away from its minimum; MSE(const, varying
    tgt) strictly positive; the hinge at constant pred has d_true ~ d_conf (both are
    distances from one point to nearby targets) -> softplus(m/tau_r) > 0, not
    minimized. Collapse cannot minimize the loss.
"""

import torch
import torch.nn.functional as F

NAME = "antiretrieval_ring_negatives"
DESCRIPTION = (
    "Champion free-energy precision-weighted focal listwise L2 contrastive whose in-batch "
    "negatives are importance-weighted by a detached CONFUSABILITY RING on target-target "
    "distances — band-pass: near-identical targets (false negatives, the repeated-config-file "
    "case) get ~0 weight, close-but-distinct targets (what retrieve-by-cmd / within-trajectory "
    "retrieval would supply on cat/grep) get up to (1+kappa)x weight — plus a small gated "
    "repulsion hinge requiring the prediction to beat the confusable set by a margin in the "
    "eval's own squared-L2 decision variable."
)

# ---- champion constants (unchanged) ----
_TEMP = 0.25       # softmax temperature on mean-1-normalized per-dim-mean sqL2
_GAMMA = 1.0       # focal focus on not-yet-#1 rows
_ANCHOR = 0.05     # small MSE anchor (anti-collapse + absolute placement)
_BETA = 0.5        # precision temper Pi_d^BETA
_EPS = 1e-2        # MSE floor inside the precision
_WMIN, _WMAX = 0.25, 4.0

# ---- anti-retrieval constants ----
_KAPPA = 4.0       # peak extra negative mass on a maximally confusable pair (log(1+k) ~ 1.6,
                   # comparable to but below the O(1)/_TEMP ~ 4-8 logit gaps: a tilt, not a takeover)
_DELTA = 0.05      # per-dim sq-dist below which two targets are treated as the SAME answer
                   # (false negative -> ring ~ 0). Standardized random pairs sit near 2.0.
_LAM_FRAC = 0.5    # confusability kernel scale = _LAM_FRAC * mean off-diag target-target dist
_LAMBDA_REP = 0.1  # repulsion hinge weight (small: the weighted softmax does most of the work)
_MARGIN = 0.5      # required per-dim sqL2 gap over the confusable set (off-diag mean ~ 2.0)
_TAU_R = 0.25      # hinge sharpness (same scale as _TEMP)
_GEPS = 1e-3       # ring-mass floor inside the row gate


def loss(pred, tgt):
    n, d = pred.shape

    # Absolute-placement anchor + anti-collapse guarantee.
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # --- Free-energy precision: Pi_d = 1 / Var(error_d), batch-estimated, DETACHED. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)                # [d]
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)                  # tempered precision
        w = w / w.mean().clamp_min(1e-12)                      # mean 1 (match eval scale)
        w = w.clamp(_WMIN, _WMAX)                              # band
        w = w / w.mean().clamp_min(1e-12)                      # re-normalize
        sw = w.sqrt().unsqueeze(0)                             # [1, d]

    # Precision-weighted per-dim-mean squared L2 (champion geometry).
    pw = pred * sw                                             # [n, d]
    tw = tgt * sw                                              # [n, d]
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)                 # [n, 1]
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)                 # [n, 1]
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())            # [n, n]
    dist2 = dist2.clamp_min(0.0) / float(d)

    # --- Confusability ring on TARGET-TARGET distances (all DETACHED). ---
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)  # [n, n]
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)  # scalar mean off-diag distance
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)                          # 1 at identical -> 0 far
        dupmask = 1.0 - torch.exp(-tt / _DELTA)                # ~0 identical -> ~1 distinct
        ring = (confus * dupmask).masked_fill(eye, 0.0)        # [n, n] band-pass, zero diag

        # Importance weights: off-diag mean 1 per row (effective negative count unchanged,
        # so temperature/scale stay comparable to the champion), diagonal exactly 1.
        a_raw = 1.0 + _KAPPA * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0)
        log_a = a.clamp_min(1e-6).log()                        # [n, n]

        # Confusable-candidate distribution + smooth row gate for the repulsion hinge.
        mass = ring.sum(dim=1)                                 # [n]
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)           # [n, n], rows sum to ~1 (or ~0)
        gate = mass / (mass + _GEPS)                           # [n] in [0, 1): ~0 if no confusables

    # --- Champion listwise term with importance-weighted negatives. ---
    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    # --- Anti-retrieval repulsion: beat the confusable set by a margin, where it exists. ---
    d_true = dist2.diagonal()                                  # [n], carries grad
    d_conf = (q * dist2).sum(dim=1)                            # [n], grad via dist2 only
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep
