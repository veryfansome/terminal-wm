"""objective chunk: MUTATION-COUNTERFACTUAL TWIN margin — the r6 free-energy precision
geometry + a hardest-single-twin, mutation-presence-gated relative margin that replaces
r12's diffuse *soft-ring* reweighting with a SHARP, per-row commitment against the single
most-confusable counterfactual foil the eval actually decides on.

WHY this should raise the v3 (dynamical-world) margin — the mutation lever:
On v3 a read of a mutated path (ls|mutated / cat|mutated) has, sitting IN THE SAME BATCH of
targets, its own COUNTERFACTUAL TWIN — the PRE-mutation content of that very path (the earlier
`cat X -> orig` position's target) — and the retrieval eval ranks the POST-mutation truth against
exactly that stale twin. The decisive top-1 flip on the high-value cells is therefore a SINGLE,
IDENTIFIABLE binary decision: prefer `new` over its twin `orig`. r12 spreads its
anti-retrieval pressure across a soft continuous RING of all close-but-distinct pairs and an
EXPECTED distance to that ring — on a static/read-only world the confusable mass is diffuse and a
soft ring is right, but on v3 the mass CONCENTRATES on one partner per mutated path, so the eval's
gradient is spent, per row, on beating that ONE twin. This objective mines that twin explicitly.

MECHANISM (built on r12/r6 precision geometry; the ONE new piece is the twin margin):
  1. Keep the r6 backbone unchanged: per-dim free-energy PRECISION weighting
     (Pi_d = 1/Var(err_d), detached, tempered, banded, mean-1) applied to per-dim-mean squared L2,
     a row-only focal listwise contrastive term, and a small MSE anchor. This already scores ~0.38
     on v3 and supplies the absolute-placement + anti-collapse floor.
  2. TWIN MINING on TARGET-TARGET geometry (detached, model-independent, stable from step 0 — the
     key design invariant r12 also relies on): for each row i, the counterfactual twin is
         h(i) = argmin_{j != i, tt[i,j] > delta} tt[i,j]
     the nearest OTHER target that is NOT a near-duplicate (dup = same config file across systems =
     a FALSE negative; excluded exactly as r12's dupmask excludes them). h(i) is the single
     hard foil retrieval would supply on a mutated/duplicated path.
  3. MUTATION-PRESENCE GATE (detached band-pass on the twin distance, reusing r12's own
     confusability calibration): gate_i = exp(-hf/lam) * (1 - exp(-hf/delta)), lam = lam_frac *
     mean-off-diag(tt). gate_i ~ 1 only when row i HAS a distinctly-close-but-distinct twin (a
     mutated or cross-system-duplicated path — precisely the v3 high-value rows); gate_i ~ 0 on a
     native, unambiguous read (no twin in batch, or its nearest neighbor is a pure duplicate). So
     the sharp margin pressure is allocated ONLY where the mutation signal lives, leaving the plain
     listwise to handle the rest — differential help, not a uniform tightening the baselines match.
  4. HARDEST-TWIN RELATIVE MARGIN in the eval's own squared-L2 decision variable:
         L_twin = mean_i gate_i * softplus( (d_true_i + margin - d_twin_i) / tau )
     d_true_i = precision-dist2(pred_i, tgt_i), d_twin_i = precision-dist2(pred_i, tgt_{h(i)}).
     "Be closer to your CURRENT content than to your PRE-mutation twin by a margin." Because twins
     are mutually mined (orig's nearest non-dup is often new and vice-versa), the pre-read row and
     the post-read row each reject the other's content — the BIDIRECTIONAL counterfactual punish
     the eval applies, emerging for free from per-row hardest-foil mining.

Difference from r12 (not a tweak): r12 = soft ring reweight of the FULL softmax + expected-distance
hinge to a SOFT confusable *distribution*; this = HARD single-twin argmin selection + a per-row
band-pass presence gate + a hardest-foil MARGIN. The ring smears O(n) partners; the twin margin
commits to the one the v3 eval decides on. Difference from the chunked-delta probe: that is an
ARCH memory; this is the matched OBJECTIVE that turns a resolved twin into top-1 margin.

Contract / safety:
  * Pure function of (pred, tgt). Precision, tt, h(i) indices, gate all DETACHED; grad flows only
    through d_true/d_twin (i.e. through pred). No state, no in-place edits; two [n,n] ops + one
    argmin/gather beyond the r6 backbone (fast on MPS).
  * NaN-safe: eps floors in precision/lam/means; dist2, tt clamp_min(0); argmin over a masked tt
    where diag+dups are +inf, and rows with NO valid twin get hf=+inf -> gate = exp(-inf)=0 (term
    vanishes, no NaN); softplus is finite; n<2 -> MSE anchor only.
  * Anti-collapse: constant pred -> the precision-weighted logit vector is identical across ROWS so
    the listwise softmax cannot favor the diagonal (NLL pinned above its min); MSE(const, varying
    tgt) > 0; and the twin term at constant pred has d_true_i, d_twin_i = distances from one point c
    to two DISTINCT targets tgt_i != tgt_{h(i)}, which collapse cannot simultaneously order in its
    favor across all gated rows -> softplus(margin/tau)-scale residual, not minimized. Collapse
    cannot minimize the loss.
"""

import torch
import torch.nn.functional as F

NAME = "r17_mutation_counterfactual_twin_margin"
DESCRIPTION = (
    "Champion free-energy precision-weighted focal-listwise L2 contrastive backbone PLUS a "
    "mutation-counterfactual twin margin: per row, mine the single nearest NON-DUPLICATE target "
    "(the pre-mutation / cross-system counterfactual twin the v3 retrieval eval decides against), "
    "gate by a band-pass mutation-presence weight so the pressure lands only on rows that HAVE a "
    "close-but-distinct twin, and apply a hardest-foil relative margin in the eval's squared-L2 "
    "decision variable requiring the prediction to prefer the CURRENT (post-mutation) content over "
    "its stale twin. Replaces r12's diffuse soft-ring/expected-distance with a sharp single-twin "
    "commitment matched to the dynamical world's concentrated confusable mass."
)

# ---- r6 backbone constants (unchanged) ----
_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

# ---- twin-margin constants ----
_DELTA = 0.05       # per-dim sqL2 below which two targets are the SAME answer (dup / false neg)
_LAM_FRAC = 0.5     # presence-gate kernel scale = _LAM_FRAC * mean off-diag target-target dist
_MARGIN = 0.5       # required per-dim sqL2 gap of the twin over the true content
_TAU_R = 0.25       # margin sharpness (r12 scale)
_LAMBDA_TWIN = 0.2  # weight on the twin margin (listwise backbone does the bulk placement)


def loss(pred, tgt):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    # --- Free-energy precision (detached), r6 geometry. ---
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        sw = w.sqrt().unsqueeze(0)                                # [1, d]

    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())
    dist2 = dist2.clamp_min(0.0) / float(d)                       # [n, n], grad via pred

    # --- r6 focal listwise term. ---
    logits = -dist2 / _TEMP
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    # --- Mutation-counterfactual twin mining + presence gate (all DETACHED). ---
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)   # [n, n]
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)

        big = torch.finfo(tt.dtype).max
        tt_masked = tt.masked_fill(eye, big)                    # exclude self
        tt_masked = tt_masked.masked_fill(tt <= _DELTA, big)    # exclude near-duplicates (false negs)
        hf = tt_masked.min(dim=1)                               # nearest non-dup other target
        hf_val = hf.values                                      # [n]; = big if no valid twin
        hf_idx = hf.indices                                     # [n]

        # Band-pass presence gate: ~1 only for a distinctly close-but-distinct twin.
        confus = torch.exp(-hf_val / lam)                       # ->0 when hf_val = big
        dupmask = 1.0 - torch.exp(-hf_val / _DELTA)             # ->1 for distinct
        gate = (confus * dupmask).clamp(0.0, 1.0)               # [n]

    # --- Hardest-twin relative margin (grad through pred only). ---
    d_true = dist2.diagonal()                                   # [n]
    d_twin = dist2.gather(1, hf_idx[:, None]).squeeze(1)        # [n]
    twin = (gate * F.softplus((d_true + _MARGIN - d_twin) / _TAU_R)).mean()

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_TWIN * twin
