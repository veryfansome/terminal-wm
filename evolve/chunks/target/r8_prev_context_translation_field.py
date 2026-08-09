"""LEARNED target chunk: a bounded learned TRANSLATION FIELD conditioned on the previous
observation — make_target = z_obs + phi(z_prev), to_obs = pred - phi(z_prev). Exact inverse for
ANY phi, no orthogonality machinery, identity at init (phi == 0 exactly).

WHY THIS FAMILY, IN THE FASTWEIGHTS CONTEXT (the target chunk's job changed with the champion).
The r7_path_delta_fastweights arch stores raw OBS embeddings as delta-rule memory VALUES and adds
a SCALAR-gated read directly into the prediction: pred = head(h) + sigmoid(g) * target_read. The
read is a linear blend sum_k a_k z_obs_k of past observations. Under a target transform with a
non-identity LINEAR part T (diag gate, rotation), the target demands T z_obs while the read
supplies untransformed obs-space blends — a scalar gate cannot apply T, so training is structurally
pushed to keep T ~ identity (both learned transforms landed within noise of identity in R6/R7).
Per-sample TRANSLATIONS t = z_obs + phi(z_prev) are the unique invertible family whose linear part
is exactly identity: the memory-read pathway stays perfectly aligned (blends of past z_obs still
predict the z_obs part of the target), while phi still reshapes the training geometry.

WHAT phi DOES (the mechanism, not a metaphor). z_prev is in the model's causal input (the obs
token immediately before the command), so the trunk can absorb phi into its prediction; the
POSITIVE-pair error is then unchanged. What changes is the in-batch NEGATIVE geometry of the
listwise L2 contrastive: dist2[i,j] = ||(pred_obs_i - z_obs_j) + (phi_i - phi_j)||^2. Gradient
descent learns phi to displace targets by a code of their prev-context, separating cross-context
negatives (already-easy ones) and concentrating the contrastive pressure on same-context,
genuinely-hard negatives — a learned, context-conditional complement to the sysblock hard-negative
batcher. The fixed delta / partial-residual targets are the special cases phi = -alpha * z_prev of
this family: the delta-family trait RETRIED LEARNED, in the changed (fastweights) context.

HONESTY / SAFETY.
* LEARNED extension: LEARNED = True; make(D) -> nn.Module with make_target / to_obs / reg. Params
  are registered on the net and trained jointly; NO train stats, NO file/state; phi depends only on
  the causally-available z_prev (zeros at step 0 -> a learned constant, still causal).
* Exact reconstruction by construction: to_obs(make_target(z, p), p) == z bit-near-exactly for any
  parameters — a degenerate phi cannot distort the eval, which stays in the fixed obs space.
* Identity at init: the output layer and the diagonal skip are zero-initialized, so phi == 0 and
  this genome starts bit-equal to the identity-target champion.
* Bounded: a radial soft-clip caps ||phi|| < RHO (= 8; obs norms ~ sqrt(768) ~ 28, hard-negative
  distances ~ O(10-40)), so phi can re-rank hard negatives but can never blow up the logits or
  drive the contrastive loss to zero by translation alone; reg() adds a weak pull toward phi = 0
  so the field only departs identity when it pays.
* Anti-collapse intact: a per-sample translation of varied targets is still varied — a constant
  prediction cannot match it; to_obs(zeros, z_prev) is finite and the predict-mean calibration
  guard (computed in raw obs space, target-independent) is unaffected. The objective's precision
  weights are estimated from matched-pair errors, in which an absorbed phi cancels — the diagonal
  precision estimate is undisturbed.
* NaN-safe: the only division is by (RHO + ||phi_raw||) >= RHO > 0; everything else is Linear/SiLU.
"""

import torch
import torch.nn as nn

NAME = "r8_prev_context_translation_field"
DESCRIPTION = (
    "Learned bounded translation field: make_target = z_obs + phi(z_prev) (MLP + diagonal skip, "
    "zero-init -> exact identity at init), to_obs = pred - phi(z_prev) (exact inverse for any phi). "
    "Identity linear part keeps the fastweights obs-space memory read perfectly aligned; phi acts "
    "only on the in-batch negative geometry, separating cross-context negatives so the contrastive "
    "pressure concentrates on same-context hard ones. Learned generalization of delta/partial "
    "residual (phi = -alpha*z_prev is a special case)."
)

LEARNED = True

_HID = 192      # MLP hidden width (768->192->768 ~ 0.3M params, lean next to the ~1.5M trunk)
_RHO = 8.0      # radial soft-clip: ||phi|| < RHO (a re-ranking nudge, never a logit blow-up)
_REG_LAM = 1e-3  # weak activation pull toward phi = 0 (identity unless the field pays for itself)


class PrevContextTranslationField(nn.Module):
    """t = z_obs + phi(z_prev); phi = softclip(MLP(z_prev) + diag(g) z_prev), zero at init."""

    def __init__(self, dim, hid=_HID):
        super().__init__()
        self.dim = int(dim)
        self.inp = nn.Linear(self.dim, hid)          # default init: gives the MLP signal at step 1
        self.act = nn.SiLU()
        self.out = nn.Linear(hid, self.dim)
        self.diag = nn.Parameter(torch.zeros(self.dim))  # explicit partial-residual special case
        nn.init.zeros_(self.out.weight)              # phi == 0 at init -> exact identity target
        nn.init.zeros_(self.out.bias)
        self._phi_pen = None                         # per-batch activation penalty, set in _phi

    def _phi(self, z_prev, track_pen):
        raw = self.out(self.act(self.inp(z_prev))) + self.diag * z_prev
        norm = raw.norm(dim=-1, keepdim=True)
        phi = raw * (_RHO / (_RHO + norm))           # smooth radial clip: ||phi|| < RHO, id near 0
        if track_pen:
            self._phi_pen = (phi * phi).sum(dim=-1).mean()  # mean ||phi||^2 over the batch
        return phi

    def make_target(self, z_obs, z_prev):
        return z_obs + self._phi(z_prev, track_pen=True)

    def to_obs(self, pred, z_prev):
        # Exact inverse of make_target for ANY parameters: subtract the same translation.
        return pred - self._phi(z_prev, track_pen=False)

    def reg(self):
        if self._phi_pen is None:                    # reg before any forward: exact zero, on-graph
            return self.out.weight.sum() * 0.0
        pen = self._phi_pen
        self._phi_pen = None                         # never reuse a freed graph across steps
        return _REG_LAM * pen


def make(dim):
    return PrevContextTranslationField(dim)
