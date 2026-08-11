TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board, measured on the deep windows where analytic shortcuts run out.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

THE CONTRACT — axis 'objective': Expose loss(pred, tgt) -> a scalar tensor carrying grad, where pred and tgt are aligned prediction and target matrices over command rows. Optionally set WANTS_CTX = True, which changes the signature to loss(pred, tgt, ctx); ctx carries a command-input embedding per command row and the strict-causal previous observation, both aligned to pred and tgt and both causal, so the no-future-leakage contract still holds. Must be anti-collapse safe: a constant prediction must not minimize it.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any objective impl: expose `loss(pred, tgt) -> scalar tensor` carrying grad.
  pred: [n, D] predicted next-observation embeddings at command positions
  tgt : [n, D] the tensor produced by the target axis, aligned row-for-row with pred
Both are the flattened cmd-position tensors for the training batch, so batch-level objectives
can be formed from them directly. Must be anti-collapse-safe: a constant prediction must not
minimize it.

CONTRACT EXTENSION (opt-in): a module may set `WANTS_CTX = True`, which changes the signature
to `loss(pred, tgt, ctx) -> scalar`. `ctx` is a dict aligned row-for-row with pred/tgt:
  ctx["cmd"]:  [n, D] the COMMAND embedding that produced each prediction (a model input)
  ctx["prev"]: [n, D] the strict-causal previous-observation embedding per cmd-row
Both are causal, so the no-future-leakage contract still holds. Modules without the flag are
called `loss(pred, tgt)`."""

NAME = "mse"
DESCRIPTION = "R4 baseline: mean squared error to the standardized target embedding."


def loss(pred, tgt):
    return ((pred - tgt) ** 2).mean()
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-11-masked-endpoint-trunk
  its fitness       +0.0037   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r22_masked_endpoint_trunk_task

YOUR PARENT'S CURRENT objective IMPL — r12_antiretrieval_ring_negatives (this is the code you are mutating):
--------------------------------------------------------------------------------
import torch
import torch.nn.functional as F

NAME = "antiretrieval_ring_negatives"
DESCRIPTION = (
    "The free-energy precision-weighted focal listwise L2 contrastive whose in-batch "
    "negatives are importance-weighted by a detached CONFUSABILITY RING on target-target "
    "distances — band-pass: near-identical targets (false negatives, the repeated-config-file "
    "case) get ~0 weight, close-but-distinct targets (what retrieve-by-cmd / within-trajectory "
    "retrieval would supply on cat/grep) get up to (1+kappa)x weight — plus a small gated "
    "repulsion hinge requiring the prediction to beat the confusable set by a margin in the "
    "eval's own squared-L2 decision variable."
)

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

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

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
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

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the swapped content [withheld]. Next-obs retrieval health [withheld].
comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the sw

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r21_counterfactual_history_guidance

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot. Invent a genuinely different mechanism, or a novel RECOMBINATION of ideas already in the archive. Do not resubmit an impl you were shown. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before is NOT off-limits. Evolution recombines: a trait that failed ALONE can win in a CHANGED context. You MAY retry a past idea in a new context; argue why the change could flip it.
- LOOK OUTSIDE MACHINE LEARNING — the largest gains in this line of work have come from cross-domain lenses. Read beyond machine learning (neuroscience: predictive coding, hippocampal and episodic memory, place and grid cells; biology; physics; information theory) and translate ONE concrete mechanism into code — equations, not metaphor.
- IGNORE ANY PERFORMANCE FIGURE YOU FIND IN IMPL SOURCE. Some carried mechanisms document measurements taken on a different objective and a different data root, and those numbers mean nothing for what is scored here. Read that source for its MECHANISM — the equations, the interface, what it does and why — and never as a target to match or beat.
- TRACK CONTENT, NOT DISTURBANCE — the scored windows are exactly the ones where knowing THAT something moved is not enough. A mechanism that marks a location as touched, or that keys on the name being asked about, or on which item moved first or last, cancels to zero by construction. Only carrying an item's identity across several hops earns anything.
- NEVER touch the eval, the metric, the split, or the no-leakage guard. The harness re-checks, and a violation makes the candidate unusable regardless of its number.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0037 (g0-11-masked-endpoint-trunk, full budget, inner split).
The unmodified baseline scores +0.0037 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
