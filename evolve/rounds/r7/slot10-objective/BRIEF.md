TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

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
  id                r5-04-persistent-gradient-mixture
  its fitness       +0.0037   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r22_exact_target_equivalence_quotient
  arch                r24_backward_multihop_pointer_chase   params {"addr_hidden": 256, "chase_gate_bias": -2.0, "chase_hops": 6, "chase_logit_scale": 8.0, "chase_step_bias": 1.0, "content_proj": 32, "hop_emb": 16, "n_addr": 2}
  optim               r5_persistent_gradient_mixture_channel_opener
  target              identity
  batcher             r2_routed_collision_depth_batcher   params {"depth_beta": 0.8, "group_size": 4, "hard_frac_max": 0.5, "ramp_frac": 0.3, "size_pow": 0.5}
  stream              baseline_interleave
  head                r3_occupancy_routed_copy_transport

YOUR PARENT'S CURRENT objective IMPL — r22_exact_target_equivalence_quotient (this is the code you are mutating):
--------------------------------------------------------------------------------
import torch
import torch.nn.functional as F

NAME = "r22_exact_target_equivalence_quotient"
DESCRIPTION = (
    "Exact-target quotient of the r12 precision/ring L2 objective: numerically equal "
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

_EQ_EPS = 1e-5
_NUM_EPS = 1e-12


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
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0)
    dist2 = dist2 / float(d)

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

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r12_antiretrieval_ring_negatives (axis objective)
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

--- r2_routed_recall_namedecoy (axis objective)
import torch
import torch.nn.functional as F

NAME = "r2_routed_recall_namedecoy"
DESCRIPTION = (
    "Plain MSE to the standardized target as the backbone, plus a second regression term and a "
    "listwise softmax term that are both restricted to ROUTED-RECALL rows: rows whose target is "
    "an exact duplicate of another row's target inside a small duplicate group (group size <= 4, "
    "which separates repeated file contents from the large cluster of empty-output observations) "
    "AND whose command embedding is not the same command as its duplicate partner's, so the "
    "content reached this row by an intervening command rather than by re-issuing the same read. "
    "The listwise term draws its negatives only from a band of neighbouring rows in the flattened "
    "cmd-position order — the rows of the same trajectory, i.e. the other contents on the same "
    "board — and importance-weights those negatives by COMMAND-embedding cosine to the anchor's "
    "command, so the candidate whose own read command most resembles the anchor's is the hardest "
    "negative. A margin hinge in the eval's squared-L2 decision variable is applied against that "
    "single most command-similar in-band candidate. Per-dimension precision weighting shapes the "
    "distance space; all masks, weights and thresholds are detached."
)

WANTS_CTX = True

_W = 32
_CHUNK = 256
_GRP_MAX = 4
_EPS_DUP_FRAC = 0.005

_BETA = 0.5
_PEPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_TEMP = 0.25
_GAMMA = 1.0
_KAPPA = 4.0
_TAU_C = 0.05
_CMD_RAMP = 0.02
_RHO = 3.0

_W_LIST = 1.0
_W_RMSE = 0.5
_LAM_REP = 0.2
_MARGIN = 0.5
_TAU_R = 0.25

_NEG_INF = -1e9


def _precision_sqrt(pred, tgt):
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _PEPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        return w.sqrt().unsqueeze(0)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 4 or ctx is None or "cmd" not in ctx:
        return mse_anchor

    sw = _precision_sqrt(pred, tgt)
    pw = pred * sw
    with torch.no_grad():
        tw = tgt * sw
        cn = F.normalize(ctx["cmd"], dim=1)
        tw_sq = (tw * tw).sum(dim=1)
        scale = (2.0 * (tw * tw).mean()).clamp_min(1e-8)
        eps_dup = (_EPS_DUP_FRAC * scale).clamp_min(1e-8)
        moved_all = torch.zeros(n, device=pred.device, dtype=pred.dtype)

    row_mse = ((pred - tgt) ** 2).mean(dim=1)
    pw_sq = (pw * pw).sum(dim=1)

    list_num = pred.new_zeros(())
    rep_num = pred.new_zeros(())
    wsum = pred.new_zeros(())

    for a0 in range(0, n, _CHUNK):
        a1 = min(a0 + _CHUNK, n)
        c0 = max(0, a0 - _W)
        c1 = min(n, a1 + _W)

        cand = tw[c0:c1]
        cand_sq = tw_sq[c0:c1]
        anch = pw[a0:a1]

        d2 = (pw_sq[a0:a1].unsqueeze(1) + cand_sq.unsqueeze(0)
              - 2.0 * (anch @ cand.t())).clamp_min(0.0) / float(d)

        with torch.no_grad():
            tt = (cand_sq.unsqueeze(1) + cand_sq.unsqueeze(0)
                  - 2.0 * (cand @ cand.t())).clamp_min(0.0) / float(d)
            dup_cc = tt < eps_dup
            grp_c = dup_cc.sum(dim=1)
            small_c = grp_c <= _GRP_MAX

            ai = torch.arange(a0, a1, device=pred.device)
            cj = torch.arange(c0, c1, device=pred.device)
            band = (ai.unsqueeze(1) - cj.unsqueeze(0)).abs() <= _W
            eye = ai.unsqueeze(1) == cj.unsqueeze(0)

            dup_ac = dup_cc[a0 - c0:a1 - c0]
            small_a = small_c[a0 - c0:a1 - c0]

            cmat = cn[a0:a1] @ cn[c0:c1].t()

            partner = dup_ac & (~eye) & band & small_c.unsqueeze(0) & small_a.unsqueeze(1)
            has_p = partner.any(dim=1)
            min_c = cmat.masked_fill(~partner, 2.0).amin(dim=1)
            moved = has_p.to(pred.dtype) * ((1.0 - min_c) / _CMD_RAMP).clamp(0.0, 1.0)
            moved_all[a0:a1] = moved

            neg = band & (~dup_ac) & (~eye) & small_c.unsqueeze(0) & small_a.unsqueeze(1)
            n_neg = neg.sum(dim=1)
            gate = (n_neg > 0).to(pred.dtype)

            s = F.softmax(cmat.masked_fill(~neg, _NEG_INF) / _TAU_C, dim=1)
            a_raw = 1.0 + _KAPPA * n_neg.unsqueeze(1).to(pred.dtype) * s
            log_a = (a_raw / (1.0 + _KAPPA)).clamp_min(1e-6).log() * neg.to(pred.dtype)

            allowed = neg | eye
            self_col = (ai - c0).unsqueeze(1)
            decoy_col = cmat.masked_fill(~neg, _NEG_INF).argmax(dim=1, keepdim=True)
            rw = (1.0 + _RHO * moved) * gate

        logits = (-d2 / _TEMP + log_a).masked_fill(~allowed, _NEG_INF)
        logp = F.log_softmax(logits, dim=1)
        nll = -logp.gather(1, self_col).squeeze(1)
        with torch.no_grad():
            focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

        d_true = d2.gather(1, self_col).squeeze(1)
        d_dec = d2.gather(1, decoy_col).squeeze(1)
        rep = F.softplus((d_true + _MARGIN - d_dec) / _TAU_R)

        list_num = list_num + (rw * focal * nll).sum()
        rep_num = rep_num + (rw * rep).sum()
        wsum = wsum + rw.sum()

    listwise = list_num / wsum.clamp_min(1e-6)
    repulse = rep_num / wsum.clamp_min(1e-6)
    mv_mass = moved_all.sum()
    mse_routed = (moved_all * row_mse).sum() / mv_mass.clamp_min(1e-6)

    return (mse_anchor + _W_RMSE * mse_routed + _W_LIST * listwise + _LAM_REP * repulse)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0037 (r5-04-persistent-gradient-mixture, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
