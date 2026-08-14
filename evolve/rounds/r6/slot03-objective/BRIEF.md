TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

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
  id                r5-06-neighbor-contrast-shaped
  its fitness       +0.0112   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r3_fwl_orthogonalized_ring_contrast
  arch                r23_dual_address_transport_pointer
  optim               r18_spectral_capped_transition_readout
  target              neighbor_contrast_shaped_target
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r18_transition_forwardmodel_consistency

YOUR PARENT'S CURRENT objective IMPL — r3_fwl_orthogonalized_ring_contrast (this is the code you are mutating):
--------------------------------------------------------------------------------
import torch
import torch.nn.functional as F

NAME = "r3_fwl_orthogonalized_ring_contrast"
DESCRIPTION = (
    "Precision-weighted focal listwise L2 contrastive with confusability-ring negatives whose "
    "NEGATIVE geometry is command-orthogonalized. Each minibatch cross-fits a detached linear "
    "kernel-ridge map from the causal command embedding (plus a bias) to the target on the "
    "opposite parity half of the rows, and subtracts rho times each row's own fitted value from "
    "the prediction and from every candidate target before distances are formed. The positive "
    "distance is algebraically unchanged by this; an in-batch negative is made hard exactly to "
    "the extent that the difference between its observation and the anchor's is NOT explained by "
    "the difference between their two commands. Rows whose command-unexplained residual carries "
    "little energy are down-weighted as anchors, exact-duplicate targets are pooled into one "
    "multi-positive class with inverse-class-size mass and excluded from the ring, the "
    "temperature / ring / duplicate / margin scales are all fractions of the batch's mean "
    "off-diagonal distance so the softmax stays at a fixed sharpness however much the "
    "orthogonalization compresses the space, and an "
    "untouched raw squared-error anchor keeps a constant prediction from minimizing the loss."
)

WANTS_CTX = True

_TEMP_FRAC = 0.125
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_KAPPA = 4.0
_LAM_FRAC = 0.5
_DUP_FRAC = 0.025
_EQ_FRAC = 1e-3
_LAMBDA_REP = 0.1
_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125
_GEPS = 1e-3

_RHO = 0.9
_RIDGE = 0.1
_FIT_ROWS = 1024
_INFO_FLOOR = 0.25
_NUM_EPS = 1e-12


def _ridge_apply(basis, target, fit_idx, apply_idx, ridge):
    zs = basis.index_select(0, fit_idx)
    ys = target.index_select(0, fit_idx)
    gram = zs @ zs.t()
    m = gram.shape[0]
    scale = gram.diagonal().mean().clamp_min(1e-6)
    reg = gram + (ridge * scale) * torch.eye(m, device=gram.device, dtype=gram.dtype)
    try:
        alpha = torch.linalg.solve(reg, ys)
    except Exception:
        return None
    if not torch.isfinite(alpha).all():
        return None
    fitted = (basis.index_select(0, apply_idx) @ zs.t()) @ alpha
    if not torch.isfinite(fitted).all():
        return None
    return fitted


def _crossfit_nuisance(basis, target, fit_rows, ridge):
    n = basis.shape[0]
    idx = torch.arange(n, device=basis.device)
    parity = idx % 2
    out = torch.zeros_like(target)
    for side in (0, 1):
        apply_idx = idx[parity == side]
        pool = idx[parity != side]
        if apply_idx.numel() == 0:
            continue
        if pool.numel() < 8:
            return None
        stride = max(1, -(-int(pool.numel()) // max(1, int(fit_rows))))
        fit_idx = pool[::stride]
        if fit_idx.numel() < 8:
            fit_idx = pool
        fitted = _ridge_apply(basis, target, fit_idx, apply_idx, ridge)
        if fitted is None:
            return None
        out.index_copy_(0, apply_idx, fitted)
    return out


def _nuisance_shift(pred, tgt, ctx):
    if ctx is None or not hasattr(ctx, "get"):
        return None
    n = pred.shape[0]
    cmd = ctx.get("cmd")
    if cmd is None or cmd.dim() != 2 or cmd.shape[0] != n:
        return None
    with torch.no_grad():
        cmd = cmd.detach().to(dtype=torch.float32)
        if not torch.isfinite(cmd).all():
            return None
        basis = torch.cat([cmd, torch.ones(n, 1, device=cmd.device, dtype=cmd.dtype)], dim=1)
        fitted = _crossfit_nuisance(basis, tgt.detach().to(dtype=torch.float32),
                                    _FIT_ROWS, _RIDGE)
    if fitted is None:
        return None
    return (_RHO * fitted).to(dtype=pred.dtype)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    raw_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return raw_anchor

    shift = _nuisance_shift(pred, tgt, ctx)
    if shift is None:
        pres, tres = pred, tgt
    else:
        pres, tres = pred - shift, tgt - shift

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(_NUM_EPS)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(_NUM_EPS)
        sw = w.sqrt().unsqueeze(0)

    pw = pres * sw
    tw = tres * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        pair_n = float(n * (n - 1))

        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / pair_n).clamp_min(_EPS)

        rw = tgt * sw
        rw_sq = (rw * rw).sum(dim=1, keepdim=True)
        tt_raw = (rw_sq + rw_sq.t() - 2.0 * (rw @ rw.t())).clamp_min(0.0) / float(d)
        mean_raw = (tt_raw.sum() / pair_n).clamp_min(_EPS)

        temp = (_TEMP_FRAC * mean_off).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        dup_scale = (_DUP_FRAC * mean_raw).clamp_min(_NUM_EPS)
        margin = _MARGIN_FRAC * mean_off
        tau = (_TAU_FRAC * mean_off).clamp_min(_EPS)

        equivalent = (tt_raw <= _EQ_FRAC * mean_raw) | eye
        distinct = ~equivalent
        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()

        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt_raw / dup_scale))
        ring = ring.masked_fill(equivalent, 0.0)

        neg_mass = distinct.to(dtype=pred.dtype)
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_mass).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log()
        log_class = -class_size.clamp_min(1.0).log().unsqueeze(0)

        mass = ring.sum(dim=1)
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GEPS)

        energy = (tres * tres).mean(dim=1)
        med = energy.median().clamp_min(_NUM_EPS)
        row_w = energy / (energy + _INFO_FLOOR * med)
        row_w = row_w * inv_class
        row_w = row_w / row_w.mean().clamp_min(_NUM_EPS)

    logits = -dist2 / temp + log_importance + log_class
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(distinct, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    listwise = (row_w * focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (row_w * gate * F.softplus((d_true + margin - d_conf) / tau)).mean()

    return listwise + _ANCHOR * raw_anchor + _LAMBDA_REP * rep
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- boardpair_swap_difference (axis objective)
import torch
import torch.nn.functional as F

NAME = "boardpair_swap_difference"
DESCRIPTION = (
    "Precision-weighted focal listwise L2 contrastive with confusability-ring negative "
    "reweighting and a gated repulsion hinge, plus two terms defined on PAIRS of content-repeat "
    "rows drawn from one trajectory. Trajectory boundaries are recovered exactly from the "
    "strict-causal previous-observation channel, which is the zero vector at the first command of "
    "every sequence and the preceding observation elsewhere. A row counts as a content repeat "
    "when its target duplicates the target of an earlier row of the same trajectory whose command "
    "embedding is not the same command, and when the batch-wide duplicate count of that target is "
    "small, which keeps the large empty-output cluster out. For two such rows carrying different "
    "contents the loss takes the regression coefficient of the PREDICTION DIFFERENCE on the "
    "TARGET DIFFERENCE — their inner product over that pair's own squared separation — and hinges "
    "it at one, so a pair of near-identical observations weighs as much as a pair of far-apart "
    "ones and isotropic prediction error contributes nothing in expectation; it adds a two-sided "
    "hinge requiring each of the two predictions to lie past the midpoint of the two targets "
    "along their difference direction, which is the same forced choice between two contents that "
    "the unweighted squared-L2 decision variable makes. Every mask, threshold, scale and pair "
    "selection is detached, and the separation is floored so a near-duplicate pair cannot "
    "amplify without bound."
)

WANTS_CTX = True

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

_DUP_FRAC = 0.005
_GRP_MAX = 12
_SAME_CMD = 0.999
_FLOOR_FRAC = 0.01
_SWAP_M = 0.25
_MAX_PAIRS = 4096
_W_DIFF = 0.5
_W_SWAP = 0.5


def _precision_sqrt(pred, tgt):
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        return w.sqrt().unsqueeze(0)


def _segment_ids(prev):
    starts = prev.abs().amax(dim=1) == 0
    return starts.long().cumsum(dim=0)


def _repeat_rows(tt, seg, cmd, eye, scale_w):
    n = tt.shape[0]
    device = tt.device
    eps_dup = (_DUP_FRAC * scale_w).clamp_min(1e-8)
    dup = (tt < eps_dup) & (~eye)
    cnt = dup.sum(dim=1)
    same_seg = seg.unsqueeze(1) == seg.unsqueeze(0)
    order = torch.arange(n, device=device)
    earlier = order.unsqueeze(0) < order.unsqueeze(1)
    cand = dup & same_seg & earlier & (cnt <= _GRP_MAX).unsqueeze(1)
    repeat = torch.zeros(n, dtype=torch.bool, device=device)
    ij = cand.nonzero(as_tuple=False)
    if ij.shape[0] > 0:
        cs = F.cosine_similarity(cmd[ij[:, 0]], cmd[ij[:, 1]], dim=1)
        moved = ij[cs < _SAME_CMD, 0]
        if moved.numel() > 0:
            repeat[moved] = True
    return repeat, dup, same_seg


def _board_pairs(repeat, dup, same_seg):
    n = repeat.shape[0]
    order = torch.arange(n, device=repeat.device)
    upper = order.unsqueeze(0) > order.unsqueeze(1)
    pm = repeat.unsqueeze(1) & repeat.unsqueeze(0) & same_seg & (~dup) & upper
    ab = pm.nonzero(as_tuple=False)
    if ab.shape[0] > _MAX_PAIRS:
        stride = (ab.shape[0] + _MAX_PAIRS - 1) // _MAX_PAIRS
        ab = ab[::stride]
    return ab


def _pair_terms(pred, tgt, ab, d, scale_raw):
    ia = ab[:, 0]
    ib = ab[:, 1]
    ta = tgt[ia]
    tb = tgt[ib]
    dt = ta - tb
    dp = pred[ia] - pred[ib]
    with torch.no_grad():
        floor = (_FLOOR_FRAC * scale_raw * float(d)).clamp_min(1e-8)
        den = (dt * dt).sum(dim=1).clamp_min(floor)
    diff = F.relu(1.0 - (dp * dt).sum(dim=1) / den).mean()
    mid = 0.5 * (ta + tb)
    sa = ((pred[ia] - mid) * dt).sum(dim=1) / den
    sb = ((pred[ib] - mid) * dt).sum(dim=1) / den
    swap = (F.relu(_SWAP_M - sa) + F.relu(_SWAP_M + sb)).mean()
    return diff, swap


def loss(pred, tgt, ctx=None):
    n, d = pred.shape

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    sw = _precision_sqrt(pred, tgt)
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

    total = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep

    if n < 4 or ctx is None or "prev" not in ctx or "cmd" not in ctx:
        return total

    with torch.no_grad():
        scale_w = (2.0 * (tw * tw).mean()).clamp_min(1e-8)
        seg = _segment_ids(ctx["prev"])
        repeat, dup, same_seg = _repeat_rows(tt, seg, ctx["cmd"], eye, scale_w)
        if int(repeat.sum()) < 2:
            return total
        ab = _board_pairs(repeat, dup, same_seg)
        scale_raw = (2.0 * (tgt * tgt).mean()).clamp_min(1e-8)
    if ab.shape[0] == 0:
        return total

    diff, swap = _pair_terms(pred, tgt, ab, d, scale_raw)
    return total + _W_DIFF * diff + _W_SWAP * swap

--- r22_exact_target_equivalence_quotient (axis objective)
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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0112 (r5-06-neighbor-contrast-shaped, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
