import torch
import torch.nn.functional as F

NAME = "r5_boardlocal_namecollision_quotient_nce"
DESCRIPTION = (
    "Multi-positive listwise objective over command rows in which the NEGATIVE candidate "
    "distribution is an explicit per-row mixture with fixed mass budgets. Numerically equal "
    "targets are merged into one positive class and every candidate's occurrence mass is divided "
    "by its class size; the remaining mass is split between four renormalized components: a "
    "uniform share, a same-trajectory share (rows are cut into trajectories at the all-zero "
    "strict-causal previous observation), a command-collision share (candidates whose command "
    "input is an upper-tail outlier in cosine similarity to the anchor's own command input while "
    "carrying a different target), and a band-pass target-confusability ring that suppresses "
    "near-duplicate targets. Each share is renormalized inside the row, so the fraction a "
    "component receives does not depend on batch size or on how many candidates it contains, and "
    "a component with no mass folds back into the uniform share. Rows whose command collides with "
    "a different target are additionally upweighted in the listwise average. Retains per-dimension "
    "precision weighting, focal hardness, a margin repulsion hinge against the confusable set in "
    "the squared-L2 decision variable, and a class-balanced squared-error anchor. Reads only the "
    "causal command input and the strict-causal previous observation; a constant prediction does "
    "not minimize it."
)

WANTS_CTX = True

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_EQ_EPS = 1e-5
_DUP_DELTA = 0.05
_LAM_FRAC = 0.5
_NUM_EPS = 1e-12

_RHO_GROUP = 0.35
_RHO_COLL = 0.25
_RHO_RING = 0.20

_Z_CUT = 2.0
_Z_TAU = 0.5
_ROW_COLL = 1.0

_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25
_GATE_EPS = 1e-3

_NEG_INF = float("-inf")


def _row_normalize(x):
    return x / x.sum(dim=1, keepdim=True).clamp_min(_NUM_EPS)


def _segment_ids(prev):
    starts = prev.abs().sum(dim=1) == 0
    if not bool(starts.any()):
        return None
    return torch.cumsum(starts.long(), dim=0)


def _collision(cmd, n, dtype):
    flat = cmd.detach().reshape(n, -1).float()
    unit = F.normalize(flat, dim=-1)
    sim = unit @ unit.t()
    diag = sim.diagonal()
    pairs = float(n) * float(n - 1)
    first = (sim.sum() - diag.sum()) / pairs
    second = ((sim * sim).sum() - (diag * diag).sum()) / pairs
    spread = (second - first * first).clamp_min(1e-12).sqrt().clamp_min(1e-6)
    z = ((sim - first) / spread).to(dtype)
    return torch.sigmoid((z - _Z_CUT) / _Z_TAU)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    per_row_mse = (pred - tgt).pow(2).mean(dim=-1)
    if n < 2:
        return per_row_mse.mean()

    dtype = pred.dtype
    eye = torch.eye(n, dtype=torch.bool, device=pred.device)

    with torch.no_grad():
        tf = tgt.detach().float()
        t_sq = (tf * tf).sum(dim=1, keepdim=True)
        tt_raw = ((t_sq + t_sq.t() - 2.0 * (tf @ tf.t())).clamp_min(0.0)) / float(d)
        equivalent = (tt_raw <= _EQ_EPS) | eye
        distinct = ~equivalent
        distinct_f = distinct.to(dtype)

        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype)
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
    dist2 = ((pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0)) / float(d)

    with torch.no_grad():
        tt = ((tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0)) / float(d)
        base = inv_class.unsqueeze(0) * distinct_f
        pair_measure = inv_class.unsqueeze(1) * base
        mean_off = (tt * pair_measure).sum() / pair_measure.sum().clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        not_dup = 1.0 - torch.exp(-tt / _DUP_DELTA)
        ring = torch.exp(-tt / lam) * not_dup * distinct_f

        group = torch.zeros_like(tt)
        prev = ctx.get("prev") if isinstance(ctx, dict) else None
        if prev is not None and prev.dim() == 2 and prev.shape[0] == n:
            seg = _segment_ids(prev)
            if seg is not None and int(seg.max()) > int(seg.min()):
                group = (seg.unsqueeze(1) == seg.unsqueeze(0)).to(dtype) * not_dup * distinct_f

        collide = torch.zeros_like(tt)
        cmd = ctx.get("cmd") if isinstance(ctx, dict) else None
        if cmd is not None and cmd.shape[0] == n:
            collide = _collision(cmd, n, dtype) * not_dup * distinct_f

        uniform = _row_normalize(base)
        has_neg = (base.sum(dim=1, keepdim=True) > 0).to(dtype)

        mass_group = (base * group).sum(dim=1, keepdim=True)
        mass_coll = (base * collide).sum(dim=1, keepdim=True)
        mass_ring = (base * ring).sum(dim=1, keepdim=True)
        share_group = (mass_group > _NUM_EPS).to(dtype) * _RHO_GROUP
        share_coll = (mass_coll > _NUM_EPS).to(dtype) * _RHO_COLL
        share_ring = (mass_ring > _NUM_EPS).to(dtype) * _RHO_RING
        neg_w = (1.0 - share_group - share_coll - share_ring) * uniform
        neg_w = neg_w + share_group * _row_normalize(base * group)
        neg_w = neg_w + share_coll * _row_normalize(base * collide)
        neg_w = neg_w + share_ring * _row_normalize(base * ring)
        neg_w = neg_w * has_neg

        pos_w = equivalent.to(dtype) * inv_class.unsqueeze(1)
        cand_w = neg_w + pos_w
        allowed = cand_w > 0
        log_w = cand_w.clamp_min(_NUM_EPS).log()

    logits = (-dist2 / _TEMP + log_w).masked_fill(~allowed, _NEG_INF)
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~equivalent, _NEG_INF), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        p_class = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_class).pow(_GAMMA)
        row_w = inv_class * (1.0 + _ROW_COLL * collide.amax(dim=1))
        row_norm = row_w.sum().clamp_min(_NUM_EPS)

    listwise = (row_w * focal * nll).sum() / row_norm

    with torch.no_grad():
        confusable = base * (ring + group + collide)
        mass = confusable.sum(dim=1)
        q = confusable / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GATE_EPS)

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep_row = gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)
    repulsion = (inv_class * rep_row).sum() / row_measure

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion
