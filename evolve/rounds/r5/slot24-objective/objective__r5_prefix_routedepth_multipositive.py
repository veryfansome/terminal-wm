import torch
import torch.nn.functional as F

NAME = "r5_prefix_routedepth_multipositive"
DESCRIPTION = (
    "Adds a second, sequence-local discrimination term to the FWL-orthogonalized ring "
    "contrastive. The flat command rows are segmented back into their source trajectories with "
    "no side channel: within a trajectory the strict-causal previous-observation row equals the "
    "preceding row's target exactly, and at a trajectory's first step it is the zero vector, so "
    "a mismatch marks a boundary. Inside each segment, rows whose target vectors are numerically "
    "equal form one class; a class with many members is the no-output class that every "
    "filesystem-mutating command shares, and a class with few members is one piece of file "
    "content. A row is taken as an anchor when its target repeats an earlier small-class target "
    "in the same segment with at least one many-class row strictly between them, and its route "
    "depth is the number of such many-class rows in that gap. Each anchor is scored by a "
    "multi-positive listwise softmax over squared distances to the targets of every small-class "
    "row at or before it in its own segment, positives being every candidate whose target is "
    "numerically equal to the anchor's, with the temperature set as a fraction of the mean "
    "anchor-to-negative target distance and each anchor's mass rising with a saturating function "
    "of its route depth. The batch-level orthogonalized ring listwise term, the gated repulsion "
    "hinge and the untouched squared-error anchor are all retained unchanged, so a constant "
    "prediction still cannot minimize the loss. The new term reads only the prediction and "
    "target matrices and the strict-causal previous observation, and it exists only inside the "
    "training loss."
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

_LAMBDA_PREFIX = 2.0
_PREFIX_TEMP_FRAC = 0.15
_STRICT_EQ_FRAC = 1e-5
_STRICT_EQ_ABS = 1e-6
_MUT_MIN = 4
_CONTENT_MAX = 3
_DEPTH_MIN = 1
_DEPTH_KAPPA = 3.0
_DEPTH_SCALE = 2.0
_SEG_TOL = 1e-5
_MIN_ANCHORS = 4


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


def _segment_ids(tgt, ctx):
    if ctx is None or not hasattr(ctx, "get"):
        return None
    prev = ctx.get("prev")
    if prev is None or prev.dim() != 2 or prev.shape != tgt.shape:
        return None
    n = tgt.shape[0]
    if n < 2:
        return None
    with torch.no_grad():
        pv = prev.detach().to(dtype=torch.float32)
        tv = tgt.detach().to(dtype=torch.float32)
        if not torch.isfinite(pv).all():
            return None
        gap = (pv[1:] - tv[:-1]).abs().amax(dim=1) > _SEG_TOL
        blank = pv[1:].abs().amax(dim=1) <= _SEG_TOL
        start = torch.zeros(n, dtype=torch.bool, device=tgt.device)
        start[0] = True
        start[1:] = gap | blank
        seg = torch.cumsum(start.to(torch.long), dim=0) - 1
    return seg


def _prefix_route_term(pred, sw, rw, rw_sq, tt_raw, mean_raw, seg, d):
    n = pred.shape[0]
    with torch.no_grad():
        thr = (_STRICT_EQ_FRAC * mean_raw).clamp_min(_STRICT_EQ_ABS)
        eq = tt_raw <= thr
        same_seg = seg.unsqueeze(1) == seg.unsqueeze(0)
        in_seg_eq = eq & same_seg
        cls_size = in_seg_eq.sum(dim=1)
        is_mut = cls_size >= _MUT_MIN
        is_content = cls_size <= _CONTENT_MAX
        idx = torch.arange(n, device=pred.device)
        both_content = is_content.unsqueeze(1) & is_content.unsqueeze(0)
        cand = same_seg & (idx.unsqueeze(1) >= idx.unsqueeze(0)) & both_content
        pos = cand & eq
        neg_any = (cand & (~pos)).any(dim=1)
        prior = in_seg_eq & (idx.unsqueeze(1) > idx.unsqueeze(0)) & both_content
        rank = idx.to(torch.float32).unsqueeze(0) + 1.0
        occ = (prior.to(torch.float32) * rank).amax(dim=1)
        has_prior = occ > 0.5
        last_occ = (occ - 1.0).clamp_min(0.0).to(torch.long)
        mut_l = is_mut.to(torch.long)
        mut_cum = torch.cumsum(mut_l, dim=0)
        before_cum = mut_cum - mut_l
        depth = (before_cum - mut_cum.index_select(0, last_occ)).clamp_min(0)
        anchor = is_content & has_prior & neg_any & (depth >= _DEPTH_MIN)
        ai = torch.nonzero(anchor, as_tuple=False).squeeze(1)
        if ai.numel() < _MIN_ANCHORS:
            return None
        cand_a = cand.index_select(0, ai)
        pos_a = pos.index_select(0, ai)
        neg_a = (cand_a & (~pos_a)).to(dtype=tt_raw.dtype)
        neg_n = neg_a.sum().clamp_min(1.0)
        mean_b = ((tt_raw.index_select(0, ai) * neg_a).sum() / neg_n).clamp_min(_EPS)
        temp_b = (_PREFIX_TEMP_FRAC * mean_b).clamp_min(_EPS)
        dep_a = depth.index_select(0, ai).to(dtype=pred.dtype)
        w_a = 1.0 + _DEPTH_KAPPA * (1.0 - torch.exp(-dep_a / _DEPTH_SCALE))
        w_a = w_a / w_a.sum().clamp_min(_NUM_EPS)

    pw_a = (pred * sw).index_select(0, ai)
    pw_a_sq = (pw_a * pw_a).sum(dim=1, keepdim=True)
    d2 = (pw_a_sq + rw_sq.t() - 2.0 * (pw_a @ rw.t())).clamp_min(0.0) / float(d)
    logits = (-d2 / temp_b.to(dtype=d2.dtype)).masked_fill(~cand_a, float("-inf"))
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~pos_a, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)
    return (w_a * nll).sum()


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

    total = listwise + _ANCHOR * raw_anchor + _LAMBDA_REP * rep

    seg = _segment_ids(tgt, ctx)
    if seg is not None:
        prefix = _prefix_route_term(pred, sw, rw, rw_sq, tt_raw, mean_raw, seg, d)
        if prefix is not None:
            total = total + _LAMBDA_PREFIX * prefix
    return total
