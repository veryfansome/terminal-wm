import math

import torch
import torch.nn.functional as F

NAME = "r4_relocation_address_ring_quotient"
DESCRIPTION = (
    "Two-geometry objective. The batch-wide head is the command-orthogonalized focal listwise "
    "L2 contrast: a cross-fit detached ridge map from the causal command embedding to the target "
    "is subtracted from the prediction and from every candidate before distances are formed, "
    "negatives are importance-weighted by a confusability ring, exact-duplicate targets are "
    "pooled into one multi-positive class, and rows are weighted by their command-unexplained "
    "energy. The second head works in the eval's own unweighted squared-L2 geometry and is "
    "restricted to RELOCATION ROWS, which the loss discovers from the batch alone: a row whose "
    "target is an exact duplicate of an earlier target in the SAME trajectory while its command "
    "is a different command, excluding targets whose duplicate class is large (the silent "
    "no-output observation that every mutating command shares). Trajectory boundaries come from "
    "the strict-causal previous observation, which is exactly zero only at a trajectory's first "
    "step. Exact duplication is decided by a fixed cosine-basis sketch compared without any "
    "matmul cancellation, conjoined with a loose full-space distance check. For each relocation "
    "row the candidate set is the other content-bearing observations of its own trajectory, and "
    "the negatives are ranked by an ADDRESS RING: a softmax over the cosine similarity between "
    "that row's command embedding and each candidate's command embedding, so the candidate whose "
    "address most resembles the address being read carries the most negative mass. That ring "
    "drives a listwise term and a margin hinge on the expected ring distance, and relocation "
    "rows are also up-weighted in the batch-wide head. An untouched raw squared-error anchor "
    "keeps a constant prediction from minimizing the loss."
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
_LAMBDA_REP = 0.1
_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125
_GEPS = 1e-3

_RHO = 0.9
_RIDGE = 0.1
_FIT_ROWS = 1024
_INFO_FLOOR = 0.25
_NUM_EPS = 1e-12

_SKETCH_K = 8
_SKETCH_CHUNK = 512
_EQ_SKETCH_FRAC = 1e-6
_EQ_FULL_FRAC = 2e-2
_EQ_COS_SLACK = 1e-2
_MAX_CLASS = 8

_RELOC_BOOST = 3.0
_ADDR_TAU = 0.05
_ADDR_KAPPA = 8.0
_WS_TEMP_FRAC = 0.125
_WS_MARGIN_FRAC = 0.5
_WS_TAU_FRAC = 0.25
_LAMBDA_WS = 0.5
_LAMBDA_ADDR = 0.25

_BASIS_CACHE = {}


def _sketch_basis(dim, k, device, dtype):
    key = (int(dim), int(k), str(device), str(dtype))
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        i = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(k, device=device, dtype=torch.float32).unsqueeze(0)
        basis = torch.cos(math.pi * (i + 0.5) * (2.0 * j + 1.0) / float(dim)).to(dtype)
        _BASIS_CACHE[key] = basis
    return basis


def _sketch_pairwise(x):
    y = x @ _sketch_basis(x.shape[1], _SKETCH_K, x.device, x.dtype)
    n = y.shape[0]
    out = y.new_empty(n, n)
    for s in range(0, n, _SKETCH_CHUNK):
        e = min(s + _SKETCH_CHUNK, n)
        out[s:e] = (y[s:e].unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    return out


def _sketch_equal(x):
    dm = _sketch_pairwise(x)
    thr = _EQ_SKETCH_FRAC * dm.mean().clamp_min(_NUM_EPS)
    return dm <= thr


def _segments(prev, n, device):
    if prev is None or prev.dim() != 2 or prev.shape[0] != n:
        return None
    starts = prev.detach().abs().amax(dim=1) == 0
    if int(starts.sum()) < 2 or not bool(starts[0]):
        return None
    return torch.cumsum(starts.to(dtype=torch.long), dim=0) - 1


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


def _nuisance_shift(pred, tgt, cmd):
    if cmd is None:
        return None
    n = pred.shape[0]
    with torch.no_grad():
        if not torch.isfinite(cmd).all():
            return None
        basis = torch.cat([cmd, torch.ones(n, 1, device=cmd.device, dtype=cmd.dtype)], dim=1)
        fitted = _crossfit_nuisance(basis, tgt.detach().to(dtype=torch.float32), _FIT_ROWS, _RIDGE)
    if fitted is None:
        return None
    return (_RHO * fitted).to(dtype=pred.dtype)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    raw_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return raw_anchor

    cmd = None
    prev = None
    if ctx is not None and hasattr(ctx, "get"):
        c = ctx.get("cmd")
        if c is not None and c.dim() == 2 and c.shape[0] == n:
            cmd = c.detach().to(dtype=torch.float32)
        p = ctx.get("prev")
        if p is not None and p.dim() == 2 and p.shape[0] == n:
            prev = p

    shift = _nuisance_shift(pred, tgt, cmd)
    if shift is None:
        pres, tres = pred, tgt
    else:
        pres, tres = pred - shift, tgt - shift

    with torch.no_grad():
        tdet = tgt.detach().to(dtype=torch.float32)
        t_sq = (tdet * tdet).sum(dim=1, keepdim=True)
        tt_plain = (t_sq + t_sq.t() - 2.0 * (tdet @ tdet.t())).clamp_min(0.0) / float(d)
        pair_n = float(n * (n - 1))
        mean_plain = (tt_plain.sum() / pair_n).clamp_min(_EPS)

        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        equivalent = _sketch_equal(tdet) & (tt_plain <= _EQ_FULL_FRAC * mean_plain)
        equivalent = equivalent | eye
        distinct = ~equivalent
        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()
        content = class_size <= float(_MAX_CLASS)

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
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / pair_n).clamp_min(_EPS)

        temp = (_TEMP_FRAC * mean_off).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        dup_scale = (_DUP_FRAC * mean_plain).clamp_min(_NUM_EPS)
        margin = _MARGIN_FRAC * mean_off
        tau = (_TAU_FRAC * mean_off).clamp_min(_EPS)

        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt_plain / dup_scale))
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

        seq = _segments(prev, n, pred.device)
        reloc = torch.zeros(n, dtype=torch.bool, device=pred.device)
        cand = None
        q_addr = None
        if seq is not None and cmd is not None:
            pos_idx = torch.arange(n, device=pred.device)
            same_seq = seq.unsqueeze(1) == seq.unsqueeze(0)
            earlier = pos_idx.unsqueeze(1) > pos_idx.unsqueeze(0)
            cnorm = F.normalize(cmd, dim=1)
            csim = cnorm @ cnorm.t()
            same_cmd = _sketch_equal(cmd) & (csim >= 1.0 - _EQ_COS_SLACK)
            source = equivalent & same_seq & earlier & (~same_cmd) & content.unsqueeze(0)
            reloc = source.any(dim=1) & content
            cand = same_seq & distinct & content.unsqueeze(0) & content.unsqueeze(1)
            scores = torch.where(cand, csim / _ADDR_TAU, torch.full_like(csim, -1e9))
            q_addr = torch.softmax(scores, dim=1) * cand.to(dtype=pred.dtype)
            q_addr = q_addr / q_addr.sum(dim=1, keepdim=True).clamp_min(_NUM_EPS)
            reloc = reloc & cand.any(dim=1)

        energy = (tres * tres).mean(dim=1)
        med = energy.median().clamp_min(_NUM_EPS)
        row_w = energy / (energy + _INFO_FLOOR * med)
        row_w = row_w * inv_class
        row_w = row_w * (1.0 + _RELOC_BOOST * reloc.to(dtype=pred.dtype))
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

    n_reloc = int(reloc.sum()) if reloc.numel() else 0
    if n_reloc == 0 or cand is None:
        return total

    rows = reloc.nonzero(as_tuple=False).squeeze(1)
    pr = pred.index_select(0, rows)
    pr_sq = (pr * pr).sum(dim=1, keepdim=True)
    draw = (pr_sq + t_sq.t() - 2.0 * (pr @ tgt.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        same_seq_rows = seq.index_select(0, rows).unsqueeze(1) == seq.unsqueeze(0)
        pos_sub = equivalent.index_select(0, rows) & same_seq_rows
        cand_sub = cand.index_select(0, rows)
        allow = pos_sub | cand_sub
        n_cand = cand_sub.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=pred.dtype)
        qa_sub = q_addr.index_select(0, rows)
        a_ws = 1.0 + _ADDR_KAPPA * (qa_sub * n_cand)
        a_ws_mean = (a_ws * cand_sub.to(dtype=pred.dtype)).sum(dim=1, keepdim=True) / n_cand
        imp_ws = (a_ws / a_ws_mean.clamp_min(1e-6)).clamp_min(1e-6).log()
        imp_ws = imp_ws.masked_fill(~cand_sub, 0.0)
        temp_ws = (_WS_TEMP_FRAC * mean_plain).clamp_min(_EPS)
        margin_ws = _WS_MARGIN_FRAC * mean_plain
        tau_ws = (_WS_TAU_FRAC * mean_plain).clamp_min(_EPS)

    logits_ws = (-draw / temp_ws + imp_ws).masked_fill(~allow, float("-inf"))
    den_ws = torch.logsumexp(logits_ws, dim=1)
    num_ws = torch.logsumexp(logits_ws.masked_fill(~pos_sub, float("-inf")), dim=1)
    nll_ws = (den_ws - num_ws).clamp_min(0.0)

    with torch.no_grad():
        focal_ws = (1.0 - (-nll_ws).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    within = (focal_ws * nll_ws).mean()

    d_own = draw.gather(1, rows.unsqueeze(1)).squeeze(1)
    d_addr = (qa_sub * draw).sum(dim=1)
    addr_rep = F.softplus((d_own + margin_ws - d_addr) / tau_ws).mean()

    return total + _LAMBDA_WS * within + _LAMBDA_ADDR * addr_rep
