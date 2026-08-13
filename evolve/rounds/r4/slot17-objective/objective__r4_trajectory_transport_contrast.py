import torch
import torch.nn.functional as F

NAME = "r4_trajectory_transport_contrast"
DESCRIPTION = (
    "Full-weight squared-error regression plus a pooled multi-positive listwise contrastive whose "
    "negatives are scored in the eval's own unweighted per-dim squared-L2, and which is organised "
    "by TRAJECTORY structure recovered inside the loss. Rows are segmented into their source "
    "sequences by the zero strict-causal previous observation that marks each sequence's first "
    "step. Targets are grouped into equivalence classes by exact duplication; classes far larger "
    "than any content class (the empty observation a filesystem-mutating command returns, the "
    "boilerplate root listing) are marked generic and pooled instead of being forced apart. A row "
    "is marked TRANSPORTED when its target duplicates a non-generic target seen earlier in the "
    "same trajectory while every earlier occurrence carried a different command, so the content "
    "now under this command was previously exposed under another one; the mark is gated smoothly "
    "by the cosine distance between the two commands, which zeroes a re-read of the same path. "
    "Transported rows carry a large share of the listwise mass, get a second cross-entropy whose "
    "candidate set is restricted to the other distinct non-generic contents of their own "
    "trajectory, and get a margin hinge against the nearest of those contents. In-batch negatives "
    "everywhere are additionally importance-weighted by a band-pass confusability ring on "
    "target-target distance and by a same-trajectory bonus. Class sizes bias the partition "
    "function, all scales are fractions of the batch's mean off-diagonal target distance, and the "
    "raw squared-error term keeps a constant prediction from minimizing the loss."
)

WANTS_CTX = True

_ANCHOR = 1.0
_LAM_LIST = 0.25
_LAM_TRANS = 0.6
_LAM_HINGE = 0.15

_TEMP_FRAC = 0.125
_GAMMA = 1.0

_EQ_FRAC = 1e-4
_EQ_MIN = 1e-5
_EQ_MAX = 1e-2

_GENERIC_FRAC = 0.01
_GENERIC_MIN = 8.0
_CLASS_FLOOR = 4.0

_DUP_FRAC = 0.025
_RING_FRAC = 0.5
_KAPPA_RING = 4.0
_KAPPA_LOCAL = 2.0

_TRANSPORT_BOOST = 6.0
_CMD_GATE_FRAC = 0.25

_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125

_EPS = 1e-2
_NUM_EPS = 1e-12
_MASS_FLOOR = 1e-3
_BIG = 1e6


def _segment_ids(prev, n):
    if prev is None or prev.dim() != 2 or prev.shape[0] != n:
        return None
    starts = prev.detach().abs().sum(dim=1) == 0
    if int(starts.sum()) < 2:
        return None
    return torch.cumsum(starts.to(torch.long), dim=0)


def _command_gate(cmd, prior, n, pair_n):
    if cmd is None or cmd.dim() != 2 or cmd.shape[0] != n:
        return None
    c = cmd.detach().to(dtype=torch.float32)
    if not torch.isfinite(c).all():
        return None
    cn = F.normalize(c, dim=1)
    cd = (1.0 - cn @ cn.t()).clamp_min(0.0)
    mean_cd = (cd.sum() / pair_n).clamp_min(_NUM_EPS)
    scale = (_CMD_GATE_FRAC * mean_cd).clamp_min(_NUM_EPS)
    cd_min = cd.masked_fill(~prior, _BIG).amin(dim=1)
    return 1.0 - torch.exp(-cd_min / scale)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    mse = ((pred - tgt) ** 2).mean()
    if n < 4:
        return mse

    device = pred.device
    has_ctx = ctx is not None and hasattr(ctx, "get")

    with torch.no_grad():
        t = tgt.detach().to(dtype=torch.float32)
        t_sq = (t * t).sum(dim=1, keepdim=True)
        tt = (t_sq + t_sq.t() - 2.0 * (t @ t.t())).clamp_min(0.0) / float(d)

        eye = torch.eye(n, dtype=torch.bool, device=device)
        pair_n = float(n * (n - 1))
        mean_off = (tt.sum() / pair_n).clamp_min(_EPS)

        thr_eq = (_EQ_FRAC * mean_off).clamp(_EQ_MIN, _EQ_MAX)
        equivalent = (tt <= thr_eq) | eye
        distinct = ~equivalent

        class_size = equivalent.sum(dim=1).to(dtype=pred.dtype)
        generic = class_size >= max(_GENERIC_MIN, _GENERIC_FRAC * float(n))
        info = ~generic

        seg = _segment_ids(ctx.get("prev") if has_ctx else None, n)
        transport = torch.zeros(n, device=device, dtype=pred.dtype)
        if seg is None:
            local = torch.zeros(n, n, dtype=torch.bool, device=device)
        else:
            same_seg = seg.unsqueeze(1) == seg.unsqueeze(0)
            pair_info = info.unsqueeze(1) & info.unsqueeze(0)
            local = same_seg & distinct & pair_info
            order = torch.arange(n, device=device)
            earlier = order.unsqueeze(1) > order.unsqueeze(0)
            prior = equivalent & same_seg & earlier & pair_info
            has_prior = prior.any(dim=1)
            gate = _command_gate(ctx.get("cmd") if has_ctx else None, prior, n, pair_n)
            if gate is not None:
                transport = (gate * has_prior.to(dtype=gate.dtype)).to(dtype=pred.dtype)

        cls_w = _CLASS_FLOOR / class_size.clamp_min(_CLASS_FLOOR)
        row_w = (1.0 + _TRANSPORT_BOOST * transport) * cls_w
        row_w = row_w / row_w.mean().clamp_min(_NUM_EPS)

        lam = (_RING_FRAC * mean_off).clamp_min(_EPS)
        dup_scale = (_DUP_FRAC * mean_off).clamp_min(_NUM_EPS)
        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / dup_scale))
        ring = ring.masked_fill(equivalent, 0.0)

        a_raw = 1.0 + _KAPPA_RING * ring + _KAPPA_LOCAL * local.to(dtype=pred.dtype)
        d_mass = distinct.to(dtype=pred.dtype)
        a_mean = (a_raw * d_mass).sum(dim=1, keepdim=True)
        a_mean = a_mean / d_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log()
        log_class = -class_size.clamp_min(1.0).log().unsqueeze(0)

        temp = (_TEMP_FRAC * mean_off).clamp_min(_EPS)
        margin = _MARGIN_FRAC * mean_off
        tau = (_TAU_FRAC * mean_off).clamp_min(_EPS)

        local_any = local.any(dim=1).to(dtype=pred.dtype)

    p_sq = (pred * pred).sum(dim=1, keepdim=True)
    g_sq = (tgt * tgt).sum(dim=1, keepdim=True)
    dist2 = (p_sq + g_sq.t() - 2.0 * (pred @ tgt.t())).clamp_min(0.0) / float(d)

    neg_inf = float("-inf")
    biased = -dist2 / temp + log_class

    log_num = torch.logsumexp(biased.masked_fill(distinct, neg_inf), dim=1)
    log_den = torch.logsumexp(biased + log_importance, dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    listwise = (row_w * focal * nll).mean()

    keep = equivalent | local
    t_den = torch.logsumexp(biased.masked_fill(~keep, neg_inf), dim=1)
    t_nll = (t_den - log_num).clamp_min(0.0)
    transported = (transport * local_any).sum().clamp_min(_MASS_FLOOR)
    trans = (transport * local_any * t_nll).sum() / transported

    d_true = dist2.diagonal()
    d_hard = dist2.masked_fill(~local, _BIG).amin(dim=1)
    hinge = F.softplus((d_true + margin - d_hard) / tau)
    hinge_term = (transport * local_any * hinge).sum() / transported

    return _ANCHOR * mse + _LAM_LIST * listwise + _LAM_TRANS * trans + _LAM_HINGE * hinge_term
