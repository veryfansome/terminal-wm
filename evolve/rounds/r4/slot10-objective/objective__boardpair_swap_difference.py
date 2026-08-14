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
