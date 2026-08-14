import torch
import torch.nn.functional as F

NAME = "r6_boardlocal_forced_choice"
DESCRIPTION = (
    "The r4 dup-class ring objective, plus a BOARD-LOCAL FORCED CHOICE on routed-read rows. "
    "Trajectory boundaries come from the strict-causal previous-observation channel, which is the "
    "zero vector only at a sequence's first command. A row is a routed read when its target "
    "coincides with the target of an EARLIER row of the SAME trajectory that was produced by a "
    "DIFFERENT command embedding, and when that target's batch-wide coincidence count is small, "
    "which keeps the large empty-output cluster and cross-image stock-file clusters out. For those "
    "rows only, the loss adds a second listwise negative log-likelihood whose candidate set is "
    "restricted to the rows of that row's own trajectory - the contents actually on that board - "
    "scored in the UNWEIGHTED squared-L2 decision variable the measurement itself uses rather than "
    "the precision-weighted one, with the same coincidence-class quotient as the global term: every "
    "candidate's mass is divided by its class multiplicity and the numerator is the log-sum-exp "
    "over the whole positive class. Averaging over routed reads alone concentrates the term's whole "
    "gradient mass on the rows where a chain of moves decides the answer, while its total magnitude "
    "stays comparable to the global listwise term. Every mask, threshold and selection is detached; "
    "a constant prediction cannot minimize it, since different rows of one board carry different "
    "positives."
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

_DUP_TT = 1e-3
_NEG_INF = -1e9

_TEMP_BOARD = 0.25
_GRP_MAX = 16
_SAME_CMD = 0.999
_W_BOARD = 1.0


def _routed_rows(coincident, eye, same_seg, cmd):
    n = coincident.shape[0]
    device = coincident.device
    dup = coincident & (~eye)
    order = torch.arange(n, device=device)
    earlier = order.unsqueeze(0) < order.unsqueeze(1)
    small = (dup.sum(dim=1) <= _GRP_MAX).unsqueeze(1)
    cand = dup & same_seg & earlier & small
    routed = torch.zeros(n, dtype=torch.bool, device=device)
    ij = cand.nonzero(as_tuple=False)
    if ij.shape[0] > 0:
        cs = F.cosine_similarity(cmd[ij[:, 0]], cmd[ij[:, 1]], dim=1)
        moved = ij[cs < _SAME_CMD, 0]
        if moved.numel() > 0:
            routed[moved] = True
    return routed


def _board_nll(pred, tgt, rows, board_rows, coincident_rows, log_mult, d):
    pr = pred[rows]
    pr_sq = (pr * pr).sum(dim=1, keepdim=True)
    t_sq = (tgt * tgt).sum(dim=1, keepdim=True)
    raw2 = (pr_sq + t_sq.t() - 2.0 * (pr @ tgt.t())).clamp_min(0.0) / float(d)
    logits = -raw2 / _TEMP_BOARD + board_rows - log_mult
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits + coincident_rows, dim=1)
    return (log_den - log_num).clamp_min(0.0)


def loss(pred, tgt, ctx=None):
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

        coincident = (tt < _DUP_TT) | eye
        log_a = log_a.masked_fill(coincident, 0.0)
        mult = coincident.sum(dim=0).clamp_min(1).to(dist2.dtype)
        log_mult = mult.log().unsqueeze(0)
        log_coincident = torch.zeros_like(tt).masked_fill(~coincident, _NEG_INF)

    logits = -dist2 / _TEMP + log_a - log_mult
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits + log_coincident, dim=1)
    nll = (log_den - log_num).clamp_min(0.0)
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
        seg = (ctx["prev"].abs().amax(dim=1) == 0).long().cumsum(dim=0)
        same_seg = seg.unsqueeze(1) == seg.unsqueeze(0)
        routed = _routed_rows(coincident, eye, same_seg, ctx["cmd"])
        rows = routed.nonzero(as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return total
        board_rows = torch.zeros_like(tt[rows]).masked_fill(~same_seg[rows], _NEG_INF)
        coincident_rows = log_coincident[rows]

    nll_board = _board_nll(pred, tgt, rows, board_rows, coincident_rows, log_mult, d)
    with torch.no_grad():
        p_board = (-nll_board).exp().clamp(0.0, 1.0)
        focal_board = (1.0 - p_board).pow(_GAMMA)
    board = (focal_board * nll_board).mean()

    return total + _W_BOARD * board
