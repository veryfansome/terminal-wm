import torch
import torch.nn.functional as F

NAME = "r7_board_conditional_routed_contrast"
DESCRIPTION = (
    "Board-conditional supervised contrast. The strict-causal previous-observation channel is "
    "exactly zero only at each trajectory's first command row, so cumsum over that indicator "
    "recovers which board every flattened row belongs to. The listwise candidate set is then "
    "restricted to the anchor's own board instead of the whole batch, which replaces the parent's "
    "global target-distance confusability ring with the structurally correct confusable set: the "
    "other contents sitting on the same filesystem. Exact-duplicate targets inside a board remain "
    "one multi-positive class with occurrence mass divided by class size, and each anchor is "
    "additionally weighted by a causal routing indicator - it has an EARLIER same-board duplicate "
    "of its target, that duplicate group is small, and the earlier read used a different command - "
    "which selects exactly the reads whose content arrived at this location through a chain of "
    "silent moves. Class-balanced MSE anchor, per-dimension precision weighting, focal hardness and "
    "the gated repulsion margin against the nearest distinct in-board content are retained. "
    "Anti-collapse-safe: a constant prediction leaves a strictly positive in-board negative log "
    "likelihood, a positive hinge and a positive MSE."
)

WANTS_CTX = True

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25

_EQ_EPS = 1e-5
_NUM_EPS = 1e-12

_GRP_MAX = 4
_CMD_RAMP = 0.02
_RHO = 3.0
_FAR = 1.0e4


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    per_row_mse = (pred - tgt).pow(2).mean(dim=-1)
    if n < 2 or ctx is None or "prev" not in ctx or "cmd" not in ctx:
        return per_row_mse.mean()

    with torch.no_grad():
        board_start = (ctx["prev"].abs().sum(dim=1) == 0).long()
        seg = torch.cumsum(board_start, dim=0)
        same_board = seg.unsqueeze(1) == seg.unsqueeze(0)
        del board_start, seg

        tf = tgt.float()
        t0 = (tf * tf).sum(dim=1, keepdim=True)
        tt = (t0 + t0.t() - 2.0 * (tf @ tf.t())).clamp_min(0.0) / float(d)
        equivalent = (tt <= _EQ_EPS) & same_board
        distinct = same_board & (~equivalent)
        del tf, t0, tt

        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()
        row_measure = inv_class.sum().clamp_min(_NUM_EPS)
        log_class_measure = -class_size.log().unsqueeze(0)

        order = torch.arange(n, device=pred.device)
        earlier = order.unsqueeze(1) > order.unsqueeze(0)
        not_self = order.unsqueeze(1) != order.unsqueeze(0)
        small = class_size <= float(_GRP_MAX)

        cn = F.normalize(ctx["cmd"].float(), dim=1)
        cmd_new = ((1.0 - (cn @ cn.t())).clamp_min(0.0) / _CMD_RAMP).clamp(0.0, 1.0)
        del cn

        source = equivalent & not_self & earlier & small.unsqueeze(1) & small.unsqueeze(0)
        route = (source.to(pred.dtype) * cmd_new.to(dtype=pred.dtype)).amax(dim=1)
        del source, cmd_new, earlier, not_self, small

        row_w = inv_class * (1.0 + _RHO * route)
        w_measure = row_w.sum().clamp_min(_NUM_EPS)

        err2 = (pred.detach() - tgt).pow(2)
        mse_d = (inv_class.unsqueeze(1) * err2).sum(dim=0) / row_measure
        precision = (1.0 / (mse_d + _EPS)).pow(_BETA)
        precision = precision / precision.mean().clamp_min(_NUM_EPS)
        precision = precision.clamp(_WMIN, _WMAX)
        precision = precision / precision.mean().clamp_min(_NUM_EPS)
        sqrt_precision = precision.sqrt().unsqueeze(0)
        del err2, mse_d, precision

        gate = distinct.any(dim=1).to(dtype=pred.dtype)

    mse_anchor = (inv_class * per_row_mse).sum() / row_measure

    pw = pred * sqrt_precision
    tw = tgt * sqrt_precision
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    logits = (-dist2 / _TEMP + log_class_measure).masked_fill(~same_board, float("-inf"))
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~equivalent, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    listwise = (row_w * focal * nll).sum() / w_measure

    d_true = dist2.diagonal()
    d_near = dist2.masked_fill(~distinct, _FAR).amin(dim=1)
    rep_row = gate * F.softplus((d_true + _MARGIN - d_near) / _TAU_R)
    repulsion = (row_w * rep_row).sum() / w_measure

    return listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion
