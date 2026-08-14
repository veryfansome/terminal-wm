import torch

NAME = "board_transport_permutation_matching"
DESCRIPTION = (
    "Entropic optimal-transport assignment inside each board. The strict-causal previous-"
    "observation channel segments the flattened command rows back into their trajectories; each "
    "trajectory becomes one padded block of reads against the contents that trajectory actually "
    "produced. Per block the squared-L2 cost between every prediction and every content is turned "
    "into a soft permutation by log-domain Sinkhorn-Knopp with uniform row and column marginals, "
    "unrolled with gradient, at an entropic temperature set to a fixed fraction of the mean "
    "distinct target-target separation so the sharpness follows the embedding scale rather than a "
    "hand-set constant. Numerically identical targets form one class, for which the ideal plan "
    "spreads a row's unit mass evenly over its class and still satisfies both marginals exactly, "
    "so duplicates are handled without breaking the bijection. The loss is the negative log of the "
    "transported mass a read places on its own content class, plus the symmetric column term "
    "requiring every content on the board to be claimed by the reads that returned it, each "
    "focal-weighted and divided by class size so the flood of empty mv outputs weighs as one "
    "content. Because the column marginal is pinned, predicting the board's most common content "
    "for every read earns exactly what a constant prediction earns: the equipartition constraint "
    "removes the generic-content shortcut arithmetically instead of penalising it heuristically. "
    "The global in-batch softmax, the confusability ring and the repulsion hinge are gone; a small "
    "unweighted MSE anchor keeps the prediction in the target space the probe measures in."
)

WANTS_CTX = True

_EPS_FRAC = 0.1
_EPS_FLOOR = 1e-3
_ITERS = 15
_ANCHOR = 0.1
_GAMMA = 1.0
_W_ROW = 1.0
_W_COL = 0.5
_EQ_EPS = 1e-5
_KMAX = 96
_FALLBACK_BLOCK = 32
_NEG = -1.0e4
_NUM = 1e-12


def _segment_starts(ctx, n, device):
    if isinstance(ctx, dict):
        prev = ctx.get("prev", None)
        if prev is not None and prev.dim() == 2 and prev.shape[0] == n:
            starts = (prev.detach().abs().amax(dim=1) == 0).clone()
            starts[0] = True
            if int(starts.sum().item()) > 1:
                return starts
    return (torch.arange(n, device=device) % _FALLBACK_BLOCK) == 0


def _blocks(pred, tgt, starts):
    n, d = pred.shape
    device = pred.device
    order = torch.arange(n, device=device)
    seg = starts.long().cumsum(dim=0) - 1
    first = order[starts]
    n_seg = int(first.numel())
    pos = order - first[seg]
    keep = pos < _KMAX
    if int(keep.sum().item()) < 2:
        return None
    bidx = seg[keep]
    pidx = pos[keep]
    kmax = int(pidx.max().item()) + 1
    p_block = torch.zeros(n_seg, kmax, d, dtype=pred.dtype, device=device).index_put(
        (bidx, pidx), pred[keep]
    )
    t_block = torch.zeros(n_seg, kmax, d, dtype=tgt.dtype, device=device).index_put(
        (bidx, pidx), tgt[keep]
    )
    valid = torch.zeros(n_seg, kmax, dtype=torch.bool, device=device)
    valid[bidx, pidx] = True
    return p_block, t_block, valid


def _sinkhorn(log_kernel, valid, iters):
    zero = torch.zeros_like(valid, dtype=log_kernel.dtype)
    f = zero
    g = zero
    for _ in range(iters):
        f = torch.where(valid, -torch.logsumexp(log_kernel + g.unsqueeze(1), dim=2), zero)
        g = torch.where(valid, -torch.logsumexp(log_kernel + f.unsqueeze(2), dim=1), zero)
    log_plan_col = log_kernel + f.unsqueeze(2) + g.unsqueeze(1)
    f_row = torch.where(valid, -torch.logsumexp(log_kernel + g.unsqueeze(1), dim=2), zero)
    log_plan_row = log_kernel + f_row.unsqueeze(2) + g.unsqueeze(1)
    return log_plan_row, log_plan_col


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    mse_anchor = (pred - tgt).pow(2).mean()
    if n < 2:
        return mse_anchor

    packed = _blocks(pred, tgt, _segment_starts(ctx, n, pred.device))
    if packed is None:
        return mse_anchor
    p_block, t_block, valid = packed

    p_sq = p_block.pow(2).sum(dim=-1)
    t_sq = t_block.pow(2).sum(dim=-1)
    cost = (
        p_sq.unsqueeze(2) + t_sq.unsqueeze(1) - 2.0 * torch.bmm(p_block, t_block.transpose(1, 2))
    ).clamp_min(0.0) / float(d)

    with torch.no_grad():
        td = t_block.detach()
        td_sq = td.pow(2).sum(dim=-1)
        sep = (
            td_sq.unsqueeze(2) + td_sq.unsqueeze(1) - 2.0 * torch.bmm(td, td.transpose(1, 2))
        ).clamp_min(0.0) / float(d)
        pair_ok = valid.unsqueeze(2) & valid.unsqueeze(1)
        same_content = (sep <= _EQ_EPS) & pair_ok
        distinct = pair_ok & (~same_content)
        n_distinct = distinct.sum().to(sep.dtype).clamp_min(1.0)
        mean_sep = (sep * distinct.to(sep.dtype)).sum() / n_distinct
        temp = (_EPS_FRAC * mean_sep).clamp_min(_EPS_FLOOR)
        class_size = same_content.sum(dim=2).clamp_min(1).to(pred.dtype)
        inv_class = torch.where(valid, class_size.reciprocal(), torch.zeros_like(class_size))
        measure = inv_class.sum().clamp_min(_NUM)

    floor = torch.full_like(cost, _NEG)
    log_kernel = torch.where(pair_ok, -cost / temp.to(cost.dtype), floor)
    log_plan_row, log_plan_col = _sinkhorn(log_kernel, valid, _ITERS)

    row_mass = torch.logsumexp(torch.where(same_content, log_plan_row, floor), dim=2)
    col_mass = torch.logsumexp(torch.where(same_content, log_plan_col, floor), dim=1)
    row_nll = (-row_mass).clamp_min(0.0)
    col_nll = (-col_mass).clamp_min(0.0)

    with torch.no_grad():
        row_focal = (1.0 - (-row_nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)
        col_focal = (1.0 - (-col_nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    row_term = (inv_class * row_focal * row_nll).sum() / measure
    col_term = (inv_class * col_focal * col_nll).sum() / measure

    total = _W_ROW * row_term + _W_COL * col_term + _ANCHOR * mse_anchor
    if not torch.isfinite(total):
        return mse_anchor
    return total
