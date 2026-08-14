import torch
import torch.nn.functional as F

NAME = "r6_episodic_board_prototype_choice"
DESCRIPTION = (
    "Rebuilds the evaluation's own N-way forced choice out of the training trajectories and makes "
    "it the dominant term, as a prototypical-network episodic cross-entropy. Each trajectory is an "
    "EPISODE, recovered exactly from the strict-causal previous-observation channel, which is the "
    "zero vector at the first command of a sequence and the preceding observation everywhere else. "
    "Inside an episode the SUPPORT set of a row is the distinct contents already exposed at earlier "
    "steps of that same episode; numerically equal targets are pooled into one class and classes "
    "whose in-batch multiplicity exceeds a batch-relative cap are dropped, which removes the empty-"
    "output cluster and the system files that repeat across every trajectory of a blocked batch. A "
    "row becomes a QUERY exactly when its own content numerically repeats a support content that "
    "was first exposed under a DIFFERENT command, i.e. when the content standing at this location "
    "arrived there by a move or a copy rather than by the command naming it. The query loss is the "
    "prototypical softmax over that episode's support classes in the evaluation's own squared-L2 "
    "decision variable at a temperature that is a fixed fraction of the batch's mean target "
    "separation, implemented as a class-pooled logsumexp with a minus-log-multiplicity correction "
    "so each distinct content counts once however often it was re-read. Because the same board also "
    "admits a purely positional answer, each query is reweighted by inverse propensity to a uniform "
    "prior over the RECENCY RANK of its true content among the episode's support classes: buckets "
    "of rank that happen to be common in a batch are down-weighted and rare ones up-weighted, with "
    "the weight clamped, so a constant most-recent / least-recent policy cannot profit from the "
    "empirical imbalance. A hinge on the same queries demands the true content beat the single "
    "hardest board-mate by a batch-relative margin in the same decision variable. A batch-global "
    "class-pooled listwise term over all content rows keeps the whole embedding geometry "
    "discriminative and delivers gradient to every row, and a small raw squared-error anchor keeps "
    "a constant prediction from minimizing the loss."
)

WANTS_CTX = True

_EPS = 1e-6
_NUM_EPS = 1e-12
_MIN_ROWS = 16
_EQ_FRAC = 1e-3
_CMD_EQ_FRAC = 1e-3
_SEP_FLOOR_REL = 0.1
_SEP_FLOOR_ABS = 5e-3
_GRP_MIN = 16
_GRP_FRAC = 0.02
_TEMP_FRAC = 0.125
_TEMP_EP_FRAC = 0.25
_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125
_RANK_MAX = 7
_WMIN, _WMAX = 0.25, 4.0
_MAX_QUERIES = 512
_W_GLOBAL = 1.0
_W_EPISODE = 2.0
_W_MARGIN = 0.25
_W_ANCHOR = 0.05
_NEG = float("-inf")


def _sqdist(a, b, d):
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True)
    return (a2 + b2.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(d)


def _side_info(ctx, n):
    if not isinstance(ctx, dict):
        return None, None
    cmd = ctx.get("cmd")
    prev = ctx.get("prev")
    if not torch.is_tensor(cmd) or not torch.is_tensor(prev):
        return None, None
    if cmd.dim() != 2 or prev.dim() != 2:
        return None, None
    if cmd.shape[0] != n or prev.shape[0] != n:
        return None, None
    return cmd, prev


def _rank_weights(rank, dtype):
    bucket = rank.round().clamp(0.0, float(_RANK_MAX)).long()
    counts = torch.bincount(bucket, minlength=_RANK_MAX + 1).to(torch.float32)
    w = counts.clamp_min(1.0).reciprocal().index_select(0, bucket)
    w = w / w.mean().clamp_min(_NUM_EPS)
    w = w.clamp(_WMIN, _WMAX)
    w = w / w.mean().clamp_min(_NUM_EPS)
    w = w.clamp(_WMIN, _WMAX)
    return w.to(dtype)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    anchor = ((pred - tgt) ** 2).mean()
    if n < _MIN_ROWS:
        return anchor

    with torch.no_grad():
        tf = tgt.detach().to(dtype=torch.float32)
        if not torch.isfinite(tf).all():
            return anchor
        tt = _sqdist(tf, tf, d)
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        mean_off = (tt.sum() / float(n * (n - 1))).clamp_min(_EPS)
        equivalent = (tt <= _EQ_FRAC * mean_off) | eye
        class_size = equivalent.sum(dim=1)
        cap = max(_GRP_MIN, int(_GRP_FRAC * n))
        content = class_size <= cap
        temp = (_TEMP_FRAC * mean_off).clamp_min(_EPS)
        content_idx = content.nonzero(as_tuple=False).squeeze(1)

    total = _W_ANCHOR * anchor

    if int(content_idx.numel()) >= 4:
        with torch.no_grad():
            eq_c = equivalent.index_select(0, content_idx).index_select(1, content_idx)
            log_mult_c = eq_c.sum(dim=1).clamp_min(1).to(dtype=pred.dtype).log().unsqueeze(0)
        dist_c = _sqdist(pred.index_select(0, content_idx),
                         tgt.index_select(0, content_idx), d)
        logits_c = -dist_c / temp - log_mult_c
        den_c = torch.logsumexp(logits_c, dim=1)
        num_c = torch.logsumexp(logits_c.masked_fill(~eq_c, _NEG), dim=1)
        total = total + _W_GLOBAL * (den_c - num_c).clamp_min(0.0).mean()

    cmd, prev = _side_info(ctx, n)
    if cmd is None:
        return total

    with torch.no_grad():
        starts = prev.detach().abs().amax(dim=1) == 0
        if int(starts.sum()) < 2:
            return total
        seg = starts.long().cumsum(dim=0)
        order = torch.arange(n, device=pred.device)
        support = (
            (order.unsqueeze(0) < order.unsqueeze(1))
            & (seg.unsqueeze(1) == seg.unsqueeze(0))
            & content.unsqueeze(0)
            & content.unsqueeze(1)
        )
        cf = cmd.detach().to(dtype=torch.float32)
        cdist = _sqdist(cf, cf, cf.shape[1])
        cmean = (cdist.sum() / float(n * (n - 1))).clamp_min(_EPS)
        cmd_same = cdist <= _CMD_EQ_FRAC * cmean
        del cdist
        positive = support & equivalent
        candidate = ((positive & (~cmd_same)).any(dim=1)) & content
        cand_idx = candidate.nonzero(as_tuple=False).squeeze(1)
        if int(cand_idx.numel()) < 2:
            return total
        if int(cand_idx.numel()) > _MAX_QUERIES:
            stride = (int(cand_idx.numel()) + _MAX_QUERIES - 1) // _MAX_QUERIES
            cand_idx = cand_idx[::stride]

        sup_q = support.index_select(0, cand_idx)
        pos_q = positive.index_select(0, cand_idx)
        mult_q = sup_q.to(dtype=torch.float32) @ equivalent.to(dtype=torch.float32)
        mult_q = mult_q.clamp_min(1.0)
        inv_q = torch.where(sup_q, mult_q.reciprocal(), torch.zeros_like(mult_q))
        keep = inv_q.sum(dim=1) >= 1.5
        keep_idx = keep.nonzero(as_tuple=False).squeeze(1)
        if int(keep_idx.numel()) < 2:
            return total
        query_idx = cand_idx.index_select(0, keep_idx)
        sup_q = sup_q.index_select(0, keep_idx)
        pos_q = pos_q.index_select(0, keep_idx)
        inv_q = inv_q.index_select(0, keep_idx)
        log_mult_q = mult_q.index_select(0, keep_idx).log().to(dtype=pred.dtype)
        del mult_q

        nq = int(query_idx.numel())
        pos_pos = torch.where(pos_q, order.unsqueeze(0).expand(nq, n),
                              torch.zeros(1, dtype=order.dtype, device=order.device))
        last = pos_pos.amax(dim=1)
        newer = sup_q & (order.unsqueeze(0) > last.unsqueeze(1)) & (~pos_q)
        rank = (newer.to(dtype=torch.float32) * inv_q).sum(dim=1)
        row_w = _rank_weights(rank, pred.dtype)
        foil_q = sup_q & (~pos_q)
        has_foil = foil_q.any(dim=1)
        tt_q = tt.index_select(0, query_idx)
        wide = tt_q.amax().clamp_min(_EPS) + 1.0
        sep = torch.where(foil_q, tt_q, wide.expand_as(tt_q)).amin(dim=1)
        sep = torch.where(has_foil, sep, mean_off.expand_as(sep))
        sep_floor = torch.maximum(_SEP_FLOOR_REL * sep.median(),
                                  _SEP_FLOOR_ABS * mean_off)
        sep = sep.clamp_min(sep_floor).to(dtype=pred.dtype)
        del tt_q
        temp_q = (_TEMP_EP_FRAC * sep).unsqueeze(1)
        margin_q = _MARGIN_FRAC * sep
        tau_q = (_TAU_FRAC * sep).clamp_min(_NUM_EPS)

    dist_q = _sqdist(pred.index_select(0, query_idx), tgt, d)
    logits_q = (-dist_q / temp_q - log_mult_q).masked_fill(~sup_q, _NEG)
    den_q = torch.logsumexp(logits_q, dim=1)
    num_q = torch.logsumexp(logits_q.masked_fill(~pos_q, _NEG), dim=1)
    episode = (row_w * (den_q - num_q).clamp_min(0.0)).mean()

    with torch.no_grad():
        fill = dist_q.detach().amax().clamp_min(_EPS) + 1.0
    far = fill.expand_as(dist_q)
    d_true = torch.where(pos_q, dist_q, far).amin(dim=1)
    d_foil = torch.where(foil_q, dist_q, far).amin(dim=1)
    gap = F.softplus((d_true + margin_q - d_foil) / tau_q)
    gap = torch.where(has_foil, gap, torch.zeros_like(gap))
    board_margin = (row_w * gap).mean()

    return total + _W_EPISODE * episode + _W_MARGIN * board_margin
