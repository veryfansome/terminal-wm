import torch
import torch.nn.functional as F

NAME = "r6_occupancy_transport_name_decoy"
DESCRIPTION = (
    "The decision-direction metric quotient contrastive, extended with two TRAJECTORY-LOCAL terms "
    "that treat a trajectory as a registry whose contents are conserved. Trajectory boundaries are "
    "recovered exactly from the strict-causal previous-observation channel, which is the zero "
    "vector at the first command of every sequence, and every command row is placed back into a "
    "padded (sequence, position) grid. (1) OCCUPANCY TRANSPORT: within one trajectory the rows are "
    "partitioned into content classes by exact target equivalence, each class represented by its "
    "earliest row, and the assignment of rows to contents is scored by an entropic optimal-"
    "transport plan in the decision-direction metric, computed by a few log-domain Sinkhorn "
    "iterations under BOTH marginals — uniform over the trajectory's rows and, per content, exactly "
    "the number of rows that truly carry it. The loss is the negative log conditional plan mass on "
    "each row's own content. A per-row softmax normalises only over candidates, so nothing stops "
    "every read of a trajectory from claiming the same popular content; the column marginal makes "
    "contents a conserved resource, so a read that claims a content another row must also hold pays "
    "for it, and the mass a content actually has must be placed somewhere. Rows whose target "
    "duplicates an earlier row of the same trajectory under a different command — the post-chain "
    "re-read — carry extra weight. (2) NAME DECOY: for exactly those re-read rows, the decoy is the "
    "earlier row of the same trajectory, of a different content class, whose command embedding is "
    "most similar — the content that this location's NAME used to hold rather than the content the "
    "chain routed into it. A scale-free hinge in the raw unweighted squared-L2 the eval ranks in "
    "requires (||pred - t_decoy||^2 - ||pred - t_true||^2) / ||t_true - t_decoy||^2, formed "
    "algebraically as 1 + 2 e.(t_true - t_decoy) / ||t_true - t_decoy||^2, to stay above a fixed "
    "fraction of one, gated off unless that command similarity stands out above the row's own "
    "earlier-command similarity distribution. Every mask, class, marginal, decoy choice and scale "
    "is detached and derived from the batch's own targets; separations are floored so a near-"
    "duplicate pair cannot amplify without bound, and the class-measured squared-error anchor is "
    "retained so a constant prediction cannot minimise the loss."
)

WANTS_CTX = True

_EQ_FRAC = 1e-3
_LAM_FRAC = 0.5
_DUP_FRAC = 0.025
_KAPPA = 4.0
_TEMP_FRAC = 0.125
_GAMMA = 1.0
_ANCHOR = 0.05

_ALPHA = 0.75
_WMIN = 0.4
_WMAX = 3.0

_LAMBDA_REP = 0.25
_RATIO = 0.5
_TAU = 0.2
_TAU_AGG = 0.5
_GAP_FLOOR_FRAC = 0.1

_GEPS = 1e-3
_EPS = 1e-6
_NUM_EPS = 1e-12
_NEG = -1e30

_W_OCC = 0.5
_W_DEC = 0.25
_SINK_ITERS = 3
_SEG_TEMP_FRAC = 0.25
_R_MAX = 96
_MIN_ROWS = 8
_BASE_ROW_W = 1.0
_REPEAT_ROW_W = 3.0
_SAME_CMD = 0.999
_GRP_MAX = 4.0
_DECOY_Z = 1.0
_GATE_TEMP = 0.05
_DEC_RATIO = 0.5
_DEC_TAU = 0.2
_DEC_FLOOR_FRAC = 0.1
_SNEG = -1.0e9


def _pair_sq(a, b, d):
    asq = (a * a).sum(dim=2, keepdim=True)
    bsq = (b * b).sum(dim=2).unsqueeze(1)
    return (asq + bsq - 2.0 * torch.bmm(a, b.transpose(1, 2))).clamp_min(0.0) / float(d)


def _grid(prev):
    n = prev.shape[0]
    device = prev.device
    starts = prev.detach().abs().amax(dim=1) == 0
    if not bool(starts[0].item()):
        return None
    seg = starts.long().cumsum(dim=0) - 1
    first = torch.nonzero(starts, as_tuple=False).squeeze(1)
    order = torch.arange(n, device=device)
    pos = order - first[seg]
    n_seg = int(first.numel())
    width = int(pos.max().item()) + 1
    if n_seg < 1 or width < 3 or width > _R_MAX:
        return None
    slot = torch.full((n_seg, width), -1, dtype=torch.long, device=device)
    slot[seg, pos] = order
    return slot


def _classes(tgt_grid, live, eq_tol, d):
    s, r, _ = tgt_grid.shape
    device = tgt_grid.device
    tf = tgt_grid.detach().float()
    tt = _pair_sq(tf, tf, d)
    pair_live = live.unsqueeze(2) & live.unsqueeze(1)
    same = (tt <= eq_tol) & pair_live
    eye = torch.eye(r, dtype=torch.bool, device=device).unsqueeze(0)
    same = same | (eye & live.unsqueeze(2))
    rank = (float(r) - torch.arange(r, device=device, dtype=torch.float32)).view(1, 1, r)
    first_eq = (same.float() * rank).argmax(dim=2)
    onehot = F.one_hot(first_eq, r).to(torch.float32) * live.unsqueeze(2).to(torch.float32)
    class_size = onehot.sum(dim=1)
    return same, first_eq, class_size


def _transport_nll(cost, live, first_eq, class_size, temp):
    s, r, _ = cost.shape
    rep = class_size > 0.0
    rows = live.sum(dim=1).clamp_min(1).to(torch.float32)
    log_rows = rows.log().unsqueeze(1)
    log_a = torch.where(live, -log_rows.expand(s, r), torch.full_like(log_rows.expand(s, r), _SNEG))
    log_b = torch.where(rep, class_size.clamp_min(1.0).log() - log_rows,
                        torch.full_like(class_size, _SNEG))
    log_a = log_a.to(dtype=cost.dtype)
    log_b = log_b.to(dtype=cost.dtype)

    usable = live.unsqueeze(2) & rep.unsqueeze(1)
    logits = (-cost / temp).masked_fill(~usable, _SNEG)

    f = torch.zeros(s, r, dtype=cost.dtype, device=cost.device)
    g = torch.zeros(s, r, dtype=cost.dtype, device=cost.device)
    for _ in range(_SINK_ITERS):
        g = log_b - torch.logsumexp(logits + f.unsqueeze(2), dim=1)
        f = log_a - torch.logsumexp(logits + g.unsqueeze(1), dim=2)

    log_plan = logits + f.unsqueeze(2) + g.unsqueeze(1)
    hit = log_plan.gather(2, first_eq.unsqueeze(2)).squeeze(2)
    return (log_a - hit).clamp_min(0.0)


def _decoy_hinge(pred_grid, tgt_grid, same, live, repeat, sim, earlier, mean_off, d):
    cand = earlier & (~same) & live.unsqueeze(1) & repeat.unsqueeze(2)
    have = cand.any(dim=2)
    cf = cand.to(torch.float32)
    cnt = cf.sum(dim=2)
    mu = (sim * cf).sum(dim=2) / cnt.clamp_min(1.0)
    var = (((sim - mu.unsqueeze(2)) ** 2) * cf).sum(dim=2) / cnt.clamp_min(1.0)
    sd = var.clamp_min(1e-8).sqrt()
    best, star = sim.masked_fill(~cand, -2.0).max(dim=2)
    gate = torch.sigmoid((best - mu - _DECOY_Z * sd) / _GATE_TEMP)
    gate = gate * have.to(torch.float32) * (cnt >= 2.0).to(torch.float32)
    if float(gate.sum().item()) <= 0.0:
        return None

    decoy = tgt_grid.gather(1, star.clamp_min(0).unsqueeze(2).expand(-1, -1, d))
    direction = tgt_grid - decoy
    with torch.no_grad():
        floor = (_DEC_FLOOR_FRAC * mean_off * float(d)).clamp_min(_EPS)
        den = (direction * direction).sum(dim=2).clamp_min(floor).to(dtype=pred_grid.dtype)
        coef = (2.0 / (_DEC_TAU * den))
        gate_w = gate.to(dtype=pred_grid.dtype)
    err = pred_grid - tgt_grid
    align = (err * direction).sum(dim=2)
    arg = float((_DEC_RATIO - 1.0) / _DEC_TAU) - align * coef
    pen = F.softplus(arg)
    return (gate_w * pen).sum() / gate_w.sum().clamp_min(_NUM_EPS)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    per_row_mse = (pred - tgt).pow(2).mean(dim=-1)
    if n < 2:
        return per_row_mse.mean()

    with torch.no_grad():
        tf = tgt.detach().float()
        t_sq = (tf * tf).sum(dim=1, keepdim=True)
        tt = (t_sq + t_sq.t() - 2.0 * (tf @ tf.t())).clamp_min(0.0) / float(d)
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        mean_all = (tt * (~eye).to(tt.dtype)).sum() / float(n * (n - 1))
        equivalent = (tt <= _EQ_FRAC * mean_all) | eye
        distinct = ~equivalent
        usable = bool(distinct.any())

    if not usable:
        return per_row_mse.mean()

    with torch.no_grad():
        class_size = equivalent.sum(dim=1).clamp_min(1).float()
        inv_class = class_size.reciprocal()
        row_measure = inv_class.sum().clamp_min(_NUM_EPS)
        pair_measure = inv_class.unsqueeze(1) * inv_class.unsqueeze(0) * distinct.float()
        pair_mass = pair_measure.sum().clamp_min(_NUM_EPS)
        mean_off = ((tt * pair_measure).sum() / pair_mass).clamp_min(_EPS)

        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        dup = (_DUP_FRAC * mean_off).clamp_min(_EPS)
        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / dup))
        ring = ring.masked_fill(equivalent, 0.0)
        ringw = ring * inv_class.unsqueeze(1) * inv_class.unsqueeze(0)

        gap_floor = (_GAP_FLOOR_FRAC * mean_off * float(d)).clamp_min(_EPS)
        gap_eff = (tt * float(d)).clamp_min(gap_floor)

        rn = ringw / gap_eff
        row_rn = rn.sum(dim=1, keepdim=True)
        v = 2.0 * ((row_rn * tf * tf).sum(dim=0) - ((rn @ tf) * tf).sum(dim=0))
        v = v.clamp_min(0.0)
        v = v / v.mean().clamp_min(_NUM_EPS)
        dim_w = v.clamp_min(_NUM_EPS).pow(_ALPHA).clamp(_WMIN, _WMAX)
        dim_w = dim_w / dim_w.mean().clamp_min(_NUM_EPS)
        if not torch.isfinite(dim_w).all():
            dim_w = torch.ones_like(dim_w)
        sqrt_w = dim_w.sqrt().unsqueeze(0).to(dtype=pred.dtype)

    pw = pred * sqrt_w
    tw = tgt * sqrt_w
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        ttw = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0).float() / float(d)
        mean_offw = ((ttw * pair_measure).sum() / pair_mass).clamp_min(_EPS)
        temp = (_TEMP_FRAC * mean_offw).clamp_min(_EPS).to(dtype=pred.dtype)

        neg_measure = inv_class.unsqueeze(0) * distinct.float()
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_measure).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_measure.sum(dim=1, keepdim=True).clamp_min(1e-6)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log().to(dtype=pred.dtype)
        log_class = (-class_size.log()).unsqueeze(0).to(dtype=pred.dtype)

    logits = -dist2 / temp + log_importance + log_class
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(distinct, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)
        row_w = inv_class.to(dtype=pred.dtype)
        measure = row_measure.to(dtype=pred.dtype)

    listwise = (row_w * focal.to(dtype=pred.dtype) * nll).sum() / measure

    err = pred - tgt
    align = (err * tgt).sum(dim=1, keepdim=True) - err @ tgt.t()

    with torch.no_grad():
        coef = (2.0 / (_TAU * gap_eff)).to(dtype=pred.dtype)
        ring_mass = ringw.sum(dim=1)
        q = ringw / ring_mass.clamp_min(_NUM_EPS).unsqueeze(1)
        log_q = q.clamp_min(_NUM_EPS).log().to(dtype=pred.dtype)
        no_ring = ringw <= 0.0
        gate = (ring_mass / (ring_mass + _GEPS)).to(dtype=pred.dtype)

    arg = float((_RATIO - 1.0) / _TAU) - align * coef
    pen = F.softplus(arg)
    z = (pen / _TAU_AGG + log_q).masked_fill(no_ring, _NEG)
    hardest = _TAU_AGG * torch.logsumexp(z, dim=1)
    repulsion = (row_w * gate * hardest).sum() / measure

    mse_anchor = (row_w * per_row_mse).sum() / measure

    total = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * repulsion

    if ctx is None or n < _MIN_ROWS:
        return total
    prev = ctx.get("prev") if isinstance(ctx, dict) else None
    cmd = ctx.get("cmd") if isinstance(ctx, dict) else None
    if prev is None or cmd is None:
        return total
    if prev.shape[0] != n or cmd.shape[0] != n or prev.shape[1] != d or cmd.shape[1] != d:
        return total

    slot = _grid(prev)
    if slot is None:
        return total

    live = slot >= 0
    safe = slot.clamp_min(0)
    s, r = slot.shape

    pred_grid = pred[safe]
    tgt_grid = tgt[safe]

    with torch.no_grad():
        eq_tol = (_EQ_FRAC * mean_all).to(dtype=tgt.dtype)
        same, first_eq, class_size_seg = _classes(tgt_grid, live, eq_tol.float(), d)
        cmd_grid = F.normalize(cmd[safe].detach().float(), dim=2)
        sim = torch.bmm(cmd_grid, cmd_grid.transpose(1, 2))
        idx_r = torch.arange(r, device=pred.device)
        earlier = (idx_r.view(1, 1, r) < idx_r.view(1, r, 1)) & live.unsqueeze(2) & live.unsqueeze(1)
        own_class = class_size_seg.gather(1, first_eq)
        repeat = ((same & earlier & (sim < _SAME_CMD)).any(dim=2) & live
                  & (own_class <= _GRP_MAX))
        row_weight = (_BASE_ROW_W + _REPEAT_ROW_W * repeat.to(torch.float32))
        row_weight = (row_weight * live.to(torch.float32)).to(dtype=pred.dtype)
        seg_temp = (_SEG_TEMP_FRAC * mean_offw).clamp_min(_EPS).to(dtype=pred.dtype)

    cost = _pair_sq(pred_grid * sqrt_w, tgt_grid * sqrt_w, d)
    occ_rows = _transport_nll(cost, live, first_eq, class_size_seg, seg_temp)
    occupancy = (row_weight * occ_rows).sum() / row_weight.sum().clamp_min(_NUM_EPS)
    if torch.isfinite(occupancy):
        total = total + _W_OCC * occupancy

    decoy = _decoy_hinge(pred_grid, tgt_grid, same, live, repeat, sim.to(dtype=pred.dtype),
                         earlier, mean_off, d)
    if decoy is not None and torch.isfinite(decoy):
        total = total + _W_DEC * decoy

    return total
