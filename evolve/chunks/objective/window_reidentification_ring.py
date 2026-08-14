import torch
import torch.nn.functional as F

WANTS_CTX = True

NAME = "window_reidentification_ring"
DESCRIPTION = (
    "The confusability-ring precision-weighted focal listwise contrastive of the parent lineage, "
    "plus a WINDOW RE-IDENTIFICATION term mined from the batch. The strict-causal previous-"
    "observation channel is used to recover trajectory boundaries in the flattened command rows "
    "(a row whose previous observation is exactly zero starts a trajectory), giving a per-row "
    "trajectory id and step order. A row is flagged as a re-read when its target duplicates the "
    "target of an EARLIER row of the SAME trajectory, the two command embeddings are not "
    "near-identical, and the shared target is rare in the batch, so observations that repeat "
    "everywhere (empty output of a mutating command, boilerplate listings) are excluded. At each "
    "flagged row the prediction is trained with a cross-entropy over squared-L2 distances to every "
    "earlier target of the same trajectory, near-duplicates of the positive removed, labelled by "
    "the earlier occurrence: an N-way forced choice among the contents that trajectory has already "
    "shown, in the same decision variable the retrieval arms use. Flagged rows are also up-weighted "
    "inside the listwise term. Train-time only and target-side only: nothing is added to the "
    "model's input. Anti-collapse safe through the squared-error anchor and the listwise term."
)

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

_DUP_FRAC = 0.06
_CMD_FRAC = 0.02
_DUP_CAP = 4
_MIN_CAND = 3
_MAX_ROWS = 512
_N_CAP = 8192
_WRI_TEMP = 0.25
_LAMBDA_WRI = 0.5
_ROW_BOOST = 2.0
_NEG = -1e9


def _pdist2(a, b, d):
    a_sq = (a * a).sum(dim=1, keepdim=True)
    b_sq = (b * b).sum(dim=1, keepdim=True)
    return (a_sq + b_sq.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(d)


def _trajectory_ids(prev):
    n = prev.shape[0]
    starts = prev.abs().sum(dim=1) <= 0.0
    starts[0] = True
    return starts.long().cumsum(0)


@torch.no_grad()
def _mine_reidentification(tgt, cmd_emb, prev, d):
    n = tgt.shape[0]
    device = tgt.device
    empty = torch.zeros(0, dtype=torch.long, device=device)

    seg = _trajectory_ids(prev)
    idx = torch.arange(n, device=device)
    same = seg.unsqueeze(1) == seg.unsqueeze(0)
    earlier = idx.unsqueeze(0) < idx.unsqueeze(1)
    cand = same & earlier

    tt = _pdist2(tgt, tgt, d)
    mean_t = (tt.sum() / float(n * (n - 1))).clamp_min(_EPS)
    dup = tt <= (_DUP_FRAC * mean_t)

    cc = _pdist2(cmd_emb, cmd_emb, d)
    mean_c = (cc.sum() / float(n * (n - 1))).clamp_min(_EPS)
    cmd_apart = cc >= (_CMD_FRAC * mean_c)

    rare = dup.sum(dim=1) <= _DUP_CAP
    src = cand & dup & cmd_apart & rare.unsqueeze(1)
    flag = src.any(dim=1)

    rank = (float(n) - idx.to(tgt.dtype)).unsqueeze(0)
    first_src = (src.to(tgt.dtype) * rank).argmax(dim=1)

    rows = torch.nonzero(flag, as_tuple=False).squeeze(1)
    if rows.numel() == 0:
        return flag, empty, empty, None
    if rows.numel() > _MAX_ROWS:
        rows = rows[:_MAX_ROWS]

    pos = first_src[rows]
    keep = cand[rows] & (~dup[pos])
    keep[torch.arange(rows.numel(), device=device), pos] = True

    ok = keep.sum(dim=1) >= _MIN_CAND
    if not bool(ok.any()):
        return flag, empty, empty, None
    return flag, rows[ok], pos[ok], keep[ok]


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

    flag = None
    rows = None
    pos = None
    keep = None
    if isinstance(ctx, dict) and "cmd" in ctx and "prev" in ctx and 4 <= n <= _N_CAP:
        cmd_emb = ctx["cmd"]
        prev = ctx["prev"]
        if cmd_emb.shape == pred.shape and prev.shape == pred.shape:
            flag, rows, pos, keep = _mine_reidentification(tgt.detach(), cmd_emb.detach(),
                                                           prev.detach(), d)

    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
        if flag is not None:
            boost = 1.0 + _ROW_BOOST * flag.to(pred.dtype)
        else:
            boost = torch.ones(n, device=pred.device, dtype=pred.dtype)
        row_w = (boost / boost.mean().clamp_min(1e-12)) * focal
    listwise = (row_w * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    total = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep

    if keep is not None and rows.numel() > 0:
        d2_row = _pdist2(pred[rows], tgt, d)
        wri_logits = (-d2_row / _WRI_TEMP).masked_fill(~keep, _NEG)
        wri = F.cross_entropy(wri_logits, pos)
        total = total + _LAMBDA_WRI * wri

    return total
