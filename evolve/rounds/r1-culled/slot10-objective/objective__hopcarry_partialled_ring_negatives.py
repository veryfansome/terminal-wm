import torch
import torch.nn.functional as F

WANTS_CTX = True

NAME = "hopcarry_partialled_ring_negatives"
DESCRIPTION = (
    "The confusability-ring focal listwise contrastive, plus two ctx-only terms the ring loss "
    "cannot express. (1) HOP-CARRY: the episode segmentation is recovered exactly from the "
    "all-zero strict-causal prev at each sequence start, and every row is asked, over the "
    "softmax of its own squared-L2 distances to ALL strictly earlier observations of its own "
    "episode, to place its prediction on the earlier observation whose CONTENT matches the "
    "answer -- with the positive set restricted to steps at least two hops back and the row "
    "gated by how much better that far match is than the immediately preceding observation. "
    "(2) PARTIALLED RECONSTRUCTION: a per-batch dual ridge fit of the target on [cmd, prev, 1] "
    "is subtracted from both prediction and target, so the reconstruction term scores only the "
    "part of the observation that the command and the previous state cannot linearly explain."
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

_LAMBDA_TRACK = 1.0
_TRACK_TEMP = 0.125
_A_FRAC = 0.05
_G_FRAC = 0.10
_MIN_CAND = 2
_NEG = -1.0e4
_FAR = 1.0e4

_LAMBDA_COND = 0.5
_RIDGE = 1.0
_COND_MAX_N = 4096


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _sqdist(a, b, d):
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True)
    return (a2 + b2.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(d)


@torch.no_grad()
def _segments(prev):
    n = prev.shape[0]
    pos = torch.arange(n, device=prev.device)
    start = prev.abs().sum(dim=1) == 0
    if not bool(start.any()):
        return None, pos
    sid = (torch.cumsum(start.long(), dim=0) - 1).clamp_min(0)
    return sid, pos


@torch.no_grad()
def _partial_operator(cmd, prev, d):
    n = cmd.shape[0]
    z = torch.cat([cmd, prev, cmd.new_ones(n, 1)], dim=1) * (1.0 / float(d) ** 0.5)
    g = z @ z.t()
    lam = (g.diagonal().mean() * _RIDGE).clamp_min(1e-6)
    eye = torch.eye(n, device=cmd.device, dtype=cmd.dtype)
    m = lam * torch.linalg.solve(g + lam * eye, eye)
    if not bool(torch.isfinite(m).all()):
        return None
    return m


def _hop_track(dist2, tt, sid, pos, mean_off):
    with torch.no_grad():
        same = sid.unsqueeze(1) == sid.unsqueeze(0)
        delta = pos.unsqueeze(1) - pos.unsqueeze(0)
        cand = same & (delta >= 1)
        hop = same & (delta >= 2)
        live = (cand.sum(dim=1) >= _MIN_CAND) & (hop.sum(dim=1) >= 1)
        if not bool(live.any()):
            return None
        far = tt.new_full(tt.shape, _FAR)
        best = torch.where(hop, tt, far).min(dim=1).values
        d_prev = torch.where(same & (delta == 1), tt, far).min(dim=1).values
        tau_g = (_G_FRAC * mean_off).clamp_min(1e-6)
        tau_a = (_A_FRAC * mean_off).clamp_min(1e-6)
        gate = torch.sigmoid((d_prev - best) / tau_g)
        a_log = torch.where(
            hop, -(tt - best.unsqueeze(1)) / tau_a, tt.new_full(tt.shape, _NEG)
        )
        a = torch.softmax(a_log, dim=1)
        w = gate * live.to(tt.dtype)
        if float(w.sum()) <= 0.0:
            return None

    tau_q = (_TRACK_TEMP * mean_off).clamp_min(1e-6)
    q_log = torch.where(cand, -dist2 / tau_q, dist2.new_full(dist2.shape, _NEG))
    ce = -(a * F.log_softmax(q_log, dim=1)).sum(dim=1)
    return (w * ce).sum() / w.sum().clamp_min(1e-6)


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
    dist2 = _sqdist(pw, tw, d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = _sqdist(tw, tw, d)
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)
        dupmask = 1.0 - torch.exp(-tt / _DELTA)
        ring = (confus * dupmask).masked_fill(eye, 0.0)

        a_raw = 1.0 + _KAPPA * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        a_ring = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0)
        log_a = a_ring.clamp_min(1e-6).log()

        mass = ring.sum(dim=1)
        q_conf = ring / mass.clamp_min(1e-6).unsqueeze(1)
        conf_gate = mass / (mass + _GEPS)

    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q_conf * dist2).sum(dim=1)
    rep = (conf_gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    total = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep

    if ctx is None:
        return total

    cmd = ctx.get("cmd")
    prev = ctx.get("prev")
    if prev is None or prev.shape != pred.shape:
        return total
    prev = _clean(prev.detach())

    sid, pos = _segments(prev)
    if sid is not None and _LAMBDA_TRACK > 0.0:
        track = _hop_track(dist2, tt, sid, pos, mean_off)
        if track is not None and bool(torch.isfinite(track)):
            total = total + _LAMBDA_TRACK * track

    if cmd is None or cmd.shape != pred.shape or n > _COND_MAX_N or _LAMBDA_COND <= 0.0:
        return total
    cmd = _clean(cmd.detach())

    try:
        m = _partial_operator(cmd, prev, d)
    except Exception:
        m = None
    if m is None:
        return total

    pr = m @ pw
    tr = m @ tw
    with torch.no_grad():
        s2 = (tr.pow(2).sum(dim=1).mean() / float(d)).clamp_min(1e-6)
    mse_cond = (pr - tr).pow(2).sum(dim=1).mean() / float(d) / s2
    if bool(torch.isfinite(mse_cond)):
        total = total + _LAMBDA_COND * mse_cond
    return total
