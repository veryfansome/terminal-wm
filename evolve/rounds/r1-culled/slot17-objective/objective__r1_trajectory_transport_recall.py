import torch
import torch.nn.functional as F

NAME = "r1_trajectory_transport_recall"
DESCRIPTION = (
    "Ring-negative focal listwise (carried) PLUS a WITHIN-TRAJECTORY listwise whose candidate "
    "set is exactly the observations of the same shell window, PLUS per-row transport weights "
    "u = 1 + kappa * recall * surprise * depth built from the nearest EARLIER same-window target "
    "(recall), its distance from the immediately preceding target (surprise) and how many steps "
    "back it sits (depth), PLUS a prev-centred differential cosine. Segments are recovered from "
    "the exactly-zero strict-causal prev row that marks each window's first step."
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

_SEG_EPS = 1e-6
_TRAJ_TEMP = 0.25
_TRAJ_W = 1.0
_TRANSPORT_KAPPA = 3.0
_SIG_FRAC = 0.35
_SIG_MIN = 1e-3
_HOP_TAU = 1.5
_DIFF_W = 0.15
_BIG = 1e9


def _segment_structure(tt, prev):
    n = tt.shape[0]
    device = tt.device
    idx = torch.arange(n, device=device)

    start = (prev.pow(2).sum(dim=1) <= _SEG_EPS).clone()
    start[0] = True
    seg = torch.cumsum(start.long(), dim=0) - 1
    same_seg = seg.unsqueeze(1) == seg.unsqueeze(0)
    earlier = same_seg & (idx.unsqueeze(1) > idx.unsqueeze(0))
    has_earlier = earlier.any(dim=1)

    pair_cnt = earlier.sum().clamp_min(1).to(tt.dtype)
    sigma = (_SIG_FRAC * (tt * earlier.to(tt.dtype)).sum() / pair_cnt).clamp_min(_SIG_MIN)

    dmin, jstar = tt.masked_fill(~earlier, _BIG).min(dim=1)
    dmin = torch.where(has_earlier, dmin, torch.full_like(dmin, _BIG))
    hop = (idx - jstar).clamp_min(1).to(tt.dtype)

    dprev = tt.gather(1, (idx - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    dprev = torch.where(has_earlier, dprev, torch.zeros_like(dprev))

    recall = torch.exp(-dmin / sigma)
    surprise = 1.0 - torch.exp(-dprev / sigma)
    depth = 1.0 - torch.exp(-(hop - 1.0) / _HOP_TAU)

    u = 1.0 + _TRANSPORT_KAPPA * recall * surprise * depth
    u = u / u.mean().clamp_min(1e-6)
    return same_seg, u, sigma


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

    prev = None
    if isinstance(ctx, dict):
        p = ctx.get("prev")
        if p is not None and tuple(p.shape) == (n, d):
            prev = p

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

        if prev is None:
            same_seg = None
            u = pred.new_ones(n)
            sigma = mean_off
            log_traj = None
        else:
            same_seg, u, sigma = _segment_structure(tt, prev)
            log_traj = dupmask.clamp_min(1e-6).log().masked_fill(eye, 0.0)

    labels = torch.arange(n, device=pred.device)

    logits = -dist2 / _TEMP + log_a
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (u * focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    total = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep

    if same_seg is not None:
        tlogits = (-dist2 / _TRAJ_TEMP + log_traj).masked_fill(~same_seg, -_BIG)
        tlogp = F.log_softmax(tlogits, dim=1)
        tnll = -tlogp.gather(1, labels[:, None]).squeeze(1)
        total = total + _TRAJ_W * (u * tnll).mean()

    if prev is not None:
        res_t = tgt - prev
        res_p = pred - prev
        with torch.no_grad():
            mag = res_t.pow(2).mean(dim=1)
            gate_d = mag / (mag + sigma)
            live = (prev.pow(2).sum(dim=1) > _SEG_EPS).to(pred.dtype)
        cos = F.cosine_similarity(res_p, res_t, dim=1).clamp(-1.0, 1.0)
        total = total + _DIFF_W * (u * gate_d * live * (1.0 - cos)).mean()

    return total
