import torch
import torch.nn.functional as F

NAME = "r23_partialled_identity_residual_contrast"
DESCRIPTION = (
    "Partials the command-predictable component out of both prediction and target before "
    "contrasting. A detached leave-one-out Nadaraya-Watson estimate of E[z_obs | cmd] is built "
    "from the batch with a sharp cosine kernel on the causal command embedding, shrunk by a "
    "neighbourhood-tightness factor, and subtracted from pred and tgt alike; the focal listwise "
    "L2-InfoNCE, the confusability-ring negatives and the repulsion hinge all run on those "
    "residuals, with in-batch negatives additionally boosted by command similarity and exact "
    "duplicates masked out. Rows are reweighted by the conditional variance of the target within "
    "their own command neighbourhood, and a residual-direction cosine term rewards getting the "
    "sign and heading of the history-carried deviation right. A raw-space listwise term and an "
    "MSE anchor are retained so calibration and anti-collapse do not depend on the kernel."
)

WANTS_CTX = True

_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_KAPPA_RING = 4.0
_DELTA = 0.05
_LAM_FRAC = 0.5
_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25
_GEPS = 1e-3

_TAU_CMD = 0.08
_G_POW = 4.0
_KAPPA_ROW = 3.0
_ROW_MAX = 6.0
_KAPPA_CMD = 6.0
_DUP_T2 = 0.01
_RAW_W = 0.35
_DIR_W = 0.15


def _dim_precision(pred, tgt):
    mse_d = ((pred - tgt) ** 2).mean(dim=0)
    w = (1.0 / (mse_d + _EPS)).pow(_BETA)
    w = w / w.mean().clamp_min(1e-12)
    w = w.clamp(_WMIN, _WMAX)
    w = w / w.mean().clamp_min(1e-12)
    return w.sqrt().unsqueeze(0)


def _pairwise(a, b, d):
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True)
    return (a2 + b2.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(d)


def _listwise(dist2, log_a, labels):
    logits = -dist2 / _TEMP + log_a
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    return focal * nll


def _raw_only(pred, tgt, n, d):
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 3:
        return mse_anchor
    with torch.no_grad():
        sw = _dim_precision(pred, tgt)
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
    pw, tw = pred * sw, tgt * sw
    draw = _pairwise(pw, tw, d)
    with torch.no_grad():
        tt = _pairwise(tw, tw, d)
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        ring = (torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / _DELTA))).masked_fill(eye, 0.0)
        a_raw = 1.0 + _KAPPA_RING * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        log_a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0).clamp_min(1e-6).log()
    labels = torch.arange(n, device=pred.device)
    return _listwise(draw, log_a, labels).mean() + _ANCHOR * mse_anchor


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    device = pred.device
    cmd = None if ctx is None else ctx.get("cmd", None)
    if cmd is None or cmd.shape != pred.shape:
        return _raw_only(pred, tgt, n, d)

    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 3:
        return mse_anchor

    labels = torch.arange(n, device=device)
    eye = torch.eye(n, dtype=torch.bool, device=device)

    with torch.no_grad():
        sw = _dim_precision(pred, tgt)

        cu = F.normalize(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1)
        sim = (cu @ cu.t()).clamp(-1.0, 1.0)
        q = torch.softmax(sim.masked_fill(eye, float("-inf")) / _TAU_CMD, dim=1)
        q = torch.nan_to_num(q, nan=0.0)

        mhat = q @ tgt
        tightness = (q * sim.masked_fill(eye, 0.0)).sum(dim=1).clamp(0.0, 1.0).pow(_G_POW)
        shrink = tightness.unsqueeze(1)
        centre = shrink * mhat

        t_sq = (tgt * tgt).sum(dim=1)
        cvar = (((q @ t_sq) - (mhat * mhat).sum(dim=1)).clamp_min(0.0) / float(d))
        vbar = (tightness * cvar).sum() / tightness.sum().clamp_min(1e-6)
        row_w = 1.0 + _KAPPA_ROW * tightness * (cvar / vbar.clamp_min(1e-6))
        row_w = row_w.clamp(max=_ROW_MAX)
        row_w = row_w / row_w.mean().clamp_min(1e-6)

        resid_energy = ((tgt - centre) ** 2).mean(dim=1)
        dir_gate = tightness * (resid_energy / (resid_energy + vbar.clamp_min(1e-6)))

    rp = (pred - centre) * sw
    rt = (tgt - centre) * sw
    pw = pred * sw
    tw = tgt * sw

    dres = _pairwise(rp, rt, d)
    draw = _pairwise(pw, tw, d)

    with torch.no_grad():
        tt = _pairwise(tw, tw, d)
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        ring = (torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / _DELTA))).masked_fill(eye, 0.0)
        cmd_boost = sim.clamp_min(0.0).pow(2).masked_fill(eye, 0.0)
        a_raw = 1.0 + _KAPPA_RING * ring + _KAPPA_CMD * cmd_boost
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        log_a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0).clamp_min(1e-6).log()
        log_a = log_a.masked_fill((tt < _DUP_T2) & (~eye), float("-inf"))

        mass = ring.sum(dim=1)
        q_conf = ring / mass.clamp_min(1e-6).unsqueeze(1)
        rep_gate = mass / (mass + _GEPS)

    listwise_res = (row_w * _listwise(dres, log_a, labels)).mean()
    listwise_raw = _listwise(draw, log_a, labels).mean()

    rpn = rp * torch.rsqrt((rp * rp).sum(dim=1, keepdim=True).clamp_min(1e-8))
    rtn = rt * torch.rsqrt((rt * rt).sum(dim=1, keepdim=True).clamp_min(1e-8))
    cos = (rpn * rtn).sum(dim=1).clamp(-1.0, 1.0)
    direction = (row_w * dir_gate * (1.0 - cos)).sum() / dir_gate.sum().clamp_min(1e-6)

    d_true = dres.diagonal()
    d_conf = (q_conf * dres).sum(dim=1)
    rep = (rep_gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    return (
        listwise_res
        + _RAW_W * listwise_raw
        + _ANCHOR * mse_anchor
        + _DIR_W * direction
        + _LAMBDA_REP * rep
    )
