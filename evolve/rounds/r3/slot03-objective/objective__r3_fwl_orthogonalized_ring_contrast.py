import torch
import torch.nn.functional as F

NAME = "r3_fwl_orthogonalized_ring_contrast"
DESCRIPTION = (
    "Precision-weighted focal listwise L2 contrastive with confusability-ring negatives whose "
    "NEGATIVE geometry is command-orthogonalized. Each minibatch cross-fits a detached linear "
    "kernel-ridge map from the causal command embedding (plus a bias) to the target on the "
    "opposite parity half of the rows, and subtracts rho times each row's own fitted value from "
    "the prediction and from every candidate target before distances are formed. The positive "
    "distance is algebraically unchanged by this; an in-batch negative is made hard exactly to "
    "the extent that the difference between its observation and the anchor's is NOT explained by "
    "the difference between their two commands. Rows whose command-unexplained residual carries "
    "little energy are down-weighted as anchors, exact-duplicate targets are pooled into one "
    "multi-positive class with inverse-class-size mass and excluded from the ring, the "
    "temperature / ring / duplicate / margin scales are all fractions of the batch's mean "
    "off-diagonal distance so the softmax stays at a fixed sharpness however much the "
    "orthogonalization compresses the space, and an "
    "untouched raw squared-error anchor keeps a constant prediction from minimizing the loss."
)

WANTS_CTX = True

_TEMP_FRAC = 0.125
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_KAPPA = 4.0
_LAM_FRAC = 0.5
_DUP_FRAC = 0.025
_EQ_FRAC = 1e-3
_LAMBDA_REP = 0.1
_MARGIN_FRAC = 0.25
_TAU_FRAC = 0.125
_GEPS = 1e-3

_RHO = 0.9
_RIDGE = 0.1
_FIT_ROWS = 1024
_INFO_FLOOR = 0.25
_NUM_EPS = 1e-12


def _ridge_apply(basis, target, fit_idx, apply_idx, ridge):
    zs = basis.index_select(0, fit_idx)
    ys = target.index_select(0, fit_idx)
    gram = zs @ zs.t()
    m = gram.shape[0]
    scale = gram.diagonal().mean().clamp_min(1e-6)
    reg = gram + (ridge * scale) * torch.eye(m, device=gram.device, dtype=gram.dtype)
    try:
        alpha = torch.linalg.solve(reg, ys)
    except Exception:
        return None
    if not torch.isfinite(alpha).all():
        return None
    fitted = (basis.index_select(0, apply_idx) @ zs.t()) @ alpha
    if not torch.isfinite(fitted).all():
        return None
    return fitted


def _crossfit_nuisance(basis, target, fit_rows, ridge):
    n = basis.shape[0]
    idx = torch.arange(n, device=basis.device)
    parity = idx % 2
    out = torch.zeros_like(target)
    for side in (0, 1):
        apply_idx = idx[parity == side]
        pool = idx[parity != side]
        if apply_idx.numel() == 0:
            continue
        if pool.numel() < 8:
            return None
        stride = max(1, -(-int(pool.numel()) // max(1, int(fit_rows))))
        fit_idx = pool[::stride]
        if fit_idx.numel() < 8:
            fit_idx = pool
        fitted = _ridge_apply(basis, target, fit_idx, apply_idx, ridge)
        if fitted is None:
            return None
        out.index_copy_(0, apply_idx, fitted)
    return out


def _nuisance_shift(pred, tgt, ctx):
    if ctx is None or not hasattr(ctx, "get"):
        return None
    n = pred.shape[0]
    cmd = ctx.get("cmd")
    if cmd is None or cmd.dim() != 2 or cmd.shape[0] != n:
        return None
    with torch.no_grad():
        cmd = cmd.detach().to(dtype=torch.float32)
        if not torch.isfinite(cmd).all():
            return None
        basis = torch.cat([cmd, torch.ones(n, 1, device=cmd.device, dtype=cmd.dtype)], dim=1)
        fitted = _crossfit_nuisance(basis, tgt.detach().to(dtype=torch.float32),
                                    _FIT_ROWS, _RIDGE)
    if fitted is None:
        return None
    return (_RHO * fitted).to(dtype=pred.dtype)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    raw_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return raw_anchor

    shift = _nuisance_shift(pred, tgt, ctx)
    if shift is None:
        pres, tres = pred, tgt
    else:
        pres, tres = pred - shift, tgt - shift

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(_NUM_EPS)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(_NUM_EPS)
        sw = w.sqrt().unsqueeze(0)

    pw = pres * sw
    tw = tres * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = (pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())).clamp_min(0.0) / float(d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        pair_n = float(n * (n - 1))

        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / pair_n).clamp_min(_EPS)

        rw = tgt * sw
        rw_sq = (rw * rw).sum(dim=1, keepdim=True)
        tt_raw = (rw_sq + rw_sq.t() - 2.0 * (rw @ rw.t())).clamp_min(0.0) / float(d)
        mean_raw = (tt_raw.sum() / pair_n).clamp_min(_EPS)

        temp = (_TEMP_FRAC * mean_off).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        dup_scale = (_DUP_FRAC * mean_raw).clamp_min(_NUM_EPS)
        margin = _MARGIN_FRAC * mean_off
        tau = (_TAU_FRAC * mean_off).clamp_min(_EPS)

        equivalent = (tt_raw <= _EQ_FRAC * mean_raw) | eye
        distinct = ~equivalent
        class_size = equivalent.sum(dim=1).clamp_min(1).to(dtype=pred.dtype)
        inv_class = class_size.reciprocal()

        ring = torch.exp(-tt / lam) * (1.0 - torch.exp(-tt_raw / dup_scale))
        ring = ring.masked_fill(equivalent, 0.0)

        neg_mass = distinct.to(dtype=pred.dtype)
        a_raw = 1.0 + _KAPPA * ring
        a_mean = (a_raw * neg_mass).sum(dim=1, keepdim=True)
        a_mean = a_mean / neg_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        importance = (a_raw / a_mean.clamp_min(1e-6)).masked_fill(equivalent, 1.0)
        log_importance = importance.clamp_min(1e-6).log()
        log_class = -class_size.clamp_min(1.0).log().unsqueeze(0)

        mass = ring.sum(dim=1)
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GEPS)

        energy = (tres * tres).mean(dim=1)
        med = energy.median().clamp_min(_NUM_EPS)
        row_w = energy / (energy + _INFO_FLOOR * med)
        row_w = row_w * inv_class
        row_w = row_w / row_w.mean().clamp_min(_NUM_EPS)

    logits = -dist2 / temp + log_importance + log_class
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(distinct, float("-inf")), dim=1)
    nll = (log_den - log_num).clamp_min(0.0)

    with torch.no_grad():
        focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

    listwise = (row_w * focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (row_w * gate * F.softplus((d_true + margin - d_conf) / tau)).mean()

    return listwise + _ANCHOR * raw_anchor + _LAMBDA_REP * rep
