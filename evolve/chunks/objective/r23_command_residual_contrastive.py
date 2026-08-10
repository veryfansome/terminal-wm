import torch
import torch.nn.functional as F

WANTS_CTX = True

NAME = "r23_command_residual_contrastive"
DESCRIPTION = (
    "The r12 anti-retrieval ring loss (verbatim base) PLUS a within-command residual contrastive: "
    "strip the leave-one-out exact-command-conditional mean (the command-decodable component, via "
    "ctx['cmd']) from prediction and target and require each sibling row to retrieve its OWN content "
    "residual against its exact-command group in the eval's squared-L2 geometry. Command-decode "
    "(r_pred=0) is a near-maximizer of the added term, so the loss routes gradient onto the "
    "history-content the ordinary next-obs loss leaves unbanked — the quantity IMAG_CA rewards."
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

_CMD_EXACT = 1.0
_RTAU = 0.25
_DUP = 0.05
_LAMBDA_RESID = 0.5


def _pdmean_sq(a, b, d):
    """Return the [len(a), len(b)] matrix of per-dimension-mean squared distances."""
    an = (a * a).sum(1, keepdim=True)
    bn = (b * b).sum(1, keepdim=True)
    return (an + bn.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(d)


def loss(pred, tgt, ctx):
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

    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    base = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep

    cmd = ctx["cmd"]
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        cn = (cmd * cmd).sum(dim=1, keepdim=True)
        dcmd = (cn + cn.t() - 2.0 * (cmd @ cmd.t())).clamp_min(0.0)
        same_cmd = dcmd < _CMD_EXACT
        sib = same_cmd & ~eye
        sib_cnt = sib.sum(dim=1)

        cmd_hat = (sib.float() @ tgt) / sib_cnt.clamp_min(1).unsqueeze(1)

        tt_plain = _pdmean_sq(tgt, tgt, d)
        dup_off = (tt_plain < _DUP) & ~eye
        cols = same_cmd & ~dup_off
        divergent_sib = (same_cmd & ~dup_off & ~eye).sum(dim=1)
        row_gate = (divergent_sib >= 1).float()

    resid = pred.new_zeros(())
    denom = row_gate.sum()
    if float(denom) > 0.0:
        r_pred = pred - cmd_hat
        r_tgt = tgt - cmd_hat
        r_dist2 = _pdmean_sq(r_pred, r_tgt.detach(), d)
        rlogits = (-r_dist2 / _RTAU).masked_fill(~cols, float("-inf"))
        rlogp = F.log_softmax(rlogits, dim=1)
        rnll = -rlogp.diagonal()
        resid = (row_gate * rnll).sum() / denom.clamp_min(1.0)

    return base + _LAMBDA_RESID * resid
