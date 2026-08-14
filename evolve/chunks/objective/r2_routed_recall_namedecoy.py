import torch
import torch.nn.functional as F

NAME = "r2_routed_recall_namedecoy"
DESCRIPTION = (
    "Plain MSE to the standardized target as the backbone, plus a second regression term and a "
    "listwise softmax term that are both restricted to ROUTED-RECALL rows: rows whose target is "
    "an exact duplicate of another row's target inside a small duplicate group (group size <= 4, "
    "which separates repeated file contents from the large cluster of empty-output observations) "
    "AND whose command embedding is not the same command as its duplicate partner's, so the "
    "content reached this row by an intervening command rather than by re-issuing the same read. "
    "The listwise term draws its negatives only from a band of neighbouring rows in the flattened "
    "cmd-position order — the rows of the same trajectory, i.e. the other contents on the same "
    "board — and importance-weights those negatives by COMMAND-embedding cosine to the anchor's "
    "command, so the candidate whose own read command most resembles the anchor's is the hardest "
    "negative. A margin hinge in the eval's squared-L2 decision variable is applied against that "
    "single most command-similar in-band candidate. Per-dimension precision weighting shapes the "
    "distance space; all masks, weights and thresholds are detached."
)

WANTS_CTX = True

_W = 32
_CHUNK = 256
_GRP_MAX = 4
_EPS_DUP_FRAC = 0.005

_BETA = 0.5
_PEPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0

_TEMP = 0.25
_GAMMA = 1.0
_KAPPA = 4.0
_TAU_C = 0.05
_CMD_RAMP = 0.02
_RHO = 3.0

_W_LIST = 1.0
_W_RMSE = 0.5
_LAM_REP = 0.2
_MARGIN = 0.5
_TAU_R = 0.25

_NEG_INF = -1e9


def _precision_sqrt(pred, tgt):
    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _PEPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        return w.sqrt().unsqueeze(0)


def loss(pred, tgt, ctx=None):
    n, d = pred.shape
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 4 or ctx is None or "cmd" not in ctx:
        return mse_anchor

    sw = _precision_sqrt(pred, tgt)
    pw = pred * sw
    with torch.no_grad():
        tw = tgt * sw
        cn = F.normalize(ctx["cmd"], dim=1)
        tw_sq = (tw * tw).sum(dim=1)
        scale = (2.0 * (tw * tw).mean()).clamp_min(1e-8)
        eps_dup = (_EPS_DUP_FRAC * scale).clamp_min(1e-8)
        moved_all = torch.zeros(n, device=pred.device, dtype=pred.dtype)

    row_mse = ((pred - tgt) ** 2).mean(dim=1)
    pw_sq = (pw * pw).sum(dim=1)

    list_num = pred.new_zeros(())
    rep_num = pred.new_zeros(())
    wsum = pred.new_zeros(())

    for a0 in range(0, n, _CHUNK):
        a1 = min(a0 + _CHUNK, n)
        c0 = max(0, a0 - _W)
        c1 = min(n, a1 + _W)

        cand = tw[c0:c1]
        cand_sq = tw_sq[c0:c1]
        anch = pw[a0:a1]

        d2 = (pw_sq[a0:a1].unsqueeze(1) + cand_sq.unsqueeze(0)
              - 2.0 * (anch @ cand.t())).clamp_min(0.0) / float(d)

        with torch.no_grad():
            tt = (cand_sq.unsqueeze(1) + cand_sq.unsqueeze(0)
                  - 2.0 * (cand @ cand.t())).clamp_min(0.0) / float(d)
            dup_cc = tt < eps_dup
            grp_c = dup_cc.sum(dim=1)
            small_c = grp_c <= _GRP_MAX

            ai = torch.arange(a0, a1, device=pred.device)
            cj = torch.arange(c0, c1, device=pred.device)
            band = (ai.unsqueeze(1) - cj.unsqueeze(0)).abs() <= _W
            eye = ai.unsqueeze(1) == cj.unsqueeze(0)

            dup_ac = dup_cc[a0 - c0:a1 - c0]
            small_a = small_c[a0 - c0:a1 - c0]

            cmat = cn[a0:a1] @ cn[c0:c1].t()

            partner = dup_ac & (~eye) & band & small_c.unsqueeze(0) & small_a.unsqueeze(1)
            has_p = partner.any(dim=1)
            min_c = cmat.masked_fill(~partner, 2.0).amin(dim=1)
            moved = has_p.to(pred.dtype) * ((1.0 - min_c) / _CMD_RAMP).clamp(0.0, 1.0)
            moved_all[a0:a1] = moved

            neg = band & (~dup_ac) & (~eye) & small_c.unsqueeze(0) & small_a.unsqueeze(1)
            n_neg = neg.sum(dim=1)
            gate = (n_neg > 0).to(pred.dtype)

            s = F.softmax(cmat.masked_fill(~neg, _NEG_INF) / _TAU_C, dim=1)
            a_raw = 1.0 + _KAPPA * n_neg.unsqueeze(1).to(pred.dtype) * s
            log_a = (a_raw / (1.0 + _KAPPA)).clamp_min(1e-6).log() * neg.to(pred.dtype)

            allowed = neg | eye
            self_col = (ai - c0).unsqueeze(1)
            decoy_col = cmat.masked_fill(~neg, _NEG_INF).argmax(dim=1, keepdim=True)
            rw = (1.0 + _RHO * moved) * gate

        logits = (-d2 / _TEMP + log_a).masked_fill(~allowed, _NEG_INF)
        logp = F.log_softmax(logits, dim=1)
        nll = -logp.gather(1, self_col).squeeze(1)
        with torch.no_grad():
            focal = (1.0 - (-nll).exp().clamp(0.0, 1.0)).pow(_GAMMA)

        d_true = d2.gather(1, self_col).squeeze(1)
        d_dec = d2.gather(1, decoy_col).squeeze(1)
        rep = F.softplus((d_true + _MARGIN - d_dec) / _TAU_R)

        list_num = list_num + (rw * focal * nll).sum()
        rep_num = rep_num + (rw * rep).sum()
        wsum = wsum + rw.sum()

    listwise = list_num / wsum.clamp_min(1e-6)
    repulse = rep_num / wsum.clamp_min(1e-6)
    mv_mass = moved_all.sum()
    mse_routed = (moved_all * row_mse).sum() / mv_mass.clamp_min(1e-6)

    return (mse_anchor + _W_RMSE * mse_routed + _W_LIST * listwise + _LAM_REP * repulse)
