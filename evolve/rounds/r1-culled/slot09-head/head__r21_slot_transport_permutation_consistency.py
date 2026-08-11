import math

import torch
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as BASE

NAME = "r21_slot_transport_permutation_consistency"
DESCRIPTION = (
    "The R18 forward-model-consistency aux VERBATIM plus a slot-transport permutation aux for "
    "the co-designed transport arch. A label-free miner finds CONTENT REAPPEARANCE: positions "
    "whose observation nearly repeats exactly ONE earlier observation of the same trajectory "
    "(a nearest/second-nearest ratio test that rejects boilerplate such as the empty output of a "
    "move) while the querying command is NOT the earlier command. On those positions the read "
    "is trained, in the eval's squared-L2 geometry, to pick the reappearing content out of a "
    "candidate set made of the OTHER observations of the SAME trajectory plus the in-batch "
    "positives, with near-duplicate candidates masked. A periodic interventional term contrasts "
    "the native read against the same recurrence with the move gate forced to zero, so the "
    "transport must be what earns the retrieval. Both new terms are 0.0 on archs without the "
    "transport surface."
)

_EPS = 1e-8

_DEFAULTS = dict(BASE._DEFAULTS)
_DEFAULTS.update(
    {
        "tp_weight": 1.0,
        "tp_ramp_steps": 300.0,
        "tp_max_examples": 256.0,
        "tp_match_tau": 0.35,
        "tp_novel_tau": 0.02,
        "tp_uniq_tau": 0.25,
        "tp_w_floor": 0.01,
        "tp_tau": 0.25,
        "tp_dup_delta": 0.05,
        "tp_mse": 0.05,
        "iv_weight": 0.10,
        "iv_every": 4.0,
    }
)


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _pdist2(a, b):
    d2 = a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1).unsqueeze(0) - 2.0 * a @ b.t()
    return d2.clamp_min(0.0) / float(a.size(1))


def _rowdist2(a, rows):
    d2 = (
        a.pow(2).sum(-1, keepdim=True)
        + rows.pow(2).sum(-1)
        - 2.0 * torch.einsum("sd,snd->sn", a, rows)
    )
    return d2.clamp_min(0.0) / float(a.size(-1))


@torch.no_grad()
def _mine_reappearance(obs, cmd, valid, match_tau, novel_tau, uniq_tau, w_floor):
    B, n, dim = obs.shape
    device = obs.device
    o = _clean(obs)
    c = _clean(cmd)

    sq = o.pow(2).sum(-1)
    d2 = (sq.unsqueeze(2) + sq.unsqueeze(1) - 2.0 * torch.bmm(o, o.transpose(1, 2)))
    d2 = d2.clamp_min(0.0) / float(dim)

    pos = torch.arange(n, device=device)
    earlier = pos.unsqueeze(0) < pos.unsqueeze(1)
    ok = earlier.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)

    dm = torch.where(ok, d2, torch.full_like(d2, 1e9))
    min_d, best_t = dm.min(dim=2)
    has = ok.any(dim=2)
    match = torch.exp(-min_d.clamp_min(0.0) / float(match_tau))

    other = (~torch.eye(n, dtype=torch.bool, device=device)).unsqueeze(0)
    other = other & valid.unsqueeze(1) & valid.unsqueeze(2)
    da = torch.where(other, d2, torch.full_like(d2, 1e9))
    two = torch.topk(da, min(2, n), dim=2, largest=False).values
    near1 = two[..., 0]
    near2 = two[..., 1] if two.size(2) > 1 else torch.full_like(near1, 1e9)
    unique = 1.0 - torch.exp(-(near2 - near1).clamp(0.0, 1e6) / float(uniq_tau))

    cu = c * torch.rsqrt(c.pow(2).sum(-1, keepdim=True).clamp_min(_EPS))
    cs = torch.bmm(cu, cu.transpose(1, 2))
    cs_bt = torch.gather(cs, 2, best_t.unsqueeze(-1)).squeeze(-1)
    novel = 1.0 - torch.exp(-(1.0 - cs_bt).clamp_min(0.0) / float(novel_tau))

    w = match * unique * novel
    keep = has & valid & torch.isfinite(w) & (w > float(w_floor))
    nz = torch.nonzero(keep, as_tuple=False)
    sel_b = nz[:, 0]
    sel_j = nz[:, 1]
    return sel_b, sel_j, w[sel_b, sel_j]


def _transport_term(cfg, batch, net, device, ramp):
    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    B, maxn = valid.shape
    if maxn < 3:
        return 0.0

    obs = tok[:, 1::2, :][:, :maxn]
    cmd = tok[:, 0::2, :][:, :maxn]

    sel_b, sel_j, w = _mine_reappearance(
        obs,
        cmd,
        valid,
        float(cfg["tp_match_tau"]),
        float(cfg["tp_novel_tau"]),
        float(cfg["tp_uniq_tau"]),
        float(cfg["tp_w_floor"]),
    )
    if sel_b.numel() < 4:
        return 0.0
    cap = int(cfg["tp_max_examples"])
    if sel_b.numel() > cap:
        w, order = torch.topk(w, cap)
        sel_b = sel_b[order]
        sel_j = sel_j[order]

    w = (w.to(tok.dtype) / w.sum().clamp_min(_EPS)).detach()

    reads = net.transport_reads(tok, batch["types"], batch["key_pad"], 1.0)
    if reads.size(1) < maxn:
        return 0.0
    pred = _clean(reads[:, :maxn, :][sel_b, sel_j])

    target = _clean(obs[sel_b, sel_j]).detach()
    rows = _clean(obs[sel_b]).detach()
    row_valid = valid[sel_b]
    S = pred.size(0)

    cross = _pdist2(pred, target)
    inrow = _rowdist2(pred, rows)

    dup = float(cfg["tp_dup_delta"])
    with torch.no_grad():
        eye = torch.eye(S, dtype=torch.bool, device=pred.device)
        dup_cross = (_pdist2(target, target) < dup) & ~eye
        dup_row = (~row_valid) | (_rowdist2(target, rows) < dup)

    tau = float(cfg["tp_tau"])
    logits = torch.cat([-cross / tau, -inrow / tau], dim=1)
    logits = logits.masked_fill(torch.cat([dup_cross, dup_row], dim=1), -1e30)
    labels = torch.arange(S, device=pred.device)
    nll = -torch.log_softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)

    mse = (pred - target).pow(2).mean(dim=-1)
    total = (w * nll).sum() + float(cfg["tp_mse"]) * (w * mse).sum()

    iv_w = float(cfg["iv_weight"])
    every = max(1, int(cfg["iv_every"]))
    if iv_w > 0.0 and int(cfg["_tp_step"]) % every == 0:
        reads_cf = net.transport_reads(tok, batch["types"], batch["key_pad"], 0.0)
        pred_cf = _clean(reads_cf[:, :maxn, :][sel_b, sel_j])
        d_native = (pred - target).pow(2).mean(dim=-1)
        d_frozen = (pred_cf - target).pow(2).mean(dim=-1)
        total = total + iv_w * (w * F.softplus((d_native - d_frozen) / tau)).sum()

    return float(cfg["tp_weight"]) * ramp * total


def wrap(net, D, **params):
    base_params = {k: params[k] for k in BASE._DEFAULTS if k in params}
    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    cfg["D"] = int(D)
    cfg["_tp_step"] = 0
    cfg["_base"] = BASE.wrap(net, D, **base_params)
    cfg["_tp_disabled"] = not callable(getattr(net, "transport_reads", None))
    return cfg


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0

    total = BASE.aux_loss(cfg.get("_base"), batch, net, device)

    if cfg.get("_tp_disabled", True) or float(cfg["tp_weight"]) <= 0.0:
        return total
    if not BASE._interleave_layout_ok(batch):
        return total

    cfg["_tp_step"] = int(cfg.get("_tp_step", 0)) + 1
    ramp = BASE._smoothstep(cfg["_tp_step"] / max(1.0, float(cfg["tp_ramp_steps"])))
    if ramp <= 0.0:
        return total

    term = _transport_term(cfg, batch, net, device, ramp)
    if torch.is_tensor(term):
        if not bool(torch.isfinite(term).item()):
            return total
        total = total + term
    return total


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        vals["tp_weight"] >= 0.0,
        vals["tp_ramp_steps"] >= 1.0,
        vals["tp_max_examples"] >= 4.0,
        vals["tp_match_tau"] > 0.0,
        vals["tp_novel_tau"] > 0.0,
        vals["tp_uniq_tau"] > 0.0,
        vals["tp_w_floor"] >= 0.0,
        vals["tp_tau"] > 0.0,
        vals["tp_dup_delta"] >= 0.0,
        vals["tp_mse"] >= 0.0,
        vals["iv_weight"] >= 0.0,
        vals["iv_every"] >= 1.0 and float(vals["iv_every"]).is_integer(),
    ]
    return all(checks) and BASE.leak_safe(mod, params or {})
