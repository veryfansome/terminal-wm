import math

import torch

NAME = "r23_slot_transport_consistency"
DESCRIPTION = (
    "Train-only transport consistency for the shared latent-transition operator, mined by EXACT "
    "path-slot code match rather than command-embedding cosine: for a step k whose destination "
    "slot differs from its source slot, i is the latest earlier step reading the same source and "
    "j the earliest later step reading k's destination (or its directory-join), and "
    "f(obs_i, cmd_k) is required to reproduce obs_j. Supervises content preservation across an "
    "address change, which is what a multi-hop chain composes. Silently disabled (0.0) when the "
    "batch carries no slot codes or the arch has no transition operator."
)

_DEFAULTS = {
    "match_thresh": 0.999,
    "max_examples": 512,
    "cos_weight": 0.15,
    "mse_weight": 0.03,
    "aux_weight": 1.0,
    "ramp_steps": 400,
}

_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _layout_ok(b):
    for k in ("tok", "types", "key_pad", "tgt", "cmd_mask", "slot_code", "slot_valid"):
        if k not in b:
            return False
    tok, tgt, cmd_mask = b["tok"], b["tgt"], b["cmd_mask"]
    if tok.dim() != 3 or tgt.dim() != 3 or b["slot_code"].dim() != 4:
        return False
    if tok.shape[1] != 2 * tgt.shape[1]:
        return False
    if b["slot_code"].shape[:2] != cmd_mask.shape or b["slot_valid"].shape[:2] != cmd_mask.shape:
        return False
    return b["slot_code"].shape[2] >= 5


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    return cfg


@torch.no_grad()
def _mine(code, valid, cmd_mask, thresh):
    B, n, _, _ = code.shape
    device = code.device
    src = _unit(code[:, :, 0])
    dst = _unit(code[:, :, 2])
    joi = _unit(code[:, :, 4])

    live = cmd_mask.bool()
    src_ok = valid[:, :, 0].bool() & live
    dst_ok = valid[:, :, 2].bool() & live

    pos = torch.arange(n, device=device)
    lower = (pos.unsqueeze(1) > pos.unsqueeze(0)).unsqueeze(0)
    upper = (pos.unsqueeze(1) < pos.unsqueeze(0)).unsqueeze(0)

    same_src = torch.bmm(src, src.transpose(1, 2)) > thresh
    hit_dst = (torch.bmm(dst, src.transpose(1, 2)) > thresh) | (
        torch.bmm(joi, src.transpose(1, 2)) > thresh
    )
    is_move = (torch.bmm(dst, src.transpose(1, 2)).diagonal(dim1=1, dim2=2) <= thresh)

    reachable = src_ok.unsqueeze(1)
    before = same_src & lower & reachable
    after = hit_dst & upper & reachable

    posf = pos.view(1, 1, n).expand(B, n, n)
    i_idx = torch.where(before, posf, torch.full_like(posf, -1)).amax(dim=2)
    j_idx = torch.where(after, posf, torch.full_like(posf, n)).amin(dim=2)

    keep = (i_idx >= 0) & (j_idx < n) & src_ok & dst_ok & is_move
    nz = torch.nonzero(keep, as_tuple=False)
    if nz.numel() == 0:
        return nz[:, 0], nz[:, 0], nz[:, 0], nz[:, 0]
    rb = nz[:, 0]
    rk = nz[:, 1]
    return rb, i_idx[rb, rk], rk, j_idx[rb, rk]


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not _layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    cmd_mask = batch["cmd_mask"].bool()
    maxn = cmd_mask.shape[1]
    if maxn < 3:
        return 0.0

    tok = batch["tok"]
    cmd = tok[:, 0::2][:, :maxn]
    obs = tok[:, 1::2][:, :maxn]

    rb, ri, rk, rj = _mine(batch["slot_code"].float(), batch["slot_valid"],
                           cmd_mask, float(cfg["match_thresh"]))
    n_tr = int(rb.numel())
    if n_tr == 0:
        return 0.0
    cap = int(cfg["max_examples"])
    if n_tr > cap:
        sel = torch.randperm(n_tr, device=rb.device)[:cap]
        rb, ri, rk, rj = rb[sel], ri[sel], rk[sel], rj[sel]

    pre = torch.nan_to_num(obs[rb, ri].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    cmd_k = torch.nan_to_num(cmd[rb, rk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    post = torch.nan_to_num(obs[rb, rj].detach(), nan=0.0, posinf=1e4, neginf=-1e4)

    pred = torch.nan_to_num(op(pre, cmd_k), nan=0.0, posinf=1e4, neginf=-1e4)
    cos_err = (1.0 - (_unit(pred) * _unit(post)).sum(dim=-1).clamp(-1.0, 1.0)).mean()
    mse_err = (pred - post).pow(2).mean()
    total = float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    return all([
        0.0 < vals["match_thresh"] <= 1.0,
        vals["max_examples"] >= 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ])
