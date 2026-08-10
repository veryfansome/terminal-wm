import math

import torch
import torch.nn as nn

NAME = "r18_transition_forwardmodel_consistency"
DESCRIPTION = (
    "Train-only forward-model consistency on the r18 latent-transition arch's shared operator: "
    "mines same-path (pre, mutating-cmd, future-read) triples and requires f(obs_pre, cmd) to "
    "reconstruct the future post-mutation observation (cosine+MSE, change-weighted). Eval forward "
    "untouched; disabled (0.0) on archs without the transition operator. Co-designed head half of "
    "the r18 transition world-model stack."
)

_DEFAULTS = {
    "row_frac": 0.6,
    "path_thresh": 0.60,
    "change_floor": 0.25,
    "max_examples": 512,
    "cos_weight": 0.10,
    "mse_weight": 0.02,
    "aux_weight": 1.0,
    "ramp_steps": 400,
}

_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _interleave_layout_ok(b):
    for k in ("tok", "types", "key_pad", "tgt", "cmd_mask"):
        if k not in b:
            return False
    tok, types, key_pad, tgt, cmd_mask = b["tok"], b["types"], b["key_pad"], b["tgt"], b["cmd_mask"]
    if tok.dim() != 3 or tgt.dim() != 3 or types.dim() != 2 or key_pad.dim() != 2:
        return False
    if tok.shape[:2] != types.shape or tok.shape[:2] != key_pad.shape:
        return False
    if cmd_mask.shape != tgt.shape[:2] or tok.shape[0] != tgt.shape[0] or tok.shape[2] != tgt.shape[2]:
        return False
    if tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~key_pad.bool()
    if not bool(live.any().item()):
        return False
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    return cfg


@torch.no_grad()
def _mine_triples(cmd, obs, valid, path_thresh, change_floor):
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    vmask = valid.bool()

    pos = torch.arange(maxn, device=device)
    lower = pos.unsqueeze(1) > pos.unsqueeze(0)
    upper = pos.unsqueeze(1) < pos.unsqueeze(0)
    same_path = (sim > path_thresh) & vmask.unsqueeze(1)

    before = same_path & lower.unsqueeze(0)
    after = same_path & upper.unsqueeze(0)
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    i_idx = torch.where(before, posf, torch.full_like(posf, -1)).amax(dim=2)
    j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)

    has_i = i_idx >= 0
    has_j = j_idx < maxn
    triple_ok = has_i & has_j & vmask

    ic = i_idx.clamp(0, maxn - 1)
    jc = j_idx.clamp(0, maxn - 1)
    obs_i = torch.gather(obs, 1, ic.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    obs_j = torch.gather(obs, 1, jc.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    change = (obs_i - obs_j).pow(2).mean(dim=-1)
    sim_ki = torch.gather(sim, 2, ic.unsqueeze(-1)).squeeze(-1)
    sim_kj = torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1)
    w = sim_ki * sim_kj * change

    keep = triple_ok & (change >= change_floor) & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)
    sel_b = nz[:, 0]
    sel_k = nz[:, 1]
    sel_i = i_idx[sel_b, sel_k]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_i, sel_k, sel_j, sel_w


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = _mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pre = obs[r, ti].detach()
    cmd_k = cmd[r, tk].detach()
    tgt = obs[r, tj].detach()
    pre = torch.nan_to_num(pre, nan=0.0, posinf=1e4, neginf=-1e4)
    cmd_k = torch.nan_to_num(cmd_k, nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)

    pred = op(pre, cmd_k)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
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
    checks = [
        0.0 < vals["row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["change_floor"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
