import math

import torch
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as BASE

NAME = "r22_silenthop_rollout_transport"
DESCRIPTION = (
    "The r18 forward-model consistency term plus a GRADIENT-CARRYING silent-hop rollout: real "
    "training rows are re-laid-out as prefix pairs, then H consecutive observation-free command "
    "hops, then a read command, and the base net's own prediction at that read command is trained "
    "against the true observation with a duplicate-masked squared-L2 InfoNCE (the eval's decision "
    "variable) plus MSE. Unlike the partner head, the masked forward is NOT detached and there is "
    "no bolt-on observer: the gradient goes into the memory write/read addresses themselves, which "
    "is the only path by which the observation-free branch of the arch is ever trained."
)

_DEFAULTS = {
    "roll_hop_max": 3,
    "roll_read_cap": 8,
    "roll_rows": 24,
    "roll_path_thresh": 0.55,
    "roll_tau": 0.25,
    "roll_dupe_cos": 0.98,
    "roll_mse_weight": 0.20,
    "roll_weight": 0.30,
    "roll_ramp_start": 200,
    "roll_ramp_steps": 800,
    "roll_every": 1,
}

_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def wrap(net, D, **params):
    cfg = BASE.wrap(net, D, **params)
    mine = dict(_DEFAULTS)
    mine.update(params)
    cfg.update(mine)
    cfg["_roll_step"] = 0
    cfg["_roll_disabled"] = not callable(getattr(net, "_transition_reads", None))
    return cfg


def _select_windows(cfg, cmd, valid, device):
    B, maxn, _ = cmd.shape
    n = valid.sum(1)
    usable = torch.nonzero(n >= 3, as_tuple=False).squeeze(1)
    if usable.numel() < 2:
        return None
    cap = max(2, int(cfg["roll_rows"]))
    if usable.numel() > cap:
        usable = usable[torch.randperm(usable.numel(), device=device)[:cap]]

    read_cap = max(2, int(cfg["roll_read_cap"]))
    hop_max = max(1, int(cfg["roll_hop_max"]))
    r = torch.clamp(n[usable] - 1, max=read_cap)
    hops = torch.clamp(torch.clamp(r - 1, max=hop_max), min=1)
    m = r - hops

    cu = _unit(torch.nan_to_num(cmd[usable], nan=0.0, posinf=1e4, neginf=-1e4))
    idx = torch.arange(usable.numel(), device=device)
    sim = torch.einsum("nd,nmd->nm", cu[idx, r], cu)
    pos = torch.arange(maxn, device=device).unsqueeze(0)
    window = (pos >= m.unsqueeze(1)) & (pos < r.unsqueeze(1))
    best = torch.where(window, sim, torch.full_like(sim, -2.0)).amax(1)
    keep = best > float(cfg["roll_path_thresh"])
    if int(keep.sum().item()) < 2:
        return None
    return usable[keep], r[keep], m[keep]


def _masked_stream(cmd, obs, rows, r, m, device):
    n_sel = rows.numel()
    maxn = cmd.size(1)
    rmax = int(r.max().item())
    L = 2 * rmax + 1
    pair = torch.arange(rmax + 1, device=device)
    src = pair.clamp(max=maxn - 1)

    c_tok = cmd[rows][:, src]
    o_tok = obs[rows][:, src[:rmax]]
    live_c = pair.unsqueeze(0) <= r.unsqueeze(1)
    live_o = pair[:rmax].unsqueeze(0) < m.unsqueeze(1)

    tok = c_tok.new_zeros(n_sel, L, c_tok.size(-1))
    tok[:, 0::2] = c_tok * live_c.unsqueeze(-1).to(c_tok.dtype)
    if rmax > 0:
        tok[:, 1::2] = o_tok * live_o.unsqueeze(-1).to(o_tok.dtype)
    types = torch.zeros(n_sel, L, dtype=torch.long, device=device)
    types[:, 1::2] = 1
    key_pad = torch.ones(n_sel, L, dtype=torch.bool, device=device)
    key_pad[:, 0::2] = ~live_c
    if rmax > 0:
        key_pad[:, 1::2] = ~live_o
    return torch.nan_to_num(tok, nan=0.0, posinf=1e4, neginf=-1e4).detach(), types, key_pad


def _rollout_loss(cfg, batch, net, device):
    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    B, maxn = valid.shape
    if B < 2 or maxn < 3:
        return None
    cmd = tok[:, 0::2, :][:, :maxn].detach()
    obs = tok[:, 1::2, :][:, :maxn].detach()

    with torch.no_grad():
        picked = _select_windows(cfg, cmd, valid, device)
    if picked is None:
        return None
    rows, r, m = picked

    mtok, mtypes, mpad = _masked_stream(cmd, obs, rows, r, m, device)
    pred, _ = net(mtok, mtypes, mpad)
    idx = torch.arange(rows.numel(), device=device)
    point = torch.nan_to_num(pred[:, 0::2][idx, r], nan=0.0, posinf=1e4, neginf=-1e4)
    target = torch.nan_to_num(obs[rows, r], nan=0.0, posinf=1e4, neginf=-1e4)

    dist = (point.unsqueeze(1) - target.unsqueeze(0)).pow(2).mean(-1)
    logits = -dist / max(1e-4, float(cfg["roll_tau"]))
    with torch.no_grad():
        tu = _unit(target)
        eye = torch.eye(target.size(0), device=device, dtype=torch.bool)
        dupe = (tu @ tu.T > float(cfg["roll_dupe_cos"])) & ~eye
    logits = logits.masked_fill(dupe, -1e4)
    rank = F.cross_entropy(logits, idx)
    mse = (point - target).pow(2).mean()
    total = rank + float(cfg["roll_mse_weight"]) * mse
    if not bool(torch.isfinite(total).item()):
        return None
    return total


def aux_loss(head_state, batch, net, device):
    base = BASE.aux_loss(head_state, batch, net, device)
    cfg = head_state
    if cfg is None or cfg.get("_roll_disabled", True):
        return base
    if float(cfg.get("roll_weight", 0.0)) <= 0.0:
        return base
    if not BASE._interleave_layout_ok(batch):
        return base
    cfg["_roll_step"] = int(cfg.get("_roll_step", 0)) + 1
    step = cfg["_roll_step"]
    if step % max(1, int(cfg["roll_every"])) != 0:
        return base
    ramp = _smoothstep(
        (step - float(cfg["roll_ramp_start"])) / max(1.0, float(cfg["roll_ramp_steps"]))
    )
    if ramp <= 0.0:
        return base
    term = _rollout_loss(cfg, batch, net, device)
    if term is None:
        return base
    return base + float(cfg["roll_weight"]) * ramp * term


def leak_safe(mod, params):
    if not BASE.leak_safe(mod, params):
        return False
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        v = {key: float(p[key]) for key in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(x) for x in v.values()):
        return False
    return all([
        v["roll_hop_max"] >= 1,
        v["roll_read_cap"] >= 2,
        v["roll_rows"] >= 2,
        -1.0 <= v["roll_path_thresh"] < 1.0,
        v["roll_tau"] > 0.0,
        -1.0 <= v["roll_dupe_cos"] <= 1.0,
        v["roll_mse_weight"] >= 0.0,
        v["roll_weight"] >= 0.0,
        v["roll_ramp_start"] >= 0.0,
        v["roll_ramp_steps"] >= 1.0,
        v["roll_every"] >= 1,
    ])
