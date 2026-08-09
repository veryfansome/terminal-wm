"""R20 head: train-only calibration of the interventional gain/shell arch.

The R18 forward-model consistency auxiliary is retained. A second, sparse
auxiliary mines same-path mutation/read candidates, evaluates the structured no-write and
full-write hypotheses without trunk gradients, and trains only the arch's two-scalar
calibrator in the same training pass.
"""

import math

import torch
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as BASE

NAME = "r20_interventional_gain_shell_consistency"
DESCRIPTION = (
    "The R18 forward-model consistency plus a sparse train-only auxiliary for the "
    "co-designed interventional gain/shell arch. It mines same-path endpoints, constructs "
    "[prefix,c_m,PAD,c_r], obtains detached no-write/full-write hypotheses, and trains only "
    "the 12.5K two-scalar calibrator with eval-geometry InfoNCE, MSE, prior anchoring and a "
    "small radial-use penalty. Forward is never re-pointed."
)

_EPS = 1e-8
_DEFAULTS = {
    "gain_weight": 0.10,
    "gain_ramp_start": 400.0,
    "gain_ramp_full": 1200.0,
    "gain_every": 4.0,
    "gain_path_thresh": 0.60,
    "gain_max_pairs": 64.0,
    "gain_temp": 0.25,
    "gain_dup_delta": 0.05,
    "gain_mse": 0.05,
    "gain_prior": 0.02,
    "gain_shell_pen": 0.005,
}


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _d2m(a, b):
    d2 = (
        a.pow(2).sum(1, keepdim=True)
        + b.pow(2).sum(1, keepdim=True).transpose(0, 1)
        - 2.0 * a @ b.transpose(0, 1)
    )
    return d2.clamp_min(0.0) / a.shape[1]


@torch.no_grad()
def _mine_pairs(cmd, cmd_feat, valid, net, threshold, max_pairs):
    """Nearest later same-path touch for each command, prioritized by mutation confidence."""
    batch, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    pos = torch.arange(maxn, device=device)
    later = pos.unsqueeze(1) < pos.unsqueeze(0)
    same = (sim > threshold) & valid.unsqueeze(1)
    candidates = same & later.unsqueeze(0)
    pos_full = pos.view(1, 1, maxn).expand(batch, maxn, maxn)
    j_idx = torch.where(
        candidates, pos_full, torch.full_like(pos_full, maxn)
    ).amin(dim=2)
    ok = (j_idx < maxn) & valid
    if not bool(ok.any().item()):
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, empty

    j_clamped = j_idx.clamp(0, maxn - 1)
    sim_kj = sim.gather(2, j_clamped.unsqueeze(-1)).squeeze(-1)
    mutation = torch.sigmoid(net.tr_mut_gate(cmd_feat)).squeeze(-1)
    weight = sim_kj * (0.15 + mutation)
    weight = torch.where(ok, weight, torch.full_like(weight, -1.0))
    count = min(int(max_pairs), int(ok.sum().item()))
    top = torch.topk(weight.flatten(), count).indices
    sel_b = top // maxn
    sel_k = top % maxn
    return sel_b, sel_k, j_idx[sel_b, sel_k]


def _build_masked(tok, sel_b, sel_k, sel_j):
    """Build [fully observed pairs before k, c_k, PAD, c_j]."""
    device = tok.device
    count = sel_b.numel()
    length = 2 * int(sel_k.max().item()) + 3
    pos = torch.arange(length, device=device).view(1, length).expand(count, length)
    k2 = (2 * sel_k).view(count, 1)
    j2 = (2 * sel_j).view(count, 1)

    src = torch.where(pos < k2, pos, torch.zeros_like(pos))
    src = torch.where(pos == k2, k2.expand_as(pos), src)
    src = torch.where(pos == k2 + 2, j2.expand_as(pos), src)
    keep = (pos < k2) | (pos == k2) | (pos == k2 + 2)
    src = src.clamp(0, tok.size(1) - 1)

    masked_tok = tok[sel_b].gather(
        1, src.unsqueeze(-1).expand(count, length, tok.size(-1))
    )
    masked_tok = masked_tok * keep.unsqueeze(-1)
    masked_types = (pos % 2).long().contiguous()
    masked_pad = ~keep
    read_pos = 2 * sel_k + 2
    return masked_tok, masked_types, masked_pad, read_pos.long()


def wrap(net, D, **params):
    base_params = {k: params[k] for k in BASE._DEFAULTS if k in params}
    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    cfg["D"] = int(D)
    cfg["_gain_step"] = 0
    cfg["_base"] = BASE.wrap(net, D, **base_params)
    cfg["_disabled"] = not (
        bool(getattr(net, "supports_interventional_calibrator", False))
        and callable(getattr(net, "imagination_command_features", None))
        and callable(getattr(net, "imagination_calibrate", None))
        and hasattr(net, "_imag_mode")
        and hasattr(net, "tr_mut_gate")
    )
    return cfg


def _gain_loss(cfg, batch, net):
    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    maxn = valid.size(1)
    if maxn < 2:
        return None

    cmd = tok[:, 0::2, :][:, :maxn]
    obs = tok[:, 1::2, :][:, :maxn]
    with torch.no_grad():
        cmd_feat = net.imagination_command_features(
            tok, batch["types"]
        )[:, :maxn, :]
        sel_b, sel_k, sel_j = _mine_pairs(
            cmd,
            cmd_feat,
            valid,
            net,
            float(cfg["gain_path_thresh"]),
            int(cfg["gain_max_pairs"]),
        )
    if sel_b.numel() < 4:
        return None

    masked_tok, masked_types, masked_pad, read_pos = _build_masked(
        tok, sel_b, sel_k, sel_j
    )
    was_training = net.training
    old_mode = net._imag_mode
    net.eval()
    try:
        with torch.no_grad():
            net._imag_mode = "off"
            pred0, _ = net(masked_tok, masked_types, masked_pad)
            net._imag_mode = "full"
            pred1, _ = net(masked_tok, masked_types, masked_pad)
    finally:
        net._imag_mode = old_mode
        net.train(was_training)

    arange = torch.arange(sel_b.numel(), device=tok.device)
    p0 = pred0[arange, read_pos].detach()
    p1 = pred1[arange, read_pos].detach()
    target = obs[sel_b, sel_j].detach()

    with torch.no_grad():
        masked_feat = net.imagination_command_features(masked_tok, masked_types)
        x_m = masked_feat[arange, sel_k].detach()
        x_r = masked_feat[arange, sel_k + 1].detach()
        prior = torch.sigmoid(net.tr_mut_gate(x_m)).squeeze(-1).detach()

    out, alpha, beta = net.imagination_calibrate(p0, p1, x_m, x_r, prior)
    count = out.size(0)
    dist = _d2m(out, target)
    with torch.no_grad():
        target_dist = _d2m(target, target)
        duplicate = target_dist < float(cfg["gain_dup_delta"])
        duplicate = duplicate & ~torch.eye(
            count, dtype=torch.bool, device=out.device
        )
    logits = (-dist / float(cfg["gain_temp"])).masked_fill(duplicate, -1e9)
    nce = F.cross_entropy(logits, torch.arange(count, device=out.device))
    mse = (out - target).pow(2).mean()
    prior_loss = (alpha - prior).pow(2).mean()
    shell_use = beta.mean()
    return (
        nce
        + float(cfg["gain_mse"]) * mse
        + float(cfg["gain_prior"]) * prior_loss
        + float(cfg["gain_shell_pen"]) * shell_use
    )


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0

    base_term = BASE.aux_loss(cfg.get("_base"), batch, net, device)
    if cfg.get("_disabled", True) or float(cfg["gain_weight"]) <= 0.0:
        return base_term
    if not BASE._interleave_layout_ok(batch):
        return base_term

    cfg["_gain_step"] = int(cfg.get("_gain_step", 0)) + 1
    step = cfg["_gain_step"]
    every = max(1, int(cfg["gain_every"]))
    if step % every != 0:
        return base_term

    span = max(
        1.0, float(cfg["gain_ramp_full"]) - float(cfg["gain_ramp_start"])
    )
    ramp = _smoothstep((step - float(cfg["gain_ramp_start"])) / span)
    if ramp <= 0.0:
        return base_term

    gain = _gain_loss(cfg, batch, net)
    if gain is None or not bool(torch.isfinite(gain).item()):
        return base_term
    return base_term + float(cfg["gain_weight"]) * ramp * gain


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        values = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in values.values()):
        return False
    checks = [
        values["gain_weight"] >= 0.0,
        values["gain_ramp_start"] >= 0.0,
        values["gain_ramp_full"] > values["gain_ramp_start"],
        values["gain_every"] >= 1.0 and values["gain_every"].is_integer(),
        -1.0 <= values["gain_path_thresh"] < 1.0,
        values["gain_max_pairs"] >= 4.0 and values["gain_max_pairs"].is_integer(),
        values["gain_temp"] > 0.0,
        values["gain_dup_delta"] >= 0.0,
        values["gain_mse"] >= 0.0,
        values["gain_prior"] >= 0.0,
        values["gain_shell_pen"] >= 0.0,
    ]
    return all(checks) and BASE.leak_safe(mod, params or {})
