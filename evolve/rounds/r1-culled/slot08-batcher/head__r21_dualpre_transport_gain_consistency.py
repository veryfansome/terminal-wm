import math

import torch
import torch.nn.functional as F

from evolve.chunks.head import r20_dualpre_transition_consistency as DP
from evolve.chunks.head import r20_interventional_gain_shell_consistency as GS

NAME = "r21_dualpre_transport_gain_consistency"
DESCRIPTION = (
    "Union of the two parent heads on one shared transition operator: the dual-pre arm "
    "(raw-obs pre AND memory-content pre, so the operator is trained on the composed state it "
    "actually reads at depth) plus the interventional gain/shell calibrator arm, whose pair "
    "miner is replaced by a TRANSPORT miner that requires an earlier same-path read inside the "
    "prefix and a materially changed observation at the later read, so the calibrator is trained "
    "only on windows where content actually moved across the silent hop. Falls back to the "
    "parent's ungated mining whenever the transport gate starves a step."
)

_DEFAULTS = dict(GS._DEFAULTS)
_DEFAULTS.update(DP._DEFAULTS)
_DEFAULTS.update({
    "gain_change_floor": 0.25,
})


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    cfg["D"] = int(D)
    cfg["_gain_step"] = 0
    cfg["_base"] = DP.wrap(
        net, D, **{k: v for k, v in (params or {}).items() if k in DP._DEFAULTS}
    )
    cfg["_disabled"] = not (
        bool(getattr(net, "supports_interventional_calibrator", False))
        and callable(getattr(net, "imagination_command_features", None))
        and callable(getattr(net, "imagination_calibrate", None))
        and hasattr(net, "_imag_mode")
        and hasattr(net, "tr_mut_gate")
    )
    return cfg


@torch.no_grad()
def _mine_transport_pairs(cmd, obs, cmd_feat, valid, net, threshold, change_floor, max_pairs):
    batch, maxn, _ = cmd.shape
    device = cmd.device
    empty = torch.zeros(0, dtype=torch.long, device=device)

    cu = GS._unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    pos = torch.arange(maxn, device=device)
    later = pos.unsqueeze(1) < pos.unsqueeze(0)
    earlier = pos.unsqueeze(1) > pos.unsqueeze(0)
    same = (sim > threshold) & valid.unsqueeze(1)
    pos_full = pos.view(1, 1, maxn).expand(batch, maxn, maxn)

    j_idx = torch.where(
        same & later.unsqueeze(0), pos_full, torch.full_like(pos_full, maxn)
    ).amin(dim=2)
    i_idx = torch.where(
        same & earlier.unsqueeze(0), pos_full, torch.full_like(pos_full, -1)
    ).amax(dim=2)

    ok = (j_idx < maxn) & (i_idx >= 0) & valid
    if not bool(ok.any().item()):
        return empty, empty, empty

    ic = i_idx.clamp(0, maxn - 1)
    jc = j_idx.clamp(0, maxn - 1)
    width = obs.size(-1)
    obs_i = obs.gather(1, ic.unsqueeze(-1).expand(batch, maxn, width))
    obs_j = obs.gather(1, jc.unsqueeze(-1).expand(batch, maxn, width))
    change = (obs_i - obs_j).pow(2).mean(dim=-1)

    ok = ok & (change >= change_floor) & torch.isfinite(change)
    if not bool(ok.any().item()):
        return empty, empty, empty

    sim_ki = sim.gather(2, ic.unsqueeze(-1)).squeeze(-1)
    sim_kj = sim.gather(2, jc.unsqueeze(-1)).squeeze(-1)
    mutation = torch.sigmoid(net.tr_mut_gate(cmd_feat)).squeeze(-1)
    weight = sim_ki * sim_kj * (0.15 + mutation) * change
    weight = torch.where(ok, weight, torch.full_like(weight, -1.0))

    count = min(int(max_pairs), int(ok.sum().item()))
    top = torch.topk(weight.flatten(), count).indices
    sel_b = top // maxn
    sel_k = top % maxn
    return sel_b, sel_k, j_idx[sel_b, sel_k]


def _gain_loss(cfg, batch, net):
    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    maxn = valid.size(1)
    if maxn < 3:
        return None

    cmd = tok[:, 0::2, :][:, :maxn]
    obs = tok[:, 1::2, :][:, :maxn]
    with torch.no_grad():
        cmd_feat = net.imagination_command_features(tok, batch["types"])[:, :maxn, :]
        sel_b, sel_k, sel_j = _mine_transport_pairs(
            cmd,
            obs,
            cmd_feat,
            valid,
            net,
            float(cfg["gain_path_thresh"]),
            float(cfg["gain_change_floor"]),
            int(cfg["gain_max_pairs"]),
        )
        if sel_b.numel() < 4:
            sel_b, sel_k, sel_j = GS._mine_pairs(
                cmd,
                cmd_feat,
                valid,
                net,
                float(cfg["gain_path_thresh"]),
                int(cfg["gain_max_pairs"]),
            )
    if sel_b.numel() < 4:
        return None

    masked_tok, masked_types, masked_pad, read_pos = GS._build_masked(
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
    dist = GS._d2m(out, target)
    with torch.no_grad():
        target_dist = GS._d2m(target, target)
        duplicate = target_dist < float(cfg["gain_dup_delta"])
        duplicate = duplicate & ~torch.eye(count, dtype=torch.bool, device=out.device)
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

    base_term = DP.aux_loss(cfg.get("_base"), batch, net, device)
    if cfg.get("_disabled", True) or float(cfg["gain_weight"]) <= 0.0:
        return base_term
    if not DP.CH._interleave_layout_ok(batch):
        return base_term

    cfg["_gain_step"] = int(cfg.get("_gain_step", 0)) + 1
    step = cfg["_gain_step"]
    every = max(1, int(cfg["gain_every"]))
    if step % every != 0:
        return base_term

    span = max(1.0, float(cfg["gain_ramp_full"]) - float(cfg["gain_ramp_start"]))
    ramp = GS._smoothstep((step - float(cfg["gain_ramp_start"])) / span)
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
        vals = {k: float(p[k]) for k in GS._DEFAULTS}
        vals["gain_change_floor"] = float(p["gain_change_floor"])
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        vals["gain_weight"] >= 0.0,
        vals["gain_ramp_start"] >= 0.0,
        vals["gain_ramp_full"] > vals["gain_ramp_start"],
        vals["gain_every"] >= 1.0 and vals["gain_every"].is_integer(),
        -1.0 <= vals["gain_path_thresh"] < 1.0,
        vals["gain_max_pairs"] >= 4.0 and vals["gain_max_pairs"].is_integer(),
        vals["gain_temp"] > 0.0,
        vals["gain_dup_delta"] >= 0.0,
        vals["gain_mse"] >= 0.0,
        vals["gain_prior"] >= 0.0,
        vals["gain_shell_pen"] >= 0.0,
        vals["gain_change_floor"] >= 0.0,
    ]
    if not all(checks):
        return False
    inherited = {k: v for k, v in (params or {}).items() if k in DP._DEFAULTS}
    return DP.leak_safe(mod, inherited)
