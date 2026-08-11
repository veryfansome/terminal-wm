TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board, measured on the deep windows where analytic shortcuts run out.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

THE CONTRACT — axis 'head': Expose wrap(net, D, **params) -> a head state or None, aux_loss(head_state, batch, net, device) -> a scalar or zero, and leak_safe(mod, params) -> bool. wrap runs BEFORE the optimizer is built, so registered readout and auxiliary parameters are optimized. aux_loss is train-time only. Two hazards a wrapper must avoid: a parent-child module cycle (hold the base net unregistered, or moving to device recurses), and forward recursion when re-pointing forward (save the original bound method first). A head that recomputes the prediction from the trunk hidden state silently bypasses any architecture whose prediction is not a per-position function of that state.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any head impl:
  wrap(net, D, **params) -> a head state, or None for no head
      Called BEFORE the optimizer is built, so any readout or auxiliary parameters it
      registers on net are optimized. Two hazards a wrapper must avoid: a parent-child
      module cycle (hold the base net unregistered, or moving to device recurses), and
      forward recursion when re-pointing net.forward (save the original bound method first).
  aux_loss(head_state, batch, net, device) -> scalar tensor or 0.0; train-time only.
  leak_safe(mod, params) -> bool; asserted before scoring.
"""

import torch

NAME_BASELINE = "baseline_passthrough"
DESCRIPTION_BASELINE = ("Arch's own Linear readout, unchanged; no aux loss. "
                        "Bit-identical to the pre-axis harness readout.")


def wrap(net, D, **params):
    return None


# A hard 0.0 (not a zero tensor) so `main + aux` is main bit-for-bit and archived
# fitnesses replay exactly.
def aux_loss(head_state, batch, net, device):
    return 0.0


def leak_safe(mod, params):
    return True


NAME = NAME_BASELINE
DESCRIPTION = DESCRIPTION_BASELINE
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-06-endpoint-imagination
  its fitness       +0.0112   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r20_endpoint_imagination_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r20_masked_window_imagination_consistency

YOUR PARENT'S CURRENT head IMPL — r20_masked_window_imagination_consistency (this is the code you are mutating):
--------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn

NAME = "r20_masked_window_imagination_consistency"
DESCRIPTION = (
    "The r18 forward-model-consistency aux VERBATIM + a masked-window imagination "
    "aux: mines same-path endpoint (k -> nearest later touch j) pairs in-batch (weighted "
    "by the arch's own detached mutation gate), and trains the co-designed arch's "
    "`imaginer` (cross-attention over raw plan-time prefix pairs queried by the two "
    "endpoint commands) to predict z_obs_j with an eval-geometry L2-InfoNCE + MSE anchor. "
    "Imagination gradients touch only imaginer params; eval forward untouched; base-pion-"
    "equivalent on archs without `imaginer`."
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
    "imag_weight": 1.0,
    "imag_ramp_steps": 400,
    "imag_path_thresh": 0.60,
    "imag_max_examples": 256,
    "imag_tau": 0.25,
    "imag_dup_delta": 0.05,
    "imag_mse": 0.05,
    "imag_wfloor": 0.15,
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
    cfg["_imag_disabled"] = not isinstance(getattr(net, "imaginer", None), nn.Module)
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


@torch.no_grad()
def _mine_endpoint_pairs(cmd, obs, valid, net, path_thresh, wfloor):
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    vmask = valid.bool()

    pos = torch.arange(maxn, device=device)
    upper = pos.unsqueeze(1) < pos.unsqueeze(0)
    same_path = (sim > path_thresh) & vmask.unsqueeze(1)
    after = same_path & upper.unsqueeze(0)
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)
    has_j = (j_idx < maxn) & vmask

    w_mut = None
    try:
        cp = getattr(net, "cmd_proj", None)
        inn = getattr(net, "in_norm", None)
        tmg = getattr(net, "tr_mut_gate", None)
        te = getattr(net, "type_emb", None)
        if (isinstance(cp, nn.Module) and isinstance(inn, nn.Module)
                and isinstance(tmg, nn.Module) and isinstance(te, nn.Module)):
            idx0 = torch.zeros(B, maxn, dtype=torch.long, device=device)
            x_cmd = inn(cp(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)) + te(idx0))
            w_mut = torch.sigmoid(tmg(x_cmd)).squeeze(-1)
    except Exception:
        w_mut = None
    if w_mut is None:
        w_mut = torch.zeros(B, maxn, device=device, dtype=cmd.dtype)

    jc = j_idx.clamp(0, maxn - 1)
    sim_kj = torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1)
    w = sim_kj * (float(wfloor) + w_mut)

    keep = has_j & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)
    sel_b = nz[:, 0]
    sel_k = nz[:, 1]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_k, sel_j, sel_w


def _imag_nce(pred, tgt, w, tau, dup_delta, mse_w):
    n, d = pred.shape
    mse = ((pred - tgt) ** 2).mean(dim=-1)
    if n < 4:
        return (w * mse).sum()
    dist2 = (pred.pow(2).sum(1, keepdim=True) + tgt.pow(2).sum(1) - 2.0 * pred @ tgt.t()) \
        .clamp_min(0.0) / float(d)
    with torch.no_grad():
        tt = (tgt.pow(2).sum(1, keepdim=True) + tgt.pow(2).sum(1) - 2.0 * tgt @ tgt.t()) \
            .clamp_min(0.0) / float(d)
        dup = (tt < float(dup_delta)) & ~torch.eye(n, dtype=torch.bool, device=pred.device)
    logits = (-dist2 / float(tau)).masked_fill(dup, -1e30)
    nll = -torch.log_softmax(logits, dim=1).diagonal()
    return (w * nll).sum() + float(mse_w) * (w * mse).sum()


def _transition_term(cfg, batch, net, device, ramp):
    if cfg.get("_disabled", True) or float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
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
    return float(cfg["aux_weight"]) * ramp * total


def _imagination_term(cfg, batch, net, device, ramp):
    if cfg.get("_imag_disabled", True) or float(cfg.get("imag_weight", 0.0)) <= 0.0:
        return 0.0
    imaginer = getattr(net, "imaginer", None)
    if not isinstance(imaginer, nn.Module):
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 2:
        return 0.0

    cmd = tok[:, 0::2][:, :maxn].detach()
    obs = tok[:, 1::2][:, :maxn].detach()
    valid = cmd_mask

    sb, sk, sj, w = _mine_endpoint_pairs(
        cmd, obs, valid, net, float(cfg["imag_path_thresh"]), float(cfg["imag_wfloor"])
    )
    if sb.numel() < 2:
        return 0.0
    if sb.numel() > int(cfg["imag_max_examples"]):
        w, order = torch.topk(w, int(cfg["imag_max_examples"]))
        sb = sb[order]; sk = sk[order]; sj = sj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pair_cat = torch.cat([cmd, obs], dim=-1)[sb]
    pos = torch.arange(maxn, device=device)
    pmask = valid[sb] & (pos.unsqueeze(0) < sk.unsqueeze(1))
    c_m = cmd[sb, sk]
    c_r = cmd[sb, sj]
    lab = torch.nan_to_num(obs[sb, sj], nan=0.0, posinf=1e4, neginf=-1e4)

    pred = imaginer(pair_cat, pmask, c_m, c_r)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    total = _imag_nce(pred, lab, w, cfg["imag_tau"], cfg["imag_dup_delta"], cfg["imag_mse"])
    return float(cfg["imag_weight"]) * ramp * total


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if cfg.get("_disabled", True) and cfg.get("_imag_disabled", True):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    ramp_imag = _smoothstep(cfg["_step"] / max(1.0, float(cfg["imag_ramp_steps"])))

    total = 0.0
    if ramp > 0.0:
        total = total + _transition_term(cfg, batch, net, device, ramp)
    if ramp_imag > 0.0:
        total = total + _imagination_term(cfg, batch, net, device, ramp_imag)

    if torch.is_tensor(total):
        if not bool(torch.isfinite(total).item()):
            return 0.0
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
        0.0 < vals["row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["change_floor"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["imag_weight"] >= 0.0,
        vals["imag_ramp_steps"] >= 1.0,
        -1.0 <= vals["imag_path_thresh"] < 1.0,
        vals["imag_max_examples"] >= 2.0,
        vals["imag_tau"] > 0.0,
        vals["imag_dup_delta"] >= 0.0,
        vals["imag_mse"] >= 0.0,
        vals["imag_wfloor"] >= 0.0,
    ]
    return all(checks)
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the swapped content [withheld]. Next-obs retrieval health [withheld].
comp_ca -0.0112 over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the sw

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r20_interventional_gain_shell_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r20_interventional_gain_shell_consistency

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r18_transition_forwardmodel_consistency (axis head)
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

--- r20_dualpre_transition_consistency (axis head)
import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CH

NAME = "r20_dualpre_transition_consistency"
DESCRIPTION = (
    "The r18 forward-model-consistency aux VERBATIM (raw-obs pre arm, weight 1.0) plus a "
    "mem-pre arm supervising the SAME shared transition operator on the arch's own memory "
    "content s_pre_k (computed via the arch's input block + _transition_reads under no_grad, "
    "detached) toward the same mined future reads — training the operator on its deployment "
    "distribution (brief findings 6+7) inside the single pass. One mining pass, RNG-draw count "
    "identical to the r18 head; mem arm auto-disables on archs without the memory surface."
)

_DEFAULTS = dict(CH._DEFAULTS)
_DEFAULTS.update({
    "mem_arm_w": 0.5,
})

_MEM_ATTRS = ("cmd_proj", "obs_proj", "type_emb", "in_norm", "_positional", "_transition_reads")


def wrap(net, D, **params):
    cfg = CH.wrap(net, D, **{k: v for k, v in params.items() if k in CH._DEFAULTS})
    cfg["mem_arm_w"] = float(params.get("mem_arm_w", _DEFAULTS["mem_arm_w"]))
    mem_ok = all(hasattr(net, a) for a in _MEM_ATTRS) and hasattr(net, "pos_scale")
    cfg["_mem_disabled"] = bool(cfg.get("_disabled", True)) or not mem_ok
    return cfg


@torch.no_grad()
def _memory_pre(net, rows_tok, rows_types, valid, device):
    L = rows_tok.shape[1]
    t = rows_types.long().clamp(0, 1)
    x = torch.where((t == 0).unsqueeze(-1), net.cmd_proj(rows_tok), net.obs_proj(rows_tok))
    x = x + net.type_emb(t) + net.pos_scale * net._positional(L, device, x.dtype).unsqueeze(0)
    x = net.in_norm(x)
    maxn = valid.shape[1]
    x_cmd = x[:, 0::2][:, :maxn]
    obs = rows_tok[:, 1::2][:, :maxn]
    return net._transition_reads(x_cmd, obs, valid, valid, maxn, maxn)


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not CH._interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = CH._smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
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

    r, ti, tk, tj, w = CH._mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(CH._EPS)).detach()

    cmd_k = torch.nan_to_num(cmd[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(obs[r, tj].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    gu = CH._unit(tgt)

    def arm(pre):
        pred = torch.nan_to_num(op(pre, cmd_k), nan=0.0, posinf=1e4, neginf=-1e4)
        pu = CH._unit(pred)
        cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
        return float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err

    pre_raw = torch.nan_to_num(obs[r, ti].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    total = arm(pre_raw)

    mem_w = float(cfg.get("mem_arm_w", 0.0))
    if mem_w > 0.0 and not cfg.get("_mem_disabled", True):
        reads = _memory_pre(net, tok[sel], batch["types"][sel], valid, device)
        pre_mem = torch.nan_to_num(reads[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
        total = total + mem_w * arm(pre_mem)

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    p = dict(params or {})
    mw = p.pop("mem_arm_w", _DEFAULTS["mem_arm_w"])
    try:
        mw = float(mw)
    except Exception:
        return False
    if not math.isfinite(mw) or mw < 0.0 or mw > 10.0:
        return False
    return CH.leak_safe(mod, p)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot. Invent a genuinely different mechanism, or a novel RECOMBINATION of ideas already in the archive. Do not resubmit an impl you were shown. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before is NOT off-limits. Evolution recombines: a trait that failed ALONE can win in a CHANGED context. You MAY retry a past idea in a new context; argue why the change could flip it.
- LOOK OUTSIDE MACHINE LEARNING — the largest gains in this line of work have come from cross-domain lenses. Read beyond machine learning (neuroscience: predictive coding, hippocampal and episodic memory, place and grid cells; biology; physics; information theory) and translate ONE concrete mechanism into code — equations, not metaphor.
- IGNORE ANY PERFORMANCE FIGURE YOU FIND IN IMPL SOURCE. Some carried mechanisms document measurements taken on a different objective and a different data root, and those numbers mean nothing for what is scored here. Read that source for its MECHANISM — the equations, the interface, what it does and why — and never as a target to match or beat.
- TRACK CONTENT, NOT DISTURBANCE — the scored windows are exactly the ones where knowing THAT something moved is not enough. A mechanism that marks a location as touched, or that keys on the name being asked about, or on which item moved first or last, cancels to zero by construction. Only carrying an item's identity across several hops earns anything.
- NEVER touch the eval, the metric, the split, or the no-leakage guard. The harness re-checks, and a violation makes the candidate unusable regardless of its number.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0112 (g0-06-endpoint-imagination, full budget, inner split).
The unmodified baseline scores +0.0037 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
