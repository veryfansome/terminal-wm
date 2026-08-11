import math

import torch
import torch.nn.functional as F

from evolve.chunks.head import r22_masked_endpoint_trunk_task as BASE
from evolve.chunks.head import r21_counterfactual_history_guidance as GUIDE

NAME = "r23_calibrated_counterfactual_gap"
DESCRIPTION = (
    "The r22 masked-endpoint trunk task and the r21 counterfactual-history guidance in one head, "
    "plus the term that makes the second one calibrated: on the same in-batch masked-endpoint "
    "layout the trunk is also run with every position before the last gated mutation command "
    "padded away, and that history-masked arm is regressed onto a detached leave-one-out "
    "command-kernel estimate of E[z_obs | read command] taken from the batch. The conditional arm "
    "keeps its r22 supervision toward the true later observation, so the counterfactual gap the "
    "eval-time guidance extrapolates along becomes an estimate of the history-carried content "
    "rather than an untrained difference of two arms. Eval forward for even-length streams is "
    "unchanged; no new parameters."
)

_DEFAULTS = {
    "history_guidance_gain": 1.0,
    "history_guidance_rms": 0.20,
    "cf_weight": 0.25,
    "cf_ramp_steps": 800,
    "cf_period": 2,
    "cf_max_pairs": 48,
    "cf_gate_floor": 0.8,
    "cf_tau": 0.08,
    "cf_tight_pow": 4.0,
}
_EPS = 1e-8


def wrap(net, D, **params):
    cfg = BASE.wrap(net, D, **params)
    private = dict(_DEFAULTS)
    private.update({k: params[k] for k in _DEFAULTS if k in params})
    cfg.update(private)

    original_forward = net.forward
    cfg["_gap_original_forward"] = original_forward

    def wrapped_forward(tok_emb, types, key_pad):
        if tok_emb.size(1) % 2 == 0 or net.training:
            return original_forward(tok_emb, types, key_pad)
        detected = GUIDE._detect_masked_endpoint(types, key_pad)
        if detected is None:
            return original_forward(tok_emb, types, key_pad)
        conditional_pred, conditional_hidden = original_forward(tok_emb, types, key_pad)
        rows, mutation, read = detected
        no_history_pad = GUIDE._counterfactual_pad(key_pad, rows, mutation)
        no_history_pred, _ = original_forward(tok_emb, types, no_history_pad)
        guided = GUIDE._guided(
            conditional_pred[rows, read],
            no_history_pred[rows, read],
            cfg["history_guidance_gain"],
            cfg["history_guidance_rms"],
        )
        out = conditional_pred.clone()
        out[rows, read] = guided
        return out, conditional_hidden

    net.forward = wrapped_forward
    return cfg


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _mine_read_pairs(cfg, batch, net, device):
    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    Dd = tok.shape[2]
    cmd = tok[:, 0::2][:, :maxn]

    with torch.no_grad():
        gate = BASE._gate_values(net, _clean(cmd).reshape(-1, Dd)).reshape(B, maxn)
        gate = torch.nan_to_num(gate, nan=0.0)

    gated = (gate > float(cfg["cf_gate_floor"])) & cmd_mask
    pos = torch.arange(maxn, device=device)
    gpos = torch.where(
        gated,
        pos.unsqueeze(0).expand(B, maxn),
        torch.full((B, maxn), -1, dtype=torch.long, device=device),
    )
    kmax = gpos.cummax(dim=1).values
    k_of = torch.cat(
        [torch.full((B, 1), -1, dtype=torch.long, device=device), kmax[:, :-1]], dim=1
    )
    elig = cmd_mask & (~gated) & (k_of >= 1)
    if not bool(elig.any()):
        return None

    nz = elig.nonzero(as_tuple=False)
    rows, reads = nz[:, 0], nz[:, 1]
    muts = k_of[rows, reads]
    strength = gate[rows, muts]
    cap = int(cfg["cf_max_pairs"])
    if rows.numel() > cap:
        strength, order = torch.topk(strength, cap)
        rows, reads, muts = rows[order], reads[order], muts[order]
    return rows, reads, muts, strength, cmd, cmd_mask


def _content_free_target(cfg, batch, cmd, cmd_mask, rows, reads):
    tgt_full = batch["tgt"]
    device = cmd.device
    flat_cmd = _clean(cmd[cmd_mask].detach())
    flat_tgt = _clean(tgt_full[cmd_mask].detach())
    n_flat = flat_cmd.shape[0]
    if n_flat < 3:
        return None

    flat_index = torch.full(cmd_mask.shape, -1, dtype=torch.long, device=device)
    flat_index[cmd_mask] = torch.arange(n_flat, device=device)
    self_col = flat_index[rows, reads].unsqueeze(1)

    query = _clean(cmd[rows, reads].detach())
    sim = (F.normalize(query, dim=-1) @ F.normalize(flat_cmd, dim=-1).t()).clamp(-1.0, 1.0)
    sim_excl_self = sim.scatter(1, self_col, torch.full_like(self_col, -1, dtype=sim.dtype))
    kernel = torch.softmax(sim_excl_self / float(cfg["cf_tau"]), dim=1)
    kernel = torch.nan_to_num(kernel, nan=0.0)
    kernel = kernel.scatter(1, self_col, torch.zeros_like(self_col, dtype=kernel.dtype))
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(_EPS)

    centre = kernel @ flat_tgt
    tight = (kernel * sim).sum(dim=1).clamp(0.0, 1.0).pow(float(cfg["cf_tight_pow"]))
    return centre, tight


def _cf_term(cfg, batch, net, device, step):
    if cfg.get("_imag_disabled", True):
        return 0.0
    if float(cfg["cf_weight"]) <= 0.0:
        return 0.0
    period = max(1, int(cfg["cf_period"]))
    if (step % period) != 0:
        return 0.0
    ramp = BASE._smoothstep(step / max(1.0, float(cfg["cf_ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    cmd_mask_all = batch["cmd_mask"].bool()
    if cmd_mask_all.shape[1] < 3:
        return 0.0

    mined = _mine_read_pairs(cfg, batch, net, device)
    if mined is None:
        return 0.0
    rows, reads, muts, strength, cmd, cmd_mask = mined

    free = _content_free_target(cfg, batch, cmd, cmd_mask, rows, reads)
    if free is None:
        return 0.0
    centre, tight = free

    L = tok.shape[1]
    Dd = tok.shape[2]
    P = rows.numel()
    Lm = int(2 * int(muts.max().item()) + 3)
    grid = torch.arange(Lm, device=device).unsqueeze(0).expand(P, Lm)
    end_col = (2 * muts + 2).unsqueeze(1)
    mut_col = (2 * muts).unsqueeze(1)
    src = torch.where(grid == end_col, (2 * reads).unsqueeze(1), grid.clamp_max(L - 1))
    keep = (grid <= mut_col) | (grid == end_col)
    tok2 = tok[rows].gather(1, src.unsqueeze(-1).expand(P, Lm, Dd))
    tok2 = _clean(tok2 * keep.unsqueeze(-1).to(tok2.dtype))
    types2 = ((grid % 2 == 1) & (grid <= (2 * muts + 1).unsqueeze(1))).long()
    pad_no_history = (~keep) | (grid < mut_col)

    pred_all, _ = net(tok2, types2, pad_no_history)
    pred = _clean(pred_all[torch.arange(P, device=device), 2 * muts + 2])

    weight = strength.to(pred.dtype) * tight.to(pred.dtype)
    weight = (weight / weight.sum().clamp_min(_EPS)).detach()
    err = (weight * (pred - centre.to(pred.dtype)).pow(2).mean(dim=-1)).sum()
    return float(cfg["cf_weight"]) * ramp * err


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    total = BASE.aux_loss(cfg, batch, net, device)
    if not BASE._interleave_layout_ok(batch):
        return total
    extra = _cf_term(cfg, batch, net, device, int(cfg.get("_step", 0)))
    if torch.is_tensor(extra):
        if bool(torch.isfinite(extra).item()):
            total = total + extra
    elif extra:
        total = total + extra
    if torch.is_tensor(total) and not bool(torch.isfinite(total).item()):
        return 0.0
    return total


def leak_safe(mod, params):
    if not BASE.leak_safe(mod, params):
        return False
    p = dict(_DEFAULTS)
    p.update({k: (params or {})[k] for k in _DEFAULTS if k in (params or {})})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    return all(
        [
            0.0 <= vals["history_guidance_gain"] <= 4.0,
            0.0 < vals["history_guidance_rms"] <= 2.0,
            vals["cf_weight"] >= 0.0,
            vals["cf_ramp_steps"] >= 1.0,
            vals["cf_period"] >= 1.0,
            vals["cf_max_pairs"] >= 1.0,
            0.0 < vals["cf_gate_floor"] < 1.0,
            vals["cf_tau"] > 0.0,
            vals["cf_tight_pow"] >= 0.0,
        ]
    )
