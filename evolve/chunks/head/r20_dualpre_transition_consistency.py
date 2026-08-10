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
