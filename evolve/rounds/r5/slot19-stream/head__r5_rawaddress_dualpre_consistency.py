import inspect
import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CH
from evolve.chunks.head import r20_dualpre_transition_consistency as H20

NAME = "r5_rawaddress_dualpre_consistency"
DESCRIPTION = (
    "The r20 dual-pre transition-consistency aux with one change: the mem-pre arm hands the "
    "arch's memory the RAW command tokens as well as the projected command features, whenever "
    "the arch's _transition_reads declares a cmd_raw parameter. An arch whose memory addresses "
    "itself from the raw coded token then produces, inside this no-grad arm, the same memory it "
    "produces at prediction time, so the shared transition operator is supervised on its actual "
    "deployment distribution rather than on a memory addressed some other way. Archs whose "
    "_transition_reads takes no cmd_raw are called exactly as r20 calls them, and the raw-obs arm, "
    "the mining pass, the RNG-draw count and every parameter are r20's."
)

_DEFAULTS = dict(H20._DEFAULTS)


def _accepts_cmd_raw(fn):
    if fn is None or not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "cmd_raw" in sig.parameters


def wrap(net, D, **params):
    cfg = H20.wrap(net, D, **params)
    cfg["_mem_raw_cmd"] = _accepts_cmd_raw(getattr(net, "_transition_reads", None))
    return cfg


@torch.no_grad()
def _memory_pre(net, rows_tok, rows_types, valid, device, raw_cmd):
    L = rows_tok.shape[1]
    t = rows_types.long().clamp(0, 1)
    x = torch.where((t == 0).unsqueeze(-1), net.cmd_proj(rows_tok), net.obs_proj(rows_tok))
    x = x + net.type_emb(t) + net.pos_scale * net._positional(L, device, x.dtype).unsqueeze(0)
    x = net.in_norm(x)
    maxn = valid.shape[1]
    x_cmd = x[:, 0::2][:, :maxn]
    obs = rows_tok[:, 1::2][:, :maxn]
    if raw_cmd:
        cmd_raw = rows_tok[:, 0::2][:, :maxn]
        return net._transition_reads(x_cmd, obs, valid, valid, maxn, maxn, cmd_raw)
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
        reads = _memory_pre(
            net, tok[sel], batch["types"][sel], valid, device, bool(cfg.get("_mem_raw_cmd", False))
        )
        pre_mem = torch.nan_to_num(reads[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
        total = total + mem_w * arm(pre_mem)

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    return H20.leak_safe(mod, params)
