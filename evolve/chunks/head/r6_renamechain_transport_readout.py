import math

import torch

NAME = "r6_renamechain_transport_readout"
DESCRIPTION = (
    "Train-only supervision of the r18/r19 latent-transition MEMORY READ at verified content-"
    "transport reads, instead of the shared operator on adjacent same-path pairs. A transparent "
    "hook on the arch's _transition_reads keeps the reads tensor produced by the step's own "
    "forward (no recomputation, gradient intact). Mining is label-free from the batch: steps whose "
    "observation duplicates many others are SILENT mutators (empty-output mv / redirect), the rest "
    "are content steps; a pair (i, j) with i<j, duplicate observations, non-duplicate commands and "
    "at least one silent step in between is a content that was re-addressed by a silent chain and "
    "read back at j. At every such j the memory read s_pre is trained by a squared-L2 InfoNCE, in "
    "the eval's own decision variable, to pick the transported content out of the row's earlier "
    "content observations, plus a cosine alignment term; examples are weighted by the number of "
    "silent steps crossed, so deep chains dominate the gradient. Blank move observations are never "
    "a target. Introduces no parameters: every gradient lands on the arch's addressing projection, "
    "mutation gate, transition MLP and decay, all of which shape the scored read. Disabled (0.0) on "
    "archs without the transition memory."
)

_DEFAULTS = {
    "dup_frac": 0.05,
    "cmd_dup_frac": 0.01,
    "max_dup": 2,
    "min_silent": 1,
    "depth_gain": 0.6,
    "depth_cap": 6,
    "max_examples": 256,
    "nce_temp": 0.5,
    "nce_weight": 0.3,
    "cos_weight": 0.2,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _layout_ok(b):
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
    return bool(((even == 0) | (even == 2)).all().item()) and bool((odd == 1).all().item())


def wrap(net, D, **params):
    existing = getattr(net, "_renamechain_transport_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)
    state = {"cfg": cfg, "step": 0, "reads": None, "D": int(D), "disabled": True}

    original_reads = getattr(net, "_transition_reads", None)
    if callable(original_reads):
        state["disabled"] = False

        def _capturing_transition_reads(*args, **kwargs):
            out = original_reads(*args, **kwargs)
            if (torch.is_grad_enabled() and net.training
                    and torch.is_tensor(out) and out.requires_grad):
                state["reads"] = out
            else:
                state["reads"] = None
            return out

        net._transition_reads = _capturing_transition_reads

    net._renamechain_transport_state = state
    return state


@torch.no_grad()
def _mine(tok, valid, cfg):
    B, L, Dm = tok.shape
    n = valid.shape[1]
    dev = tok.device
    o = torch.nan_to_num(tok[:, 1::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    c = torch.nan_to_num(tok[:, 0::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    dm = float(Dm)

    osq = o.pow(2).sum(-1)
    csq = c.pow(2).sum(-1)
    d2o = (osq.unsqueeze(2) + osq.unsqueeze(1) - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / dm
    d2c = (csq.unsqueeze(2) + csq.unsqueeze(1) - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / dm

    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    vv = valid.unsqueeze(2) & valid.unsqueeze(1)
    off = vv & (~eye)
    cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
    ref_o = ((d2o * off).sum(dim=(1, 2)) / cnt).clamp_min(1e-8)
    ref_c = ((d2c * off).sum(dim=(1, 2)) / cnt).clamp_min(1e-8)

    dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
    dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

    clus = dup_o.sum(dim=2)
    silent = valid & (clus > int(cfg["max_dup"]))
    content = valid & (~silent)

    cs = torch.cumsum(silent.long(), dim=1)
    crossed = cs.unsqueeze(2) - cs.unsqueeze(1)

    idxn = torch.arange(n, device=dev)
    later = idxn.view(1, n, 1) > idxn.view(1, 1, n)
    keep = (dup_o & (~dup_c) & later
            & content.unsqueeze(2) & content.unsqueeze(1)
            & (crossed >= int(cfg["min_silent"])))

    exists = keep.any(dim=2)
    depth = torch.where(keep, crossed, torch.full_like(crossed, -1)).amax(dim=2)
    return o, osq, dup_o, content, exists, depth


def aux_loss(head_state, batch, net, device):
    st = head_state
    if st is None:
        return 0.0
    reads = st.get("reads")
    st["reads"] = None
    if reads is None or st.get("disabled", True):
        return 0.0
    cfg = st["cfg"]
    if float(cfg["aux_weight"]) <= 0.0:
        return 0.0
    if not _layout_ok(batch):
        return 0.0

    tok = batch["tok"]
    B, n = batch["cmd_mask"].shape
    if n < 3:
        return 0.0
    if reads.dim() != 3 or reads.size(0) != B or reads.size(1) < n or reads.size(2) != tok.size(2):
        return 0.0
    if not reads.requires_grad:
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    valid = batch["cmd_mask"].bool() & (batch["types"][:, 0::2][:, :n] != 2)
    if not bool(valid.any().item()):
        return 0.0

    o, osq, dup_o, content, exists, depth = _mine(tok, valid, cfg)

    with torch.no_grad():
        sel = torch.nonzero(exists, as_tuple=False)
        if sel.numel() == 0:
            return 0.0
        b_idx = sel[:, 0]
        j_idx = sel[:, 1]
        dep = depth[b_idx, j_idx].clamp(min=1).float()
        w = 1.0 + float(cfg["depth_gain"]) * (dep - 1.0).clamp(max=float(cfg["depth_cap"]))
        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            jitter = torch.rand_like(w) * 1e-3
            _, order = torch.topk(w + jitter, cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]
            w = w[order]

        idxn = torch.arange(n, device=tok.device)
        seen = idxn.view(1, n) < j_idx.view(-1, 1)
        cand_mask = content[b_idx] & seen
        pos_mask = dup_o[b_idx, j_idx] & cand_mask
        rows_ok = pos_mask.any(dim=1) & (cand_mask.sum(dim=1) >= 2)
        if not bool(rows_ok.any().item()):
            return 0.0
        b_idx = b_idx[rows_ok]
        j_idx = j_idx[rows_ok]
        w = w[rows_ok]
        cand_mask = cand_mask[rows_ok]
        pos_mask = pos_mask[rows_ok]
        wn = (w / w.sum().clamp_min(_EPS)).to(reads.dtype)

    cand = o[b_idx]
    cand_sq = osq[b_idx]
    tgt = o[b_idx, j_idx]
    r = reads[:, :n, :][b_idx, j_idx].float()
    dm = float(tok.size(2))

    d2 = (r.pow(2).sum(-1, keepdim=True) + cand_sq
          - 2.0 * torch.bmm(cand, r.unsqueeze(2)).squeeze(2)).clamp_min(0.0) / dm
    logits = (-d2 / max(1e-3, float(cfg["nce_temp"]))).masked_fill(~cand_mask, _NEG)
    logp = torch.log_softmax(logits, dim=1)
    nce = -torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)

    rn = r * torch.rsqrt(r.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))
    tn = tgt * torch.rsqrt(tgt.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))
    cos_err = 1.0 - (rn * tn).sum(dim=-1).clamp(-1.0, 1.0)

    total = (float(cfg["nce_weight"]) * (wn * nce).sum()
             + float(cfg["cos_weight"]) * (wn * cos_err).sum())

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss.to(tok.dtype)


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
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["min_silent"] >= 1.0,
        vals["depth_gain"] >= 0.0,
        vals["depth_cap"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        vals["nce_weight"] >= 0.0,
        vals["cos_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
