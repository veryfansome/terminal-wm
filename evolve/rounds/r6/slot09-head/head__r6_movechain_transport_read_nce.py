import math

import torch
import torch.nn.functional as F

NAME = "r6_movechain_transport_read_nce"
DESCRIPTION = (
    "Keeps the r18 forward-model-consistency arm verbatim and adds a second arm that supervises "
    "the ARCH'S OWN dual-key transition memory in place: the per-command read s_pre produced "
    "inside _transition_reads is captured with gradient by re-pointing that bound method, and at "
    "mined destination reads it is trained with a squared-L2 InfoNCE, against the other contents "
    "seen so far in the same window, to return the content that the move chain actually routed "
    "there. Destinations are mined label-free from the batch: an observation that exactly "
    "duplicates an earlier observation under a non-duplicate command, with at least one silent "
    "(large-duplicate-cluster) command in between; an extra weight rides on pairs where some "
    "silent command between them is also close to the departure command and some other is close "
    "to the arrival command, which is the signature of a move naming both addresses. The InfoNCE "
    "label is the step's own next observation, so mis-mining can only mis-weight, never "
    "mislabel. No new parameters; gradient lands on the read/write address projections, the "
    "mutation and presence gates, the shared transition operator and the input block."
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
    "tr_dup_frac": 0.03,
    "tr_cmd_dup_frac": 0.01,
    "tr_free_min": 3,
    "tr_free_frac": 0.15,
    "tr_hi_z": 1.0,
    "tr_temp": 0.5,
    "tr_base_w": 0.15,
    "tr_bonus_w": 0.7,
    "tr_bridge_w": 0.5,
    "tr_hop_w": 0.35,
    "tr_max_hops": 4,
    "tr_max_examples": 256,
    "tr_nce_weight": 0.30,
    "tr_mse_weight": 0.02,
    "tr_ramp_steps": 200,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


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
    cfg["_layout"] = None
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))

    stash = {"reads": None}
    cfg["_stash"] = stash
    cfg["_tr_disabled"] = True

    original_reads = getattr(net, "_transition_reads", None)
    if callable(original_reads) and not bool(getattr(net, "_transport_read_hooked", False)):
        def _capturing_transition_reads(*args, **kwargs):
            out = original_reads(*args, **kwargs)
            if (torch.is_grad_enabled() and isinstance(out, torch.Tensor)
                    and out.dim() == 3 and out.requires_grad):
                stash["reads"] = out
            else:
                stash["reads"] = None
            return out

        net._transition_reads = _capturing_transition_reads
        net._transport_read_hooked = True
        cfg["_tr_disabled"] = False
    return cfg


@torch.no_grad()
def _mine_triples(cmd, obs, valid, path_thresh, change_floor):
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(_clean(cmd))
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


def _forwardmodel_arm(cfg, tok, cmd_mask, op, device):
    B, maxn = cmd_mask.shape
    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = _mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return None
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pre = _clean(obs[r, ti].detach())
    cmd_k = _clean(cmd[r, tk].detach())
    tgt = _clean(obs[r, tj].detach())

    pred = _clean(op(pre, cmd_k))

    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    return float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err


def _transport_arm(cfg, tok, cmd_mask, reads):
    B, maxn = cmd_mask.shape
    n = int(min(int(reads.size(1)), int(maxn)))
    if n < 3 or reads.size(0) != B:
        return None

    dev = tok.device
    o = _clean(tok[:, 1::2, :][:, :n, :]).float()
    c = _clean(tok[:, 0::2, :][:, :n, :]).float()
    valid = cmd_mask[:, :n].bool()
    Dm = float(o.size(-1))
    pos = torch.arange(n, device=dev)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv & (~eye)
        offf = off.float()
        cnt = offf.sum(dim=(1, 2)).clamp_min(1.0)

        osq = o.pow(2).sum(-1)
        d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
               - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / Dm
        ref_o = ((d2o * offf).sum(dim=(1, 2)) / cnt).clamp_min(1e-8)
        dup_o = off & (d2o <= (float(cfg["tr_dup_frac"]) * ref_o).view(B, 1, 1))

        csq = c.pow(2).sum(-1)
        d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
               - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / Dm
        ref_c = ((d2c * offf).sum(dim=(1, 2)) / cnt).clamp_min(1e-8)
        dup_c = off & (d2c <= (float(cfg["tr_cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        clus = dup_o.sum(dim=2)
        nvalid = valid.sum(dim=1)
        thr = torch.maximum(
            torch.full_like(nvalid, int(cfg["tr_free_min"])),
            (nvalid.float() * float(cfg["tr_free_frac"])).long(),
        ).clamp_min(2)
        freestep = valid & (clus >= thr.view(B, 1))
        contentful = valid & (~freestep)

        cu = _unit(c)
        sim = torch.bmm(cu, cu.transpose(1, 2))
        mean_s = (sim * offf).sum(dim=(1, 2)) / cnt
        var_s = (((sim - mean_s.view(B, 1, 1)) ** 2) * offf).sum(dim=(1, 2)) / cnt
        hi = (mean_s + float(cfg["tr_hi_z"]) * var_s.clamp_min(1e-8).sqrt()).view(B, 1, 1)
        sim_hi = off & (sim >= hi)

        tri = (pos.view(n, 1) < pos.view(1, n)).float()
        free_col = freestep.view(B, 1, n)
        after_row = (pos.view(1, n, 1) < pos.view(1, 1, n))

        depart = (sim_hi & free_col & after_row).float()
        arrive = (sim_hi & freestep.view(B, n, 1) & after_row).float()
        free_after = (free_col & after_row).float()

        depart_cnt = torch.matmul(depart, tri)
        arrive_cnt = torch.matmul(tri, arrive)
        hop_cnt = torch.matmul(free_after, tri)

        mined = (dup_o & (~dup_c)
                 & contentful.unsqueeze(2) & contentful.unsqueeze(1)
                 & after_row
                 & (hop_cnt > 0.5))
        bridged = mined & (depart_cnt > 0.5) & (arrive_cnt > 0.5)

        has_anchor = mined.any(dim=1)
        has_bridge = bridged.any(dim=1)
        anchor_i = torch.where(
            mined, pos.view(1, n, 1).expand(B, n, n), torch.full((1, 1, 1), -1, device=dev,
                                                                 dtype=torch.long)
        ).amax(dim=1)
        hop_sel = hop_cnt.gather(1, anchor_i.clamp_min(0).unsqueeze(1)).squeeze(1)
        hop_sel = hop_sel.clamp(0.0, float(cfg["tr_max_hops"])) * has_anchor.float()

        w = (float(cfg["tr_base_w"]) * contentful.float()
             + float(cfg["tr_bonus_w"]) * has_anchor.float()
             + float(cfg["tr_bridge_w"]) * has_bridge.float()
             + float(cfg["tr_hop_w"]) * hop_sel)
        w = w * valid.float()
        w[:, 0] = 0.0
        w = w + 1e-4 * (pos.view(1, n).float() / float(max(1, n - 1))) * (w > 0.0).float()

        sel = torch.nonzero(w > 0.0, as_tuple=False)
        if sel.numel() == 0:
            return None
        b_idx = sel[:, 0]
        j_idx = sel[:, 1]
        w_sel = w[b_idx, j_idx]
        cap = int(cfg["tr_max_examples"])
        if b_idx.numel() > cap:
            _, order = torch.topk(w_sel, cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]
            w_sel = w_sel[order]

        seen = pos.view(1, n) <= j_idx.view(-1, 1)
        is_j = F.one_hot(j_idx, n).bool()
        tgt_dup = dup_o[b_idx, :, j_idx]
        cand_mask = contentful[b_idx] & seen & ((~tgt_dup) | is_j)

        keep_row = cand_mask.sum(dim=1) >= 2
        if not bool(keep_row.any().item()):
            return None
        b_idx = b_idx[keep_row]
        j_idx = j_idx[keep_row]
        w_sel = w_sel[keep_row]
        cand_mask = cand_mask[keep_row]
        cand_sq = osq[b_idx]
        wn = w_sel / w_sel.sum().clamp_min(_EPS)

    cand = o[b_idx]
    truth = o[b_idx, j_idx]
    r_sel = reads[b_idx, j_idx].float()

    dot = torch.bmm(cand, r_sel.unsqueeze(2)).squeeze(2)
    d2 = (cand_sq + r_sel.pow(2).sum(-1, keepdim=True) - 2.0 * dot).clamp_min(0.0) / Dm
    logits = (-d2 / max(1e-3, float(cfg["tr_temp"]))).masked_fill(~cand_mask, _NEG)
    ce = F.cross_entropy(logits, j_idx, reduction="none")

    nce = (wn * ce).sum()
    mse = (wn * (r_sel - truth).pow(2).mean(dim=-1)).sum()
    return float(cfg["tr_nce_weight"]) * nce + float(cfg["tr_mse_weight"]) * mse


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0

    stash = cfg.get("_stash")
    reads = None
    if stash is not None:
        reads = stash.get("reads")
        stash["reads"] = None

    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0

    layout = cfg.get("_layout")
    if layout is None:
        layout = bool(_interleave_layout_ok(batch))
        cfg["_layout"] = layout
    if not layout:
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    step = int(cfg["_step"])
    total = None

    op = getattr(net, "transition_from_emb", None)
    ramp_fm = _smoothstep(step / max(1.0, float(cfg["ramp_steps"])))
    if (not cfg.get("_disabled", True)) and callable(op) and ramp_fm > 0.0:
        term = _forwardmodel_arm(cfg, tok, cmd_mask, op, device)
        if term is not None:
            total = ramp_fm * term

    ramp_tr = _smoothstep(step / max(1.0, float(cfg["tr_ramp_steps"])))
    if (not cfg.get("_tr_disabled", True)) and reads is not None and ramp_tr > 0.0:
        term = _transport_arm(cfg, tok, cmd_mask, reads)
        if term is not None:
            term = ramp_tr * term
            total = term if total is None else total + term

    if total is None:
        return 0.0

    out_loss = float(cfg["aux_weight"]) * total
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
        0.0 < vals["row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["change_floor"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        0.0 < vals["tr_dup_frac"] < 1.0,
        0.0 < vals["tr_cmd_dup_frac"] < 1.0,
        vals["tr_free_min"] >= 2.0,
        0.0 < vals["tr_free_frac"] < 1.0,
        vals["tr_hi_z"] >= 0.0,
        vals["tr_temp"] > 0.0,
        vals["tr_base_w"] >= 0.0,
        vals["tr_bonus_w"] >= 0.0,
        vals["tr_bridge_w"] >= 0.0,
        vals["tr_hop_w"] >= 0.0,
        vals["tr_max_hops"] >= 0.0,
        vals["tr_max_examples"] >= 1.0,
        vals["tr_nce_weight"] >= 0.0,
        vals["tr_mse_weight"] >= 0.0,
        vals["tr_ramp_steps"] >= 1.0,
    ]
    return all(checks)
