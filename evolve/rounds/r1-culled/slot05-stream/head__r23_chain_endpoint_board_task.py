import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r23_chain_endpoint_board_task"
DESCRIPTION = (
    "The r18 forward-model consistency term, the r22 in-layout masked-endpoint trunk task and "
    "the r20 direct imaginer supervision, fused. The endpoint task now selects its (mutation m, "
    "read j) pairs from the stream's causal chain annotations when they are present — j is a read "
    "whose content arrived through at least one move, m the last move before it — and its "
    "contrastive candidate set is augmented with same-sequence other-slot contents, so the "
    "discrimination is between the files on one board rather than across the batch."
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
    "imag_ramp_steps": 600,
    "imag_gate_floor": 0.8,
    "imag_max_pairs": 48,
    "imag_neg_extra": 96,
    "imag_tau": 0.25,
    "imag_kappa": 4.0,
    "imag_dup_cos": 0.98,
    "imag_mse_anchor": 0.05,
    "imag_period": 3,
    "board_negs": 3,
    "chain_min_depth": 1,
    "depth_boost": 1.0,
    "chain_prior": 4.0,
    "iw_weight": 0.5,
    "iw_ramp_steps": 400,
    "iw_path_thresh": 0.60,
    "iw_max_examples": 256,
    "iw_tau": 0.25,
    "iw_dup_delta": 0.05,
    "iw_mse": 0.05,
    "iw_wfloor": 0.15,
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


def _has_gate(net):
    for attr in ("tr_mut_gate", "cmd_proj", "type_emb", "in_norm"):
        if getattr(net, attr, None) is None:
            return False
    return True


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    cfg["_ep_disabled"] = not _has_gate(net)
    cfg["_iw_disabled"] = not isinstance(getattr(net, "imaginer", None), nn.Module)
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


def _transition_term(cfg, batch, net, device, ramp):
    if cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
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


@torch.no_grad()
def _gate_values(net, cmd_flat):
    idx0 = torch.zeros(cmd_flat.size(0), dtype=torch.long, device=cmd_flat.device)
    x = net.in_norm(net.cmd_proj(cmd_flat) + net.type_emb(idx0))
    return torch.sigmoid(net.tr_mut_gate(x)).squeeze(-1)


@torch.no_grad()
def _chain_pairs(cfg, batch, device):
    need = ("step_kind", "chain_depth", "cmd_mask")
    for k in need:
        if k not in batch:
            return None
    kind = batch["step_kind"].long()
    depth = batch["chain_depth"].long()
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    pos = torch.arange(maxn, device=device)
    is_move = (kind == 2) | (kind == 3)
    mpos = torch.where(is_move & cmd_mask, pos.unsqueeze(0).expand(B, maxn),
                       torch.full((B, maxn), -1, dtype=torch.long, device=device))
    mmax = mpos.cummax(dim=1).values
    m_of = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=device),
                      mmax[:, :-1]], dim=1)
    elig = cmd_mask & (kind == 1) & (depth >= int(cfg["chain_min_depth"])) & (m_of >= 1)
    if not bool(elig.any()):
        return None
    nz = elig.nonzero(as_tuple=False)
    rows, js = nz[:, 0], nz[:, 1]
    ks = m_of[rows, js]
    pw = 1.0 + float(cfg["depth_boost"]) * depth[rows, js].to(torch.float32)
    return rows, js, ks, pw


@torch.no_grad()
def _gate_pairs(cfg, batch, net, device):
    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    Dd = tok.shape[2]
    cmd = tok[:, 0::2][:, :maxn]
    w = _gate_values(net, torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)
                     .reshape(-1, Dd)).reshape(B, maxn)
    w = torch.nan_to_num(w, nan=0.0)
    floor = float(cfg["imag_gate_floor"])
    gated = (w > floor) & cmd_mask
    pos = torch.arange(maxn, device=device)
    gpos = torch.where(gated, pos.unsqueeze(0).expand(B, maxn),
                       torch.full((B, maxn), -1, dtype=torch.long, device=device))
    kmax = gpos.cummax(dim=1).values
    k_of = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=device),
                      kmax[:, :-1]], dim=1)
    elig = cmd_mask & (~gated) & (k_of >= 1)
    if not bool(elig.any()):
        return None
    nz = elig.nonzero(as_tuple=False)
    rows, js = nz[:, 0], nz[:, 1]
    ks = k_of[rows, js]
    pw = w[rows, ks].to(torch.float32)
    return rows, js, ks, pw


def _endpoint_term(cfg, batch, net, device, step):
    if cfg.get("_ep_disabled", True):
        return 0.0
    if float(cfg["imag_weight"]) <= 0.0:
        return 0.0
    period = max(1, int(cfg["imag_period"]))
    if (step % period) != 0:
        return 0.0
    ramp = _smoothstep(step / max(1.0, float(cfg["imag_ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    tgt_full = batch["tgt"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0
    L = tok.shape[1]
    Dd = tok.shape[2]
    cmd = tok[:, 0::2][:, :maxn]

    cap = int(cfg["imag_max_pairs"])
    sel_c = _chain_pairs(cfg, batch, device)
    n_chain = 0 if sel_c is None else int(sel_c[0].numel())
    sel_g = _gate_pairs(cfg, batch, net, device) if n_chain < cap else None
    parts = [s for s in (sel_c, sel_g) if s is not None]
    if not parts:
        return 0.0
    if len(parts) == 1:
        rows, js, ks, pw = parts[0]
        if sel_c is not None:
            pw = pw * float(cfg["chain_prior"])
    else:
        rows = torch.cat([parts[0][0], parts[1][0]])
        js = torch.cat([parts[0][1], parts[1][1]])
        ks = torch.cat([parts[0][2], parts[1][2]])
        pw = torch.cat([parts[0][3] * float(cfg["chain_prior"]), parts[1][3]])

    P = rows.numel()
    if P > cap:
        pick = torch.multinomial(pw.clamp_min(1e-6).float(), cap, replacement=False)
        rows, js, ks, pw = rows[pick], js[pick], ks[pick], pw[pick]
        P = cap

    mm = ks
    rr = js
    Lm = int(2 * int(mm.max().item()) + 3)
    p2 = torch.arange(Lm, device=device).unsqueeze(0).expand(P, Lm)
    end_col = (2 * mm + 2).unsqueeze(1)
    src = torch.where(p2 == end_col, (2 * rr).unsqueeze(1), p2.clamp_max(L - 1))
    keep = (p2 <= (2 * mm).unsqueeze(1)) | (p2 == end_col)
    tok2 = tok[rows].gather(1, src.unsqueeze(-1).expand(P, Lm, Dd))
    tok2 = tok2 * keep.unsqueeze(-1).to(tok2.dtype)
    types2 = ((p2 % 2 == 1) & (p2 <= (2 * mm + 1).unsqueeze(1))).long()
    kp2 = ~keep
    tok2 = torch.nan_to_num(tok2, nan=0.0, posinf=1e4, neginf=-1e4)

    pred_all, _ = net(tok2, types2, kp2)
    pred = pred_all[torch.arange(P, device=device), 2 * mm + 2]
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    zj = torch.nan_to_num(tgt_full[rows, rr].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    cj = cmd[rows, rr].detach()

    flat_t = tgt_full[cmd_mask].detach()
    flat_c = cmd[cmd_mask].detach()
    extra = int(cfg["imag_neg_extra"])
    if flat_t.shape[0] > extra > 0:
        es = torch.randint(0, flat_t.shape[0], (extra,), device=device)
        neg_t, neg_c = flat_t[es], flat_c[es]
    else:
        neg_t, neg_c = flat_t, flat_c

    parts_t = [zj, neg_t]
    parts_c = [cj, neg_c]
    nb = int(cfg["board_negs"])
    if nb > 0 and "slot_id" in batch and "step_kind" in batch:
        with torch.no_grad():
            slot = batch["slot_id"].long()
            kind = batch["step_kind"].long()
            avail = (cmd_mask & (kind == 1))[rows].to(torch.float32)
            diff = (slot[rows] != slot[rows, rr].unsqueeze(1)).to(torch.float32)
            bm = avail * diff
            fallback = cmd_mask[rows].to(torch.float32)
            probs = torch.where((bm.sum(dim=1, keepdim=True) > 0.0).expand_as(bm), bm, fallback)
            bidx = torch.multinomial(probs.clamp_min(1e-9), nb, replacement=True)
            br = rows.unsqueeze(1).expand(P, nb)
        parts_t.append(tgt_full[br.reshape(-1), bidx.reshape(-1)].detach())
        parts_c.append(cmd[br.reshape(-1), bidx.reshape(-1)].detach())

    cand_t = torch.nan_to_num(torch.cat(parts_t, dim=0), nan=0.0, posinf=1e4, neginf=-1e4)
    cand_c = torch.cat(parts_c, dim=0)

    d2 = (pred.pow(2).sum(1, keepdim=True) + cand_t.pow(2).sum(1).unsqueeze(0)
          - 2.0 * pred @ cand_t.t()).clamp_min(0.0) / float(Dd)

    ar = torch.arange(P, device=device)
    with torch.no_grad():
        vsim = (F.normalize(cj, dim=-1) @ F.normalize(cand_c, dim=-1).t()).clamp(-1.0, 1.0)
        a = 1.0 + float(cfg["imag_kappa"]) * vsim.clamp_min(0.0)
        tsim = F.normalize(zj, dim=-1) @ F.normalize(cand_t, dim=-1).t()
        dup = tsim > float(cfg["imag_dup_cos"])
        dup[ar, ar] = False
        loga = a.clamp_min(1e-9).log().masked_fill(dup, float("-inf"))
        loga[ar, ar] = 0.0

    logits = -d2 / float(cfg["imag_tau"]) + loga
    nll = F.cross_entropy(logits, ar, reduction="none")
    wn = (pw.to(pred.dtype) / pw.to(pred.dtype).sum().clamp_min(_EPS)).detach()
    nce = (wn * nll).sum()
    anchor = (wn * (pred - zj).pow(2).mean(dim=-1)).sum()
    return float(cfg["imag_weight"]) * ramp * (nce + float(cfg["imag_mse_anchor"]) * anchor)


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
    cp = getattr(net, "cmd_proj", None)
    inn = getattr(net, "in_norm", None)
    tmg = getattr(net, "tr_mut_gate", None)
    te = getattr(net, "type_emb", None)
    if (isinstance(cp, nn.Module) and isinstance(inn, nn.Module)
            and isinstance(tmg, nn.Module) and isinstance(te, nn.Module)):
        idx0 = torch.zeros(B, maxn, dtype=torch.long, device=device)
        x_cmd = inn(cp(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)) + te(idx0))
        w_mut = torch.sigmoid(tmg(x_cmd)).squeeze(-1)
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


def _imaginer_term(cfg, batch, net, device, ramp):
    if cfg.get("_iw_disabled", True) or float(cfg.get("iw_weight", 0.0)) <= 0.0:
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

    sb, sk, sj, w = _mine_endpoint_pairs(
        cmd, obs, cmd_mask, net, float(cfg["iw_path_thresh"]), float(cfg["iw_wfloor"])
    )
    if sb.numel() < 2:
        return 0.0
    if sb.numel() > int(cfg["iw_max_examples"]):
        w, order = torch.topk(w, int(cfg["iw_max_examples"]))
        sb = sb[order]; sk = sk[order]; sj = sj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pair_cat = torch.cat([cmd, obs], dim=-1)[sb]
    pos = torch.arange(maxn, device=device)
    pmask = cmd_mask[sb] & (pos.unsqueeze(0) < sk.unsqueeze(1))
    c_m = cmd[sb, sk]
    c_r = cmd[sb, sj]
    lab = torch.nan_to_num(obs[sb, sj], nan=0.0, posinf=1e4, neginf=-1e4)

    pred = imaginer(pair_cat, pmask, c_m, c_r)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    total = _imag_nce(pred, lab, w, cfg["iw_tau"], cfg["iw_dup_delta"], cfg["iw_mse"])
    return float(cfg["iw_weight"]) * ramp * total


def _accumulate(total, term):
    if torch.is_tensor(term):
        if bool(torch.isfinite(term).item()):
            return total + term
        return total
    if term:
        return total + term
    return total


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    step = cfg["_step"]
    ramp = _smoothstep(step / max(1.0, float(cfg["ramp_steps"])))
    ramp_iw = _smoothstep(step / max(1.0, float(cfg["iw_ramp_steps"])))

    total = 0.0
    if ramp > 0.0:
        total = _accumulate(total, _transition_term(cfg, batch, net, device, ramp))
    if ramp_iw > 0.0:
        total = _accumulate(total, _imaginer_term(cfg, batch, net, device, ramp_iw))
    total = _accumulate(total, _endpoint_term(cfg, batch, net, device, step))

    if torch.is_tensor(total) and not bool(torch.isfinite(total).item()):
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
        0.0 < vals["imag_gate_floor"] < 1.0,
        vals["imag_max_pairs"] >= 1.0,
        vals["imag_neg_extra"] >= 0.0,
        vals["imag_tau"] > 0.0,
        vals["imag_kappa"] >= 0.0,
        0.0 < vals["imag_dup_cos"] <= 1.0,
        vals["imag_mse_anchor"] >= 0.0,
        vals["imag_period"] >= 1.0,
        vals["board_negs"] >= 0.0,
        vals["chain_min_depth"] >= 0.0,
        vals["depth_boost"] >= 0.0,
        vals["chain_prior"] > 0.0,
        vals["iw_weight"] >= 0.0,
        vals["iw_ramp_steps"] >= 1.0,
        -1.0 <= vals["iw_path_thresh"] < 1.0,
        vals["iw_max_examples"] >= 2.0,
        vals["iw_tau"] > 0.0,
        vals["iw_dup_delta"] >= 0.0,
        vals["iw_mse"] >= 0.0,
        vals["iw_wfloor"] >= 0.0,
    ]
    return all(checks)
