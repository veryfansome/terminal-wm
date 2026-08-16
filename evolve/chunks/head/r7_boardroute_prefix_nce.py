import math

import torch

NAME = "r7_boardroute_prefix_nce"
DESCRIPTION = (
    "The r18 transition forward-model consistency arm, kept verbatim, plus a second train-only "
    "arm that supervises the model's OWN next-observation prediction exactly at the reads that a "
    "silent mutation chain routed. Windows are mined from embedding geometry alone: within a row, "
    "the largest exact-duplicate observation cluster is the silent/mutating set; a duplicate pair "
    "of NON-silent observations separated by at least one silent step is a content that was "
    "exposed at i, carried by the chain, and read again at j. At the command position j the "
    "prediction is scored by an InfoNCE in the eval's own mean-squared-distance decision "
    "variable against the board's other distinct contents whose first exposure lies strictly in "
    "the prefix (so every candidate is causal), with the positive being the exposure vector at i. "
    "Windows are weighted by chain depth, so the deep chains carry the gradient. No new "
    "parameters, no change to forward; the prediction is read off a forward hook so the arm costs "
    "no second forward pass."
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
    "legacy_weight": 1.0,
    "nce_weight": 0.25,
    "nce_tau": 0.25,
    "dup_thresh": 0.98,
    "silent_min": 3.0,
    "min_chain": 1.0,
    "max_cands": 8.0,
    "max_windows": 96.0,
    "depth_pow": 1.0,
    "depth_cap": 8.0,
}

_EPS = 1e-8
_NEG_INF = -1e30


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


def _make_hook(cfg):
    def hook(module, inputs, output):
        if not (torch.is_grad_enabled() and module.training):
            return
        out = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(out) and out.dim() == 3 and out.requires_grad:
            cfg["_pred"] = out
    return hook


def wrap(net, D, **params):
    existing = getattr(net, "_boardroute_head_state", None)
    if existing is not None:
        return existing
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    cfg["_route_disabled"] = getattr(net, "target_module", None) is not None
    cfg["_pred"] = None
    cfg["_hook"] = net.register_forward_hook(_make_hook(cfg))
    net._boardroute_head_state = cfg
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


@torch.no_grad()
def _mine_windows(obs, valid, dup_thresh, silent_min, min_chain, max_cands):
    R, n, _ = obs.shape
    device = obs.device
    if n < 3:
        return None

    u = _unit(_clean(obs))
    sim = torch.bmm(u, u.transpose(1, 2))
    vv = valid.unsqueeze(2) & valid.unsqueeze(1)
    dup = (sim >= dup_thresh) & vv

    cnt = dup.sum(dim=2)
    anchor = cnt.argmax(dim=1)
    rows = torch.arange(R, device=device)
    top = cnt[rows, anchor]
    silent = dup[rows, anchor] & (top >= silent_min).unsqueeze(1) & valid
    content = valid & ~silent

    pos = torch.arange(n, device=device)
    before = (pos.unsqueeze(1) < pos.unsqueeze(0)).unsqueeze(0)
    after_c = (pos.unsqueeze(1) > pos.unsqueeze(0)).unsqueeze(0)

    prefix = torch.cat(
        [torch.zeros(R, 1, device=device, dtype=torch.long), silent.long().cumsum(dim=1)], dim=1
    )
    depth = prefix[:, :n].unsqueeze(1) - prefix[:, 1:].unsqueeze(2)

    ok = dup & before & content.unsqueeze(2) & content.unsqueeze(1) & (depth >= min_chain)
    posf = pos.view(1, n, 1).expand(R, n, n)
    i_of_j = torch.where(ok, posf, torch.full_like(posf, n)).amin(dim=1)
    has = i_of_j < n
    d_sel = depth.gather(1, i_of_j.clamp(0, n - 1).unsqueeze(1)).squeeze(1)

    earlier_dup = (dup & before & content.unsqueeze(2)).any(dim=1)
    rep = content & ~earlier_dup

    cand_mask = rep.unsqueeze(1) & after_c
    ncand = cand_mask.sum(dim=2)

    is_pos = pos.view(1, 1, n) == i_of_j.unsqueeze(2)
    big = 2 * n + 10
    base = torch.where(
        is_pos,
        torch.zeros_like(posf),
        1 + (n - pos.view(1, 1, n)).expand(R, n, n),
    )
    key = torch.where(cand_mask, base, torch.full_like(base, big))
    sorted_key, sorted_idx = key.sort(dim=2)

    C = max(2, min(int(max_cands), n))
    cand_idx = sorted_idx[:, :, :C]
    cand_ok = sorted_key[:, :, :C] < big

    keep = has & (ncand >= 2) & cand_ok[:, :, 0]
    return keep, i_of_j, d_sel, cand_idx, cand_ok


def _route_loss(cfg, batch, net, device, tok, valid, sel, nrows):
    pred_full = cfg.get("_pred", None)
    cfg["_pred"] = None
    B, L, _ = tok.shape
    if not (torch.is_tensor(pred_full) and pred_full.dim() == 3
            and pred_full.shape[0] == B and pred_full.shape[1] == L):
        pred_full = net(batch["tok"], batch["types"], batch["key_pad"])[0]
        cfg["_pred"] = None
    if not pred_full.requires_grad:
        return 0.0

    maxn = valid.shape[1]
    obs = tok[sel][:, 1::2][:, :maxn]
    vsel = valid[sel]
    mined = _mine_windows(
        obs,
        vsel,
        float(cfg["dup_thresh"]),
        int(cfg["silent_min"]),
        int(cfg["min_chain"]),
        int(cfg["max_cands"]),
    )
    if mined is None:
        return 0.0
    keep, i_of_j, d_sel, cand_idx, cand_ok = mined
    nz = torch.nonzero(keep, as_tuple=False)
    if nz.numel() == 0:
        return 0.0

    wr = nz[:, 0]
    wj = nz[:, 1]
    depth = d_sel[wr, wj].to(torch.float32)
    cap = int(cfg["max_windows"])
    if wr.numel() > cap:
        pick = torch.topk(depth, cap).indices
        wr = wr[pick]
        wj = wj[pick]
        depth = depth[pick]

    cand_pos = cand_idx[wr, wj]
    cand_live = cand_ok[wr, wj]
    cand_vec = _clean(obs[wr.unsqueeze(1), cand_pos])

    pred_cmd = pred_full[:, 0::2][:, :maxn]
    pred_at = _clean(pred_cmd[sel][wr, wj])

    d2 = (pred_at.unsqueeze(1) - cand_vec).pow(2).mean(dim=-1)
    logits = (-d2 / max(1e-4, float(cfg["nce_tau"]))).masked_fill(~cand_live, _NEG_INF)
    nll = -torch.log_softmax(logits, dim=-1)[:, 0]

    w = depth.clamp(1.0, float(cfg["depth_cap"])).pow(float(cfg["depth_pow"]))
    w = (w / w.sum().clamp_min(_EPS)).detach().to(nll.dtype)
    return (w * nll).sum()


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        cfg["_pred"] = None
        return 0.0
    if not _interleave_layout_ok(batch):
        cfg["_pred"] = None
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        cfg["_pred"] = None
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        cfg["_pred"] = None
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]

    total = 0.0

    op = getattr(net, "transition_from_emb", None)
    legacy_w = float(cfg.get("legacy_weight", 0.0))
    if callable(op) and not cfg.get("_disabled", True) and legacy_w > 0.0:
        cmd = tok[sel][:, 0::2][:, :maxn]
        obs = tok[sel][:, 1::2][:, :maxn]
        valid = cmd_mask[sel]
        r, ti, tk, tj, w = _mine_triples(
            cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
        )
        if r.numel() > 0:
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
            total = total + legacy_w * (
                float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err
            )

    nce_w = float(cfg.get("nce_weight", 0.0))
    if nce_w > 0.0 and not cfg.get("_route_disabled", False):
        route = _route_loss(cfg, batch, net, device, tok, cmd_mask, sel, nrows)
        total = total + nce_w * route
    else:
        cfg["_pred"] = None

    if isinstance(total, float):
        return 0.0
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
        vals["legacy_weight"] >= 0.0,
        vals["nce_weight"] >= 0.0,
        vals["nce_tau"] > 0.0,
        0.0 < vals["dup_thresh"] <= 1.0,
        vals["silent_min"] >= 2.0,
        vals["min_chain"] >= 1.0,
        vals["max_cands"] >= 2.0,
        vals["max_windows"] >= 1.0,
        vals["depth_pow"] >= 0.0,
        vals["depth_cap"] >= 1.0,
    ]
    return all(checks)
