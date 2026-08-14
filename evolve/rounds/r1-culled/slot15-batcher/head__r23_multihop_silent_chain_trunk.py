import math

import torch
import torch.nn.functional as F

from evolve.chunks.head import r22_masked_endpoint_trunk_task as BASE

NAME = "r23_multihop_silent_chain_trunk"
DESCRIPTION = (
    "The r22 transition-consistency base term plus a MULTI-HOP SILENT-CHAIN task posed to the "
    "trunk: mine h consecutive gate-selected mutation commands whose observations are all removed, "
    "compress them behind a fully observed prefix, append a later non-mutation read command, and "
    "train the prediction at that read against the true observation with a same-verb-weighted "
    "L2-InfoNCE plus MSE anchor. h cycles over 1..max_hops with fallback, and several read commands "
    "from the same chain enter the same batch so the read command, not the disturbance, selects the "
    "answer. Zero new parameters; eval forward untouched."
)

_EPS = 1e-8

_DEFAULTS = {
    "chain_weight": 0.5,
    "chain_ramp_steps": 800,
    "chain_gate_floor": 0.8,
    "chain_max_hops": 3,
    "chain_hop_period": 3,
    "chain_max_pairs": 64,
    "chain_neg_extra": 128,
    "chain_tau": 0.25,
    "chain_kappa": 4.0,
    "chain_dup_cos": 0.98,
    "chain_mse_anchor": 0.05,
    "chain_period": 2,
    "chain_min_pairs": 2,
}


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _base_params(params):
    p = {k: v for k, v in (params or {}).items() if k in BASE._DEFAULTS}
    p["imag_weight"] = 0.0
    return p


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_base"] = BASE.wrap(net, D, **_base_params(params))
    cfg["_disabled"] = not BASE._has_gate(net)
    return cfg


def _mine_chain(gated, cmd_mask, h):
    B, maxn = cmd_mask.shape
    device = cmd_mask.device
    pos = torch.arange(maxn, device=device)
    score = torch.where(gated, (pos + 1).to(torch.float32).unsqueeze(0).expand(B, maxn),
                        torch.zeros(B, maxn, device=device))
    top = torch.topk(score, h, dim=1).values
    m_desc = (top.long() - 1)
    row_ok = (top > 0).all(dim=1) & (m_desc.min(dim=1).values >= 1)
    m_asc = m_desc.flip(1).clamp(0, maxn - 1)
    m_last = m_asc[:, -1]
    elig = cmd_mask & (~gated) & (pos.unsqueeze(0) > m_last.unsqueeze(1)) & row_ok.unsqueeze(1)
    nz = elig.nonzero(as_tuple=False)
    if nz.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, m_asc.new_zeros(0, h)
    rows, js = nz[:, 0], nz[:, 1]
    return rows, js, m_asc[rows]


def _chain_term(cfg, batch, net, device, step):
    if cfg.get("_disabled", True):
        return 0.0
    if float(cfg["chain_weight"]) <= 0.0:
        return 0.0
    period = max(1, int(cfg["chain_period"]))
    if (step % period) != 0:
        return 0.0
    ramp = _smoothstep(step / max(1.0, float(cfg["chain_ramp_steps"])))
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

    with torch.no_grad():
        w = BASE._gate_values(net, _clean(cmd).reshape(-1, Dd)).reshape(B, maxn)
        w = torch.nan_to_num(w, nan=0.0)
    gated = (w > float(cfg["chain_gate_floor"])) & cmd_mask

    hmax = max(1, min(int(cfg["chain_max_hops"]), maxn - 2))
    hop_period = max(1, int(cfg["chain_hop_period"]))
    sched = 1 + ((step // hop_period) % hmax)
    min_pairs = max(2, int(cfg["chain_min_pairs"]))

    rows = js = None
    M = None
    h = 0
    for cand_h in range(sched, 0, -1):
        r, j, m = _mine_chain(gated, cmd_mask, cand_h)
        if r.numel() >= min_pairs:
            rows, js, M, h = r, j, m, cand_h
            break
    if rows is None:
        return 0.0

    wsel = w[rows.unsqueeze(1), M]
    pw = wsel.clamp_min(1e-6).log().mean(dim=1).exp()

    cap = int(cfg["chain_max_pairs"])
    if rows.numel() > cap:
        sel = torch.multinomial(pw.float().clamp_min(1e-6), cap, replacement=False)
        rows, js, M, pw = rows[sel], js[sel], M[sel], pw[sel]
    P = rows.numel()

    m1 = M[:, :1]
    Lm = int(2 * (int(m1.max().item()) + h) + 1)
    c = torch.arange(Lm, device=device).unsqueeze(0).expand(P, Lm)
    base = 2 * m1
    is_pre = c < base
    even = (c % 2) == 0
    t_idx = torch.div(c - base, 2, rounding_mode="floor")
    in_mut = even & (c >= base) & (t_idx >= 0) & (t_idx < h)
    is_read = c == (base + 2 * h)
    mt = M.gather(1, t_idx.clamp(0, h - 1))
    read_src = (2 * js).unsqueeze(1).expand(P, Lm)
    src = torch.where(is_pre, c, torch.where(in_mut, 2 * mt,
                                             torch.where(is_read, read_src, torch.zeros_like(c))))
    src = src.clamp(0, L - 1)
    keep = is_pre | in_mut | is_read

    tok2 = tok[rows].gather(1, src.unsqueeze(-1).expand(P, Lm, Dd))
    tok2 = _clean(tok2 * keep.unsqueeze(-1).to(tok2.dtype))
    types2 = (c % 2).long().contiguous()
    kp2 = (~keep).contiguous()

    pred_all, _ = net(tok2, types2, kp2)
    read_col = (base + 2 * h).squeeze(1)
    ar = torch.arange(P, device=device)
    pred = _clean(pred_all[ar, read_col])

    zj = _clean(tgt_full[rows, js].detach())
    cj = cmd[rows, js].detach()

    flat_t = tgt_full[cmd_mask].detach()
    flat_c = cmd[cmd_mask].detach()
    extra = int(cfg["chain_neg_extra"])
    if flat_t.shape[0] > extra > 0:
        es = torch.randint(0, flat_t.shape[0], (extra,), device=device)
        neg_t, neg_c = flat_t[es], flat_c[es]
    else:
        neg_t, neg_c = flat_t, flat_c
    cand_t = _clean(torch.cat([zj, neg_t], dim=0))
    cand_c = torch.cat([cj, neg_c], dim=0)

    d2 = (pred.pow(2).sum(1, keepdim=True) + cand_t.pow(2).sum(1).unsqueeze(0)
          - 2.0 * pred @ cand_t.t()).clamp_min(0.0) / float(Dd)

    with torch.no_grad():
        vsim = (F.normalize(cj, dim=-1) @ F.normalize(cand_c, dim=-1).t()).clamp(-1.0, 1.0)
        a = 1.0 + float(cfg["chain_kappa"]) * vsim.clamp_min(0.0)
        tsim = F.normalize(zj, dim=-1) @ F.normalize(cand_t, dim=-1).t()
        dup = tsim > float(cfg["chain_dup_cos"])
        dup[ar, ar] = False
        loga = a.clamp_min(1e-9).log().masked_fill(dup, float("-inf"))
        loga[ar, ar] = 0.0

    logits = -d2 / float(cfg["chain_tau"]) + loga
    nll = F.cross_entropy(logits, ar, reduction="none")
    wn = (pw.to(pred.dtype) / pw.to(pred.dtype).sum().clamp_min(_EPS)).detach()
    nce = (wn * nll).sum()
    anchor = (wn * (pred - zj).pow(2).mean(dim=-1)).sum()
    return float(cfg["chain_weight"]) * ramp * (nce + float(cfg["chain_mse_anchor"]) * anchor)


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    base_term = BASE.aux_loss(cfg.get("_base"), batch, net, device)
    if not BASE._interleave_layout_ok(batch):
        return base_term
    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    term = _chain_term(cfg, batch, net, device, cfg["_step"])
    if torch.is_tensor(term):
        if not bool(torch.isfinite(term).item()):
            return base_term
        return base_term + term
    if term:
        return base_term + term
    return base_term


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
        vals["chain_weight"] >= 0.0,
        vals["chain_ramp_steps"] >= 1.0,
        0.0 < vals["chain_gate_floor"] < 1.0,
        vals["chain_max_hops"] >= 1.0,
        vals["chain_hop_period"] >= 1.0,
        vals["chain_max_pairs"] >= 2.0,
        vals["chain_neg_extra"] >= 0.0,
        vals["chain_tau"] > 0.0,
        vals["chain_kappa"] >= 0.0,
        0.0 < vals["chain_dup_cos"] <= 1.0,
        vals["chain_mse_anchor"] >= 0.0,
        vals["chain_period"] >= 1.0,
        vals["chain_min_pairs"] >= 2.0,
    ]
    return all(checks) and BASE.leak_safe(mod, _base_params(params))
