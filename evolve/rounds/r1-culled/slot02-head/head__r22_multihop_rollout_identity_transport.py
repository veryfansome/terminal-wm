import math

import torch
import torch.nn as nn

NAME = "r22_multihop_rollout_identity_transport"
DESCRIPTION = (
    "Multi-hop rollout supervision for the shared latent-transition operator. Mines, for every "
    "candidate terminal read, the ordered chain of its last same-path touches and rolls the "
    "operator forward 2-3 times from the earliest revealed content to the terminal read. Trains "
    "that rollout with a duplicate-masked L2-InfoNCE plus three paired hinges that hold one factor "
    "fixed at a time: same command chain / swapped initial content, same initial content / swapped "
    "command chain, and same commands in exchanged order. Adds an obs-absent arch rollout that "
    "feeds the net a synthetic type-2 imagined chain so the path memory must compose the operator "
    "without intermediate observations, paired against a start-swapped copy in the same forward. "
    "The r18 single-hop consistency is retained at half weight as an observation-space anchor. "
    "Head is parameter-free and never re-points net.forward; eval forward is the arch's own."
)

_DEFAULTS = {
    "row_frac": 0.6,
    "path_thresh": 0.60,
    "change_floor": 0.25,
    "max_examples": 512,
    "cos_weight": 0.10,
    "mse_weight": 0.02,
    "aux_weight": 0.5,
    "ramp_steps": 400,
    "chain_hops": 3,
    "chain_min_hops": 2,
    "chain_thresh": 0.60,
    "chain_change_floor": 0.25,
    "chain_max_examples": 32,
    "chain_tau": 0.25,
    "chain_dupe_cos": 0.98,
    "chain_margin": 0.10,
    "chain_comm_margin": 0.30,
    "chain_comm_max_cos": 0.99,
    "chain_nce_w": 1.0,
    "chain_mse_w": 0.05,
    "chain_start_w": 1.0,
    "chain_swap_w": 0.5,
    "chain_comm_w": 0.25,
    "chain_weight": 0.25,
    "chain_ramp_start": 100,
    "chain_ramp_steps": 400,
    "imag_max_examples": 16,
    "imag_tau": 0.25,
    "imag_margin": 0.10,
    "imag_mse_w": 0.05,
    "imag_weight": 0.10,
    "imag_every": 3,
    "imag_ramp_start": 250,
    "imag_ramp_steps": 500,
}

_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _cos_dist(a, b):
    return 1.0 - (_unit(a) * _unit(b)).sum(dim=-1).clamp(-1.0, 1.0)


def _rel_sep(a, b):
    num = (a - b).pow(2).mean(dim=-1).add(1e-6).sqrt()
    den = (0.5 * (a.pow(2).mean(dim=-1) + b.pow(2).mean(dim=-1))).add(1e-6).sqrt()
    return num / den


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
    te = getattr(net, "type_emb", None)
    three_types = isinstance(te, nn.Embedding) and int(te.num_embeddings) >= 3
    cfg["_imag_disabled"] = cfg["_disabled"] or not three_types
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


def _transition_term(cfg, batch, net, device, ramp):
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

    pre = _clean(obs[r, ti].detach())
    cmd_k = _clean(cmd[r, tk].detach())
    tgt = _clean(obs[r, tj].detach())

    pred = _clean(op(pre, cmd_k))

    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    total = float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err
    return float(cfg["aux_weight"]) * ramp * total


@torch.no_grad()
def _mine_chains(cmd, obs, valid, thresh, change_floor, slots, min_slots):
    B, n, Dd = cmd.shape
    device = cmd.device
    cu = _unit(_clean(cmd))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    vm = valid.bool()

    pos = torch.arange(n, device=device)
    before = (pos.unsqueeze(0) < pos.unsqueeze(1)).unsqueeze(0)
    cand = (sim > thresh) & vm.unsqueeze(1) & vm.unsqueeze(2) & before

    k = min(int(slots), n)
    posf = pos.view(1, 1, n).expand(B, n, n)
    score = torch.where(cand, posf, torch.full_like(posf, -1))
    top = torch.flip(torch.topk(score, k, dim=2).values, dims=[2])
    if k < int(slots):
        pad = torch.full((B, n, int(slots) - k), -1, device=device, dtype=top.dtype)
        top = torch.cat([pad, top], dim=2)

    sv = top >= 0
    n_used = sv.sum(dim=2)
    t0 = (int(slots) - n_used).clamp(0, int(slots) - 1)
    start_idx = torch.gather(top, 2, t0.unsqueeze(-1)).squeeze(-1).clamp_min(0)
    obs_start = torch.gather(obs, 1, start_idx.unsqueeze(-1).expand(B, n, Dd))
    change = (obs_start - obs).pow(2).mean(dim=-1)
    sim_sj = torch.gather(sim, 2, start_idx.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)
    w = sim_sj * change * n_used.to(cmd.dtype)

    keep = (n_used >= int(min_slots)) & vm & (change >= float(change_floor))
    keep = keep & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)
    rb, rj = nz[:, 0], nz[:, 1]
    wsel = w[rb, rj]
    order = torch.argsort(wsel, descending=True)
    rb, rj = rb[order], rj[order]
    return rb, rj, top[rb, rj], sv[rb, rj], t0[rb, rj], wsel[order]


def _rollout(op, start, cmd_slots, hop_mask):
    state = start
    for t in range(cmd_slots.size(1)):
        m = hop_mask[:, t].unsqueeze(-1)
        if not bool(m.any().item()):
            continue
        state = torch.where(m, _clean(op(state, cmd_slots[:, t, :])), state)
    return state


def _nce(pred, tgt, w, tau, dupe_cos):
    n, d = pred.shape
    dist = (pred.pow(2).sum(1, keepdim=True) + tgt.pow(2).sum(1) - 2.0 * pred @ tgt.t())
    dist = dist.clamp_min(0.0) / float(d)
    with torch.no_grad():
        tu = _unit(tgt)
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        dup = ((tu @ tu.t()) > float(dupe_cos)) & ~eye
    logits = (-dist / float(tau)).masked_fill(dup, -1e30)
    nll = -torch.log_softmax(logits, dim=1).diagonal()
    return (w * nll).sum()


def _gather_chain(cmd, obs, rb, rj, idx, sv, t0, device):
    M = rb.numel()
    gi = idx.clamp_min(0)
    cmd_slots = _clean(cmd[rb.unsqueeze(1), gi].detach())
    obs_slots = obs[rb.unsqueeze(1), gi].detach()
    ar = torch.arange(cmd_slots.size(1), device=device)
    hop = sv & (ar.unsqueeze(0) > t0.unsqueeze(1))
    start = _clean(obs_slots[torch.arange(M, device=device), t0])
    tgt = _clean(obs[rb, rj].detach())
    qcmd = _clean(cmd[rb, rj].detach())
    return cmd_slots, hop, start, tgt, qcmd, ar


def _chain_term(cfg, net, cmd, obs, mined, device):
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    rb, rj, idx, sv, t0, wraw = mined
    cap = int(cfg["chain_max_examples"])
    M = min(rb.numel(), cap)
    if M < 2:
        return 0.0
    rb, rj, idx, sv, t0, wraw = rb[:M], rj[:M], idx[:M], sv[:M], t0[:M], wraw[:M]
    w = (wraw / wraw.sum().clamp_min(_EPS)).detach().to(cmd.dtype)

    cmd_slots, hop, start, tgt, _, _ = _gather_chain(cmd, obs, rb, rj, idx, sv, t0, device)
    slots = cmd_slots.size(1)

    pred = _rollout(op, start, cmd_slots, hop)
    perm = torch.roll(torch.arange(M, device=device), 1)
    pred_start_swap = _rollout(op, start[perm], cmd_slots, hop)
    pred_chain_swap = _rollout(op, start, cmd_slots[perm], hop[perm])

    tail_a = cmd_slots[:, slots - 2, :]
    tail_b = cmd_slots[:, slots - 1, :]
    distinct = (_unit(tail_a) * _unit(tail_b)).sum(dim=-1) < float(cfg["chain_comm_max_cos"])
    both_tail = hop[:, slots - 1] & hop[:, slots - 2] & distinct
    cmd_order_swap = cmd_slots.clone()
    pick = both_tail.unsqueeze(-1)
    cmd_order_swap[:, slots - 2, :] = torch.where(pick, tail_b, tail_a)
    cmd_order_swap[:, slots - 1, :] = torch.where(pick, tail_a, tail_b)
    pred_order_swap = _rollout(op, start, cmd_order_swap, hop)

    d_true = _cos_dist(pred, tgt)
    margin = float(cfg["chain_margin"])
    hinge_start = torch.relu(d_true - _cos_dist(pred_start_swap, tgt) + margin)
    hinge_chain = torch.relu(d_true - _cos_dist(pred_chain_swap, tgt) + margin)
    non_commute = torch.relu(
        float(cfg["chain_comm_margin"]) - _rel_sep(pred, pred_order_swap)
    ) * both_tail.to(pred.dtype)

    total = float(cfg["chain_mse_w"]) * (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    total = total + float(cfg["chain_start_w"]) * (w * hinge_start).sum()
    total = total + float(cfg["chain_swap_w"]) * (w * hinge_chain).sum()
    total = total + float(cfg["chain_comm_w"]) * (w * non_commute).sum()
    if M >= 4:
        total = total + float(cfg["chain_nce_w"]) * _nce(
            pred, tgt, w, cfg["chain_tau"], cfg["chain_dupe_cos"]
        )
    return total


def _imagination_term(cfg, net, cmd, obs, mined, device):
    rb, rj, idx, sv, t0, wraw = mined
    cap = int(cfg["imag_max_examples"])
    M = min(rb.numel(), cap)
    if M < 2:
        return 0.0
    rb, rj, idx, sv, t0, wraw = rb[:M], rj[:M], idx[:M], sv[:M], t0[:M], wraw[:M]
    w = (wraw / wraw.sum().clamp_min(_EPS)).detach().to(cmd.dtype)

    cmd_slots, _, start, tgt, qcmd, ar = _gather_chain(cmd, obs, rb, rj, idx, sv, t0, device)
    slots = cmd_slots.size(1)
    Dd = cmd_slots.size(-1)
    L = 2 * slots + 1
    cpos = 2 * ar
    opos = 2 * ar + 1

    n_used = (slots - t0).clamp(1, slots)
    src = (t0.unsqueeze(1) + ar.unsqueeze(0)).clamp(max=slots - 1)
    cmd_c = torch.gather(cmd_slots, 1, src.unsqueeze(-1).expand(M, slots, Dd))
    valid_c = ar.unsqueeze(0) < n_used.unsqueeze(1)

    perm = torch.roll(torch.arange(M, device=device), 1)
    start_pair = torch.cat([start, start[perm]], dim=0)
    cmd_pair = cmd_c.repeat(2, 1, 1)
    live = valid_c.repeat(2, 1)
    is_start = live & (ar.unsqueeze(0) == 0)
    is_hop = live & (ar.unsqueeze(0) > 0)
    qpos = 2 * n_used.repeat(2)

    tok2 = cmd_slots.new_zeros(2 * M, L, Dd)
    types2 = torch.zeros(2 * M, L, dtype=torch.long, device=device)
    pad2 = torch.ones(2 * M, L, dtype=torch.bool, device=device)

    tok2[:, cpos, :] = cmd_pair * live.unsqueeze(-1).to(tok2.dtype)
    types2[:, cpos] = is_hop.long() * 2
    pad2[:, cpos] = ~live

    tok2[:, opos, :] = start_pair.unsqueeze(1) * is_start.unsqueeze(-1).to(tok2.dtype)
    types2[:, opos] = 1
    pad2[:, opos] = ~is_start

    rows = torch.arange(2 * M, device=device)
    tok2[rows, qpos] = qcmd.repeat(2, 1)
    types2[rows, qpos] = 0
    pad2[rows, qpos] = False

    pred_all, _ = net(tok2, types2, pad2)
    read = _clean(pred_all[rows, qpos, :])
    pred = read[:M]
    pred_start_swap = read[M:]

    d_true = _cos_dist(pred, tgt)
    hinge = torch.relu(d_true - _cos_dist(pred_start_swap, tgt) + float(cfg["imag_margin"]))
    total = (w * hinge).sum()
    total = total + float(cfg["imag_mse_w"]) * (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    if M >= 4:
        total = total + _nce(pred, tgt, w, cfg["imag_tau"], cfg["chain_dupe_cos"])
    return total


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    step = cfg["_step"]

    total = 0.0
    ramp = _smoothstep(step / max(1.0, float(cfg["ramp_steps"])))
    if ramp > 0.0 and float(cfg.get("aux_weight", 0.0)) > 0.0:
        total = total + _transition_term(cfg, batch, net, device, ramp)

    chain_ramp = _smoothstep(
        (step - float(cfg["chain_ramp_start"])) / max(1.0, float(cfg["chain_ramp_steps"]))
    )
    imag_ramp = _smoothstep(
        (step - float(cfg["imag_ramp_start"])) / max(1.0, float(cfg["imag_ramp_steps"]))
    )
    want_chain = chain_ramp > 0.0 and float(cfg["chain_weight"]) > 0.0
    want_imag = (
        imag_ramp > 0.0
        and float(cfg["imag_weight"]) > 0.0
        and not cfg.get("_imag_disabled", True)
        and step % max(1, int(cfg["imag_every"])) == 0
    )

    if want_chain or want_imag:
        tok = batch["tok"]
        maxn = batch["cmd_mask"].shape[1]
        if maxn >= 3 and tok.shape[0] >= 2:
            cmd = tok[:, 0::2][:, :maxn].detach()
            obs = tok[:, 1::2][:, :maxn].detach()
            mined = _mine_chains(
                cmd,
                obs,
                batch["cmd_mask"].bool(),
                float(cfg["chain_thresh"]),
                float(cfg["chain_change_floor"]),
                int(cfg["chain_hops"]) + 1,
                int(cfg["chain_min_hops"]) + 1,
            )
            if mined[0].numel() >= 2:
                if want_chain:
                    total = total + float(cfg["chain_weight"]) * chain_ramp * _chain_term(
                        cfg, net, cmd, obs, mined, device
                    )
                if want_imag:
                    try:
                        term = _imagination_term(cfg, net, cmd, obs, mined, device)
                    except Exception:
                        cfg["_imag_disabled"] = True
                        term = 0.0
                    total = total + float(cfg["imag_weight"]) * imag_ramp * term

    if torch.is_tensor(total):
        if not bool(torch.isfinite(total).item()):
            return 0.0
    return total


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        v = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(x) for x in v.values()):
        return False
    return all([
        0.0 < v["row_frac"] <= 1.0,
        -1.0 <= v["path_thresh"] < 1.0,
        v["change_floor"] >= 0.0,
        v["max_examples"] >= 1.0,
        v["cos_weight"] >= 0.0,
        v["mse_weight"] >= 0.0,
        v["aux_weight"] >= 0.0,
        v["ramp_steps"] >= 1.0,
        v["chain_hops"] >= 2.0,
        v["chain_min_hops"] >= 1.0,
        v["chain_min_hops"] <= v["chain_hops"],
        -1.0 <= v["chain_thresh"] < 1.0,
        v["chain_change_floor"] >= 0.0,
        v["chain_max_examples"] >= 2.0,
        v["chain_tau"] > 0.0,
        -1.0 <= v["chain_dupe_cos"] <= 1.0,
        v["chain_margin"] >= 0.0,
        v["chain_comm_margin"] >= 0.0,
        -1.0 <= v["chain_comm_max_cos"] <= 1.0,
        v["chain_nce_w"] >= 0.0,
        v["chain_mse_w"] >= 0.0,
        v["chain_start_w"] >= 0.0,
        v["chain_swap_w"] >= 0.0,
        v["chain_comm_w"] >= 0.0,
        v["chain_weight"] >= 0.0,
        v["chain_ramp_start"] >= 0.0,
        v["chain_ramp_steps"] >= 1.0,
        v["imag_max_examples"] >= 2.0,
        v["imag_tau"] > 0.0,
        v["imag_margin"] >= 0.0,
        v["imag_mse_w"] >= 0.0,
        v["imag_weight"] >= 0.0,
        v["imag_every"] >= 1.0,
        v["imag_ramp_start"] >= 0.0,
        v["imag_ramp_steps"] >= 1.0,
    ])
