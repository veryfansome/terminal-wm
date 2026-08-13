import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r3_rename_registry_occupancy_routing"
DESCRIPTION = (
    "A slot registry in which a move RE-ADDRESSES a slot instead of copying its value: each slot "
    "holds (address key, content vector, fill scalar); every command embedding yields two candidate "
    "address keys, and which one acts as source and which as destination is decided at run time by "
    "the current memory OCCUPANCY under each key rather than by a fixed extractor identity. A move "
    "step rotates the source slot's address toward the destination key and leaves its content byte-"
    "identical; a non-move step writes the step's observation into the slot chosen by address "
    "affinity and freeness. The registry read at each command position enters the arch's own "
    "prediction as a residual through a zero-initialised per-dimension scale, so the wrapped forward "
    "equals the unwrapped forward at initialisation. Train-time aux: (a) the registry read at a "
    "position is trained toward that position's true next observation with a squared-L2 InfoNCE over "
    "the sequence's other observations plus an MSE anchor, weighted up at positions mined as the "
    "later end of a BRIDGE in the command-similarity graph (a command strongly similar to one "
    "earlier and one later command that are themselves dissimilar) and weighted down at the bridge "
    "commands themselves; (b) a mixture-of-experts style routing regulariser that makes the address "
    "attention peaked per step while uniform on average over the batch; (c) a content-preservation "
    "consistency term on the arch's shared latent-transition operator at mined bridge commands, "
    "skipped on archs without that operator. Eval forward runs the registry; no aux runs at eval."
)

_DEFAULTS = {
    "slots": 16,
    "key_d": 64,
    "hid": 192,
    "addr_tau": 0.25,
    "route_tau": 0.2,
    "free_w": 2.0,
    "nce_temp": 0.5,
    "hi_z": 1.0,
    "lo_z": 0.25,
    "change_floor": 0.5,
    "dup_frac": 0.05,
    "base_w": 0.25,
    "bonus_w": 1.0,
    "max_examples": 256,
    "nce_weight": 0.6,
    "mse_weight": 0.05,
    "ent_weight": 0.02,
    "load_weight": 0.02,
    "trans_weight": 0.05,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


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
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


class _RenameRegistry(nn.Module):
    def __init__(self, d_model, slots, key_d, hid, addr_tau, route_tau, free_w):
        super().__init__()
        self.d_model = int(d_model)
        self.n_slots = max(2, int(slots))
        self.key_d = max(4, int(key_d))
        h = max(16, int(hid))
        self.free_w = float(free_w)
        self.route_tau = max(1e-2, float(route_tau))

        self.cmd_enc = nn.Linear(self.d_model, h)
        self.obs_enc = nn.Linear(self.d_model, h)
        self.addr_a = nn.Linear(h, self.key_d, bias=False)
        self.addr_b = nn.Linear(h, self.key_d, bias=False)
        self.slot_init = nn.Parameter(torch.randn(self.n_slots, self.key_d) / math.sqrt(self.key_d))
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(1e-3, float(addr_tau)))))

        self.move_gate = nn.Linear(2 * h + 1, 1)
        self.write_gate = nn.Linear(2 * h + 1, 1)
        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.move_gate.bias, -1.0)
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.read_gate.bias, 0.0)

        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def run(self, tok, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, _ = tok.shape
        n = L // 2
        if n < 1:
            return None
        dev = tok.device
        dt = tok.dtype
        S = self.n_slots

        c = torch.nan_to_num(tok[:, 0:2 * n:2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        o = torch.nan_to_num(tok[:, 1:2 * n:2, :], nan=0.0, posinf=1e4, neginf=-1e4)

        if key_pad is None:
            vc = torch.ones(B, n, dtype=torch.bool, device=dev)
            vo = vc
        else:
            v = ~key_pad.bool()
            vc = v[:, 0:2 * n:2]
            vo = v[:, 1:2 * n:2]
        rd_active = vc.to(dt)
        wr_active = (vc & vo).to(dt)

        fc = F.gelu(self.cmd_enc(c))
        fo = F.gelu(self.obs_enc(o))
        ka = _unit(self.addr_a(fc))
        kb = _unit(self.addr_b(fc))
        rgate = torch.sigmoid(self.read_gate(fc))

        tau = self.log_tau.exp().clamp(0.05, 2.0)

        addr = _unit(self.slot_init).unsqueeze(0).expand(B, S, self.key_d).contiguous().to(dt)
        cont = tok.new_zeros(B, S, self.d_model)
        fill = tok.new_zeros(B, S)

        reads = []
        routes = []
        for i in range(n):
            kai = ka[:, i, :]
            kbi = kb[:, i, :]
            aa = torch.softmax(torch.bmm(addr, kai.unsqueeze(2)).squeeze(2) / tau, dim=1)
            ab = torch.softmax(torch.bmm(addr, kbi.unsqueeze(2)).squeeze(2) / tau, dim=1)
            occ_a = (aa * fill).sum(dim=1, keepdim=True)
            occ_b = (ab * fill).sum(dim=1, keepdim=True)
            p = torch.sigmoid((occ_a - occ_b) / self.route_tau)

            w_src = p * aa + (1.0 - p) * ab
            k_src = _unit(p * kai + (1.0 - p) * kbi)
            k_dst = _unit((1.0 - p) * kai + p * kbi)

            r_i = torch.bmm(w_src.unsqueeze(1), cont).squeeze(1)
            reads.append(r_i)
            routes.append(w_src)

            occ_src = (w_src * fill).sum(dim=1, keepdim=True)
            gin = torch.cat([fc[:, i, :], fo[:, i, :], occ_src], dim=-1)
            m_i = torch.sigmoid(self.move_gate(gin)) * wr_active[:, i:i + 1]
            u_i = torch.sigmoid(self.write_gate(gin)) * (1.0 - m_i) * wr_active[:, i:i + 1]

            l_alloc = torch.bmm(addr, k_src.unsqueeze(2)).squeeze(2) / tau + self.free_w * (1.0 - fill)
            w_alloc = torch.softmax(l_alloc, dim=1)

            mv_w = (m_i * w_src).unsqueeze(2)
            al_w = (u_i * w_alloc).unsqueeze(2)
            addr = addr + mv_w * (k_dst.unsqueeze(1) - addr) + al_w * (k_src.unsqueeze(1) - addr)
            addr = _unit(torch.nan_to_num(addr, nan=0.0, posinf=1e2, neginf=-1e2))
            cont = cont + al_w * (o[:, i, :].unsqueeze(1) - cont)
            cont = torch.nan_to_num(cont, nan=0.0, posinf=1e4, neginf=-1e4)
            fill = fill + (u_i * w_alloc) * (1.0 - fill)

        reads_t = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        routes_t = torch.stack(routes, dim=1)
        contrib = rgate * reads_t * self.out_scale.view(1, 1, -1) * rd_active.unsqueeze(-1)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)
        return {"reads": reads_t, "routes": routes_t, "contrib": contrib, "valid": vc & vo}


def wrap(net, D, **params):
    existing = getattr(net, "_rename_registry_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _RenameRegistry(int(D), cfg["slots"], cfg["key_d"], cfg["hid"],
                          cfg["addr_tau"], cfg["route_tau"], cfg["free_w"])
    net.rename_registry = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "layout": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"]:
            return pred, h
        if tok_emb.dim() != 3 or tok_emb.size(1) < 2:
            return pred, h
        out = mod.run(tok_emb, key_pad)
        if out is None:
            return pred, h
        contrib = out["contrib"]
        n = min(contrib.size(1), pred.size(1) // 2)
        if n < 1:
            return pred, h
        add = torch.zeros_like(pred)
        add[:, 0:2 * n:2, :] = contrib[:, :n, :]
        merged = pred + add
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": out["reads"],
                              "routes": out["routes"], "valid": out["valid"]}
        return merged, h

    net.forward = _forward
    net._rename_registry_state = state
    return state


@torch.no_grad()
def _mine_bridges(c, o, valid, hi_z, lo_z, change_floor):
    B, n, Dm = c.shape
    dev = c.device
    dtp = c.dtype
    cu = _unit(c)
    sim = torch.bmm(cu, cu.transpose(1, 2))

    pos = torch.arange(n, device=dev)
    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    vv = valid.unsqueeze(2) & valid.unsqueeze(1)
    off = vv & (~eye)
    cnt = off.sum(dim=(1, 2)).clamp_min(1).to(dtp)
    offf = off.to(dtp)
    mean_s = (sim * offf).sum(dim=(1, 2)) / cnt
    var_s = (((sim - mean_s.view(B, 1, 1)) ** 2) * offf).sum(dim=(1, 2)) / cnt
    std_s = var_s.clamp_min(1e-8).sqrt()
    hi = (mean_s + float(hi_z) * std_s).view(B, 1)
    lo = (mean_s + float(lo_z) * std_s).view(B, 1)

    before = off & (pos.view(1, 1, n) < pos.view(1, n, 1))
    after = off & (pos.view(1, 1, n) > pos.view(1, n, 1))
    neg = torch.finfo(dtp).min
    sim_ik, i_star = sim.masked_fill(~before, neg).max(dim=2)
    sim_kj, j_star = sim.masked_fill(~after, neg).max(dim=2)
    has_i = before.any(dim=2)
    has_j = after.any(dim=2)

    idx_i = i_star.unsqueeze(2).expand(B, n, n)
    sim_rows_i = sim.gather(1, idx_i)
    sim_ij = sim_rows_i.gather(2, j_star.unsqueeze(2)).squeeze(2)

    osq = o.pow(2).sum(-1)
    d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
           - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / float(Dm)
    ref_o = ((d2o * offf).sum(dim=(1, 2)) / cnt).clamp_min(1e-8)
    d_rows_i = d2o.gather(1, idx_i)
    d_ik = d_rows_i.gather(2, pos.view(1, n, 1).expand(B, n, 1)).squeeze(2)

    keep = (has_i & has_j & valid
            & (sim_ik >= hi) & (sim_kj >= hi) & (sim_ij <= lo)
            & (d_ik >= float(change_floor) * ref_o.view(B, 1)))
    return keep, i_star, j_star, osq, ref_o


def aux_loss(head_state, batch, net, device):
    st = head_state
    if st is None:
        return 0.0
    cfg = st["cfg"]
    stash = st.get("stash")
    st["stash"] = None

    if float(cfg["aux_weight"]) <= 0.0:
        return 0.0
    if stash is None:
        return 0.0
    layout = st.get("layout")
    if layout is None:
        layout = bool(_layout_ok(batch))
        st["layout"] = layout
    if not layout:
        return 0.0

    tok = batch["tok"]
    if stash["tok"] is not tok:
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    reads = stash["reads"]
    routes = stash["routes"]
    n = reads.size(1)
    if n < 3:
        return 0.0

    B = tok.size(0)
    Dm = tok.size(-1)
    dev = tok.device
    c = torch.nan_to_num(tok[:, 0:2 * n:2, :], nan=0.0, posinf=1e4, neginf=-1e4)
    o = torch.nan_to_num(tok[:, 1:2 * n:2, :], nan=0.0, posinf=1e4, neginf=-1e4)
    valid = stash["valid"][:, :n] & batch["cmd_mask"][:, :n].bool()
    if not bool(valid.any().item()):
        return 0.0

    keep, i_star, j_star, osq, ref_o = _mine_bridges(
        c, o, valid, cfg["hi_z"], cfg["lo_z"], cfg["change_floor"])

    with torch.no_grad():
        dest_f = torch.zeros(B, n, device=dev, dtype=torch.float32)
        dest_f.scatter_add_(1, j_star.clamp(0, n - 1), keep.float())
        dest = dest_f > 0.5
        w = (float(cfg["base_w"]) * (valid & (~keep)).float()
             + float(cfg["bonus_w"]) * (dest & valid).float())
        w[:, 0] = 0.0
        sel = torch.nonzero(w > 0.0, as_tuple=False)

    total = reads.sum() * 0.0

    if sel.numel() > 0:
        b_idx = sel[:, 0]
        j_idx = sel[:, 1]
        w_sel = w[b_idx, j_idx]
        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            with torch.no_grad():
                jitter = torch.rand_like(w_sel) * 1e-3
                _, order = torch.topk(w_sel + jitter, cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]
            w_sel = w_sel[order]
        wn = (w_sel / w_sel.sum().clamp_min(_EPS)).detach().to(reads.dtype)

        cand = o[b_idx]
        cand_sq = osq[b_idx]
        cand_valid = valid[b_idx]
        tgt = o[b_idx, j_idx]
        r_sel = reads[b_idx, j_idx]

        with torch.no_grad():
            d2_tj = (cand_sq + tgt.pow(2).sum(-1, keepdim=True)
                     - 2.0 * torch.bmm(cand, tgt.unsqueeze(2)).squeeze(2)).clamp_min(0.0) / float(Dm)
            dup = d2_tj <= (float(cfg["dup_frac"]) * ref_o[b_idx]).unsqueeze(1)
            is_j = F.one_hot(j_idx, n).bool()
            cand_mask = cand_valid & ((~dup) | is_j)

        d2 = (cand_sq + r_sel.pow(2).sum(-1, keepdim=True)
              - 2.0 * torch.bmm(cand, r_sel.unsqueeze(2)).squeeze(2)).clamp_min(0.0) / float(Dm)
        logits = (-d2 / max(1e-3, float(cfg["nce_temp"]))).masked_fill(~cand_mask, _NEG)
        ce = F.cross_entropy(logits, j_idx, reduction="none")
        nce = (wn * ce).sum()
        mse = (wn * (r_sel - tgt).pow(2).mean(dim=-1)).sum()
        total = total + float(cfg["nce_weight"]) * nce + float(cfg["mse_weight"]) * mse

    if float(cfg["ent_weight"]) > 0.0 or float(cfg["load_weight"]) > 0.0:
        vm = valid.to(routes.dtype).unsqueeze(-1)
        denom = vm.sum().clamp_min(1.0)
        logr = (routes + 1e-6).log()
        ent = -((routes * logr) * vm).sum() / denom
        share = (routes * vm).sum(dim=(0, 1)) / denom
        share = share / share.sum().clamp_min(_EPS)
        n_slots = share.numel()
        load = (share * (share * float(n_slots) + 1e-6).log()).sum()
        total = total + float(cfg["ent_weight"]) * ent + float(cfg["load_weight"]) * load

    op = getattr(net, "transition_from_emb", None)
    if callable(op) and float(cfg["trans_weight"]) > 0.0:
        nzk = torch.nonzero(keep, as_tuple=False)
        if nzk.numel() > 0:
            kb_idx = nzk[:, 0]
            kk_idx = nzk[:, 1]
            cap = int(cfg["max_examples"])
            if kb_idx.numel() > cap:
                kb_idx = kb_idx[:cap]
                kk_idx = kk_idx[:cap]
            src = o[kb_idx, i_star[kb_idx, kk_idx]].detach()
            cmd_k = c[kb_idx, kk_idx].detach()
            moved = torch.nan_to_num(op(src, cmd_k), nan=0.0, posinf=1e4, neginf=-1e4)
            cos_err = (1.0 - (_unit(moved) * _unit(src)).sum(dim=-1).clamp(-1.0, 1.0)).mean()
            mse_err = (moved - src).pow(2).mean()
            total = total + float(cfg["trans_weight"]) * (cos_err + 0.1 * mse_err)

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
        vals["slots"] >= 2.0,
        vals["key_d"] >= 4.0,
        vals["hid"] >= 16.0,
        vals["addr_tau"] > 0.0,
        vals["route_tau"] > 0.0,
        vals["free_w"] >= 0.0,
        vals["nce_temp"] > 0.0,
        vals["hi_z"] >= vals["lo_z"],
        vals["change_floor"] >= 0.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["base_w"] >= 0.0,
        vals["bonus_w"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["ent_weight"] >= 0.0,
        vals["load_weight"] >= 0.0,
        vals["trans_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
