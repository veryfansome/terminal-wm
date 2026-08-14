TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

THE CONTRACT — axis 'head': Expose wrap(net, D, **params) -> a head state or None, aux_loss(head_state, batch, net, device) -> a scalar or zero, and leak_safe(mod, params) -> bool. wrap runs BEFORE the optimizer is built, so registered readout and auxiliary parameters are optimized. aux_loss is train-time only. Two hazards a wrapper must avoid: a parent-child module cycle (hold the base net unregistered, or moving to device recurses), and forward recursion when re-pointing forward (save the original bound method first). A head that recomputes the prediction from the trunk hidden state silently bypasses any architecture whose prediction is not a per-position function of that state.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any head impl:
  wrap(net, D, **params) -> a head state, or None for no head
      Called BEFORE the optimizer is built, so any readout or auxiliary parameters it
      registers on net are optimized. Two hazards a wrapper must avoid: a parent-child
      module cycle (hold the base net unregistered, or moving to device recurses), and
      forward recursion when re-pointing net.forward (save the original bound method first).
  aux_loss(head_state, batch, net, device) -> scalar tensor or 0.0; train-time only.
  leak_safe(mod, params) -> bool; asserted before scoring.
"""

import torch

NAME_BASELINE = "baseline_passthrough"
DESCRIPTION_BASELINE = ("Arch's own Linear readout, unchanged; no aux loss. "
                        "Bit-identical to the pre-axis harness readout.")


def wrap(net, D, **params):
    return None


# A hard 0.0 (not a zero tensor) so `main + aux` is main bit-for-bit and archived
# fitnesses replay exactly.
def aux_loss(head_state, batch, net, device):
    return 0.0


def leak_safe(mod, params):
    return True


NAME = NAME_BASELINE
DESCRIPTION = DESCRIPTION_BASELINE
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                r5-05-shared-pathid-signed
  its fitness       +0.0187   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           whitened_metric_ring_mse
  arch                r19_obspresent_imagination_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ridge_zca_whitened_obs
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              r5_shared_pathid_signed_role_binding
  head                r3_rename_registry_occupancy_routing

YOUR PARENT'S CURRENT head IMPL — r3_rename_registry_occupancy_routing (this is the code you are mutating):
--------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r20_dualpre_transition_consistency (axis head)
import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CH

NAME = "r20_dualpre_transition_consistency"
DESCRIPTION = (
    "The r18 forward-model-consistency aux VERBATIM (raw-obs pre arm, weight 1.0) plus a "
    "mem-pre arm supervising the SAME shared transition operator on the arch's own memory "
    "content s_pre_k (computed via the arch's input block + _transition_reads under no_grad, "
    "detached) toward the same mined future reads — training the operator on its deployment "
    "distribution inside the single pass. One mining pass, RNG-draw count "
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

--- r18_transition_forwardmodel_consistency (axis head)
import math

import torch
import torch.nn as nn

NAME = "r18_transition_forwardmodel_consistency"
DESCRIPTION = (
    "Train-only forward-model consistency on the r18 latent-transition arch's shared operator: "
    "mines same-path (pre, mutating-cmd, future-read) triples and requires f(obs_pre, cmd) to "
    "reconstruct the future post-mutation observation (cosine+MSE, change-weighted). Eval forward "
    "untouched; disabled (0.0) on archs without the transition operator. Co-designed head half of "
    "the r18 transition world-model stack."
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


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
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


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
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
    ]
    return all(checks)

--- r5_rawaddress_dualpre_consistency (axis head)
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

--- r2_dualaddress_move_transport (axis head)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r2_dualaddress_move_transport"
DESCRIPTION = (
    "Adds a dual-address transport memory on top of the arch's own prediction: every command "
    "embedding is mapped through two extractors and one SHARED path-to-key encoder into a source "
    "key and a destination key, and a delta-rule outer-product memory is updated causally by "
    "writing, at the destination key, a gated blend of the value currently read at the source key "
    "and the step's observation. The memory read at each command position is injected into the "
    "arch's prediction through a per-dimension scale initialised to zero, so the wrapped net is "
    "the unwrapped net at initialisation. Train-time aux mines, label-free and from the batch "
    "alone, steps whose observation exactly duplicates an earlier step's observation under a "
    "different command with at least one content-free step in between, and trains the memory read "
    "and the final prediction with a within-sequence squared-L2 InfoNCE against the other "
    "observations seen so far, plus a small MSE anchor."
)

_DEFAULTS = {
    "key_d": 48,
    "hid": 256,
    "cand_temp": 0.5,
    "dup_frac": 0.05,
    "cmd_dup_frac": 0.01,
    "max_dup": 3,
    "max_examples": 192,
    "read_weight": 1.0,
    "pred_weight": 0.5,
    "mse_weight": 0.05,
    "aux_weight": 1.0,
    "ramp_steps": 300,
    "sym_init_noise": 0.02,
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
    if cmd_mask.shape != tgt.shape[:2] or tok.shape[0] != tgt.shape[0]:
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


class _DualAddressTransport(nn.Module):
    def __init__(self, d_model, key_d, hid, sym_init_noise):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = int(key_d)
        h = int(hid)

        self.src_ext = nn.Linear(self.d_model, h)
        self.dst_ext = nn.Linear(self.d_model, h)
        with torch.no_grad():
            noise = torch.randn_like(self.src_ext.weight) * float(sym_init_noise)
            self.dst_ext.weight.copy_(self.src_ext.weight + noise)
            self.dst_ext.bias.copy_(self.src_ext.bias)

        self.path_key = nn.Linear(h, self.key_d, bias=False)

        self.cmd_ctx = nn.Linear(self.d_model, h)
        self.obs_ctx = nn.Linear(self.d_model, h)
        self.gate_mix = nn.Linear(2 * h, h)
        self.move_gate = nn.Linear(h, 1)
        self.write_gate = nn.Linear(h, 1)
        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.move_gate.bias, 0.0)
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.read_gate.bias, 0.0)

        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def run(self, tok, types, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None, None
        B, L, Dm = tok.shape
        n = L // 2
        if n < 1:
            return None, None
        dev = tok.device
        dt = tok.dtype

        c = torch.nan_to_num(tok[:, 0::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4)
        o = torch.nan_to_num(tok[:, 1::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4)

        if key_pad is None:
            vc = torch.ones(B, n, dtype=torch.bool, device=dev)
            vo = vc
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2][:, :n]
            vo = v[:, 1::2][:, :n]
        active = (vc & vo).to(dt).unsqueeze(-1)

        cf = F.gelu(self.cmd_ctx(c))
        of = F.gelu(self.obs_ctx(o))
        gf = F.gelu(self.gate_mix(torch.cat([cf, of], dim=-1)))
        move = torch.sigmoid(self.move_gate(gf))
        beta = torch.sigmoid(self.write_gate(gf))
        rgate = torch.sigmoid(self.read_gate(cf))

        ks = _unit(self.path_key(F.gelu(self.src_ext(c))))
        kd = _unit(self.path_key(F.gelu(self.dst_ext(c))))

        mem = tok.new_zeros(B, self.key_d, self.d_model)
        reads = []
        for i in range(n):
            ksi = ks[:, i, :].unsqueeze(1)
            kdi = kd[:, i, :].unsqueeze(1)
            r_i = torch.bmm(ksi, mem).squeeze(1)
            reads.append(r_i)
            cur_d = torch.bmm(kdi, mem).squeeze(1)
            m_i = move[:, i, :]
            v_i = m_i * r_i + (1.0 - m_i) * o[:, i, :]
            w_i = (v_i - cur_d) * beta[:, i, :] * active[:, i, :]
            w_i = torch.nan_to_num(w_i, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            mem = torch.baddbmm(mem, kdi.transpose(1, 2), w_i.unsqueeze(1))

        reads_t = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        contrib = rgate * (reads_t * self.out_scale.view(1, 1, -1))
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)
        return reads_t, contrib


def wrap(net, D, **params):
    if getattr(net, "_dualaddr_transport_state", None) is not None:
        return net._dualaddr_transport_state

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _DualAddressTransport(int(D), int(cfg["key_d"]), int(cfg["hid"]),
                                float(cfg["sym_init_noise"]))
    net.dualaddr_transport = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 2:
            return pred, h
        reads_t, contrib = mod.run(tok_emb, types, key_pad)
        if contrib is None:
            return pred, h
        n = contrib.size(1)
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        cmd_view[:, :n, :] = cmd_view[:, :n, :] + contrib
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": reads_t, "pred": out}
        return out, h

    net.forward = _forward
    net._dualaddr_transport_state = state
    return state


def aux_loss(head_state, batch, net, device):
    st = head_state
    if st is None:
        return 0.0
    cfg = st["cfg"]
    mod = st["mod"]
    stash = st.get("stash")
    st["stash"] = None

    if float(cfg["aux_weight"]) <= 0.0:
        return 0.0
    if not _layout_ok(batch):
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    B, n = valid.shape
    if n < 3:
        return 0.0

    if stash is not None and stash["tok"] is tok:
        reads_t = stash["reads"]
        pred_cmd = stash["pred"][:, 0::2, :][:, :n, :]
    else:
        reads_t, _ = mod.run(tok, batch["types"], batch["key_pad"])
        pred_cmd = None
    if reads_t is None:
        return 0.0
    reads_t = reads_t[:, :n, :].float()

    c = torch.nan_to_num(tok[:, 0::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    o = torch.nan_to_num(tok[:, 1::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    Dm = float(o.size(-1))
    dev = o.device

    max_dup = int(cfg["max_dup"])
    mut_min = max_dup + 1

    with torch.no_grad():
        osq = o.pow(2).sum(-1)
        csq = c.pow(2).sum(-1)
        d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
               - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / Dm
        d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
               - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / Dm

        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
        ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
        ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

        dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        clus = dup_o.sum(dim=2)
        contentful = valid & (clus <= max_dup)
        mutation = valid & (clus >= mut_min)
        mcs = torch.cumsum(mutation.long(), dim=1)

        pair = dup_o & (~dup_c) & contentful.unsqueeze(1) & contentful.unsqueeze(2)
        tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=-1).unsqueeze(0)
        earlier = pair & tri
        has_p = earlier.any(dim=2)

        idxn = torch.arange(n, device=dev)
        big = torch.full((1, 1, n), n, device=dev, dtype=torch.long)
        small = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)
        first_i = torch.where(earlier, idxn.view(1, 1, n), big).amin(dim=2)
        last_i = torch.where(earlier, idxn.view(1, 1, n), small).amax(dim=2)

        jm1 = (idxn.view(1, n) - 1).clamp_min(0).expand(B, n)
        mc_j = torch.gather(mcs, 1, jm1)
        mc_i = torch.gather(mcs, 1, last_i.clamp_min(0))
        mut_between = (mc_j - mc_i) > 0

        mined = has_p & contentful & mut_between & valid
        nz = torch.nonzero(mined, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        b_idx = nz[:, 0]
        j_idx = nz[:, 1]
        depth = (idxn.view(1, n) - first_i).clamp_min(0)
        dep = depth[b_idx, j_idx].float()
        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            _, order = torch.topk(dep, cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]

        seen = idxn.view(1, n) <= j_idx.view(-1, 1)
        cand_mask = contentful[b_idx] & seen
        pos_mask = (dup_o[b_idx, j_idx] | F.one_hot(j_idx, n).bool()) & cand_mask
        cand_sq = osq[b_idx]

    cand = o[b_idx]
    tgt_o = o[b_idx, j_idx]
    temp = max(1e-3, float(cfg["cand_temp"]))

    def _nce(q):
        dot = torch.bmm(cand, q.unsqueeze(2)).squeeze(2)
        d2 = (q.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / Dm
        logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
        logp = torch.log_softmax(logits, dim=1)
        return -(torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)).mean()

    r_sel = reads_t[b_idx, j_idx]
    total = float(cfg["read_weight"]) * _nce(r_sel)
    total = total + float(cfg["mse_weight"]) * (r_sel - tgt_o).pow(2).mean()
    if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
        total = total + float(cfg["pred_weight"]) * _nce(pred_cmd[b_idx, j_idx].float())

    out_loss = ramp * float(cfg["aux_weight"]) * total
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
        vals["key_d"] >= 1.0,
        vals["hid"] >= 8.0,
        vals["cand_temp"] > 0.0,
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["sym_init_noise"] >= 0.0,
    ]
    return all(checks)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0187 (r5-05-shared-pathid-signed, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
