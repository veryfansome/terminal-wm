TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

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
  id                r5-18-relsam-spectral-transport
  its fitness       +0.0037   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r5_relsam_spectral_transport_keys
  target              identity
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r5_dualextractor_displacement_transport

YOUR PARENT'S CURRENT head IMPL — r5_dualextractor_displacement_transport (this is the code you are mutating):
--------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r5_dualextractor_displacement_transport"
DESCRIPTION = (
    "A path-slot transport memory whose DESTINATION key is the SOURCE key plus a zero-initialised "
    "displacement, where the displacement is produced by a SECOND command extractor initialised as "
    "a symmetry-broken copy of the source extractor, while the source key itself comes from one "
    "shared path-to-key encoder. Destination equals source exactly at initialisation, so every "
    "command is an identity slot until the displacement grows, and the destination argument of a "
    "move gets its own extraction pathway instead of sharing the source's feature vector. A causal "
    "scan reads the slot addressed by the source key BEFORE writing, writes into the destination "
    "slot by the delta rule a move-gated blend of that read and the step's own observation, and "
    "erases the source slot in proportion to the move gate; the read is added to the arch's own "
    "prediction through a zero-init per-dimension gain and a command-only read gate, so the wrapped "
    "net is the unwrapped net at initialisation and no prediction depends on its own step's "
    "observation. The scan covers every command token, including a trailing command that has no "
    "paired observation token, which is read-only and writes nothing. A train-only aux mines, from "
    "embedding structure alone, steps whose observation duplicates an earlier step's under a "
    "different command with one or more content-free steps in between, and (a) when exactly one "
    "content-free step lies between them, trains that step's source key to retrieve the earlier "
    "step's location key and its destination key to retrieve the later step's, in one softmax over "
    "the sequence's own read commands, (b) trains the memory read and the injected prediction at "
    "the later step, through however many content-free steps intervene, to select the earlier "
    "observation among the sequence's observations, and (c) ties the two keys together on "
    "single-path read commands. The mined set is capped by a two-pool rule: a reserved quota of "
    "one-hop pairs ranked by duplicate exactness, the remainder ranked by how many content-free "
    "steps they span. Carries the forward-model consistency aux on the arch's shared transition "
    "operator with raw-observation and memory-content pre-state arms, auto-disabled on archs "
    "without that surface."
)

_DEFAULTS = {
    "key_d": 64,
    "hid": 256,
    "blank_min": 3,
    "dup_frac": 0.02,
    "cmd_dup_frac": 0.01,
    "max_hops": 8,
    "max_examples": 256,
    "key_temp": 0.1,
    "cand_temp": 0.5,
    "align_weight": 0.75,
    "read_weight": 0.75,
    "pred_weight": 0.5,
    "sym_weight": 0.1,
    "mse_weight": 0.05,
    "trans_weight": 1.0,
    "trans_cos": 0.10,
    "trans_mse": 0.02,
    "trans_row_frac": 0.6,
    "trans_path_thresh": 0.60,
    "trans_change_floor": 0.25,
    "trans_max_examples": 512,
    "mem_arm_w": 0.5,
    "aux_weight": 0.5,
    "ramp_steps": 300,
    "sym_init_noise": 0.02,
    "hop1_frac": 0.5,
}

_EPS = 1e-8
_NEG = -1e9

_MEM_ATTRS = ("cmd_proj", "obs_proj", "type_emb", "in_norm", "_positional", "_transition_reads")


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


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


def _pad_to(x, n, value=0.0):
    cur = x.size(1)
    if cur >= n:
        return x[:, :n]
    shape = (x.size(0), n - cur) + tuple(x.shape[2:])
    if x.dtype == torch.bool:
        pad = torch.full(shape, bool(value), dtype=torch.bool, device=x.device)
    else:
        pad = torch.full(shape, float(value), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


class _DualExtractorDisplacementTransport(nn.Module):
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
        self.displace = nn.Linear(h, self.key_d, bias=False)
        nn.init.zeros_(self.displace.weight)
        self.obs_in = nn.Linear(self.d_model, h)
        self.gate_mix = nn.Linear(2 * h, h)
        self.move_gate = nn.Linear(h, 1)
        self.write_gate = nn.Linear(h, 1)
        self.erase_gate = nn.Linear(h, 1)
        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.move_gate.bias, 0.0)
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.erase_gate.bias, 0.0)
        nn.init.constant_(self.read_gate.bias, 0.0)
        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def address(self, cmd):
        c = _clean(cmd)
        fs = F.gelu(self.src_ext(c))
        fd = F.gelu(self.dst_ext(c))
        base = self.path_key(fs)
        return _unit(base), _unit(base + self.displace(fd)), fs

    def run(self, tok, types, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None, None
        B, L, _ = tok.shape
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_cmd < 1:
            return None, None
        dt = tok.dtype
        c = _clean(tok[:, 0::2, :])[:, :n_cmd, :]
        o = _pad_to(_clean(tok[:, 1::2, :])[:, :n_pair, :], n_cmd)
        if key_pad is None:
            vc = torch.ones(B, n_cmd, dtype=torch.bool, device=tok.device)
            vo = vc
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2][:, :n_cmd]
            vo = _pad_to(v[:, 1::2][:, :n_pair], n_cmd, False)
        ks, kd, cf = self.address(c)
        of = F.gelu(self.obs_in(o))
        gf = F.gelu(self.gate_mix(torch.cat([cf, of], dim=-1)))
        move = torch.sigmoid(self.move_gate(gf))
        beta = torch.sigmoid(self.write_gate(gf))
        erase = torch.sigmoid(self.erase_gate(gf)) * move
        rgate = torch.sigmoid(self.read_gate(cf))
        act = (vc & vo).to(dt).unsqueeze(-1)
        mem = tok.new_zeros(B, self.key_d, self.d_model)
        reads = []
        for i in range(n_cmd):
            ksi = ks[:, i:i + 1, :]
            kdi = kd[:, i:i + 1, :]
            r_i = torch.bmm(ksi, mem).squeeze(1)
            reads.append(r_i)
            cur_d = torch.bmm(kdi, mem).squeeze(1)
            m_i = move[:, i, :]
            v_i = m_i * r_i + (1.0 - m_i) * o[:, i, :]
            w_d = ((v_i - cur_d) * beta[:, i, :] * act[:, i, :]).clamp(-1e3, 1e3)
            w_s = ((-r_i) * erase[:, i, :] * act[:, i, :]).clamp(-1e3, 1e3)
            wk = torch.cat([kdi, ksi], dim=1).transpose(1, 2)
            wv = _clean(torch.stack([w_d, w_s], dim=1))
            mem = torch.baddbmm(mem, wk, wv)
        reads_t = _clean(torch.stack(reads, dim=1))
        contrib = _clean(rgate * reads_t * self.out_scale.view(1, 1, -1))
        contrib = contrib * vc.unsqueeze(-1).to(dt)
        return reads_t, contrib


def wrap(net, D, **params):
    prev = getattr(net, "_dualdisp_state", None)
    if prev is not None:
        return prev

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _DualExtractorDisplacementTransport(int(D), int(cfg["key_d"]), int(cfg["hid"]),
                                              float(cfg["sym_init_noise"]))
    net.dualdisp_transport = mod

    state = {
        "cfg": cfg,
        "mod": mod,
        "step": 0,
        "stash": None,
        "D": int(D),
        "mem_ok": all(hasattr(net, a) for a in _MEM_ATTRS) and hasattr(net, "pos_scale"),
    }
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        out = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if not (isinstance(out, tuple) and len(out) == 2):
            return out
        pred, h = out
        if not torch.is_tensor(pred) or pred.dim() != 3 or pred.size(-1) != state["D"]:
            return out
        if not torch.is_tensor(tok_emb) or tok_emb.dim() != 3 or tok_emb.size(1) < 2:
            return out
        reads_t, contrib = mod.run(tok_emb, types, key_pad)
        if contrib is None:
            return out
        n = min(contrib.size(1), (pred.size(1) + 1) // 2)
        if n < 1:
            return out
        new_pred = pred.clone()
        cmd_view = new_pred[:, 0::2, :]
        cmd_view[:, :n, :] = cmd_view[:, :n, :] + contrib[:, :n, :]
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": reads_t, "pred": new_pred}
        return new_pred, h

    net.forward = _forward
    net._dualdisp_state = state
    return state


@torch.no_grad()
def _mine_path_triples(cmd, obs, valid, path_thresh, change_floor):
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
    return sel_b, i_idx[sel_b, sel_k], sel_k, j_idx[sel_b, sel_k], w[sel_b, sel_k]


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


def _transition_terms(st, net, batch, tok, valid, B, n, device):
    cfg = st["cfg"]
    if float(cfg["trans_weight"]) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["trans_row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :n]
    obs = tok[sel][:, 1::2][:, :n]
    val = valid[sel]

    r, ti, tk, tj, w = _mine_path_triples(
        cmd, obs, val, float(cfg["trans_path_thresh"]), float(cfg["trans_change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    cap = int(cfg["trans_max_examples"])
    if r.numel() > cap:
        w, order = torch.topk(w, cap)
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    cmd_k = _clean(cmd[r, tk].detach())
    tgt = _clean(obs[r, tj].detach())
    gu = _unit(tgt)

    def arm(pre):
        pred = _clean(op(pre, cmd_k))
        pu = _unit(pred)
        cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
        return float(cfg["trans_cos"]) * cos_err + float(cfg["trans_mse"]) * mse_err

    out = arm(_clean(obs[r, ti].detach()))
    mem_w = float(cfg["mem_arm_w"])
    if mem_w > 0.0 and bool(st.get("mem_ok", False)):
        reads = _memory_pre(net, tok[sel], batch["types"][sel], val, device)
        out = out + mem_w * arm(_clean(reads[r, tk].detach()))
    return float(cfg["trans_weight"]) * out


def _transport_terms(st, mod, batch, tok, valid, B, n, stash):
    cfg = st["cfg"]
    c = _clean(tok[:, 0::2, :][:, :n, :]).float()
    o = _clean(tok[:, 1::2, :][:, :n, :]).float()
    dm = float(o.size(-1))
    dev = o.device
    idxn = torch.arange(n, device=dev)

    with torch.no_grad():
        osq = o.pow(2).sum(-1)
        csq = c.pow(2).sum(-1)
        d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
               - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / dm
        d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
               - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / dm

        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
        ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
        ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

        dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        cross = dup_o & (~dup_c)
        blank = valid & (cross.sum(dim=2) >= int(cfg["blank_min"]))
        content = valid & (~blank)

        bc = torch.cumsum(blank.long(), dim=1)
        jm1 = (idxn - 1).clamp_min(0)
        bc_prev = bc[:, jm1]
        marked = torch.where(blank, idxn.view(1, n).expand(B, n),
                             torch.full((B, n), -1, device=dev, dtype=torch.long))
        lastb = torch.cummax(marked, dim=1).values
        k_of_j = lastb[:, jm1]

        hops = bc_prev.unsqueeze(1) - bc.unsqueeze(2)
        tri = idxn.view(1, n, 1) < idxn.view(1, 1, n)
        pair = (cross & tri & content.unsqueeze(2) & content.unsqueeze(1)
                & (hops >= 1) & (hops <= int(cfg["max_hops"])))

        nz = torch.nonzero(pair, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        bsel = nz[:, 0]; isel = nz[:, 1]; jsel = nz[:, 2]
        cap = int(cfg["max_examples"])
        if bsel.numel() > cap:
            d2sel = d2o[bsel, isel, jsel]
            hsel = hops[bsel, isel, jsel].to(d2sel.dtype)
            is_one = hsel == 1.0
            a_idx = torch.nonzero(is_one, as_tuple=False).squeeze(1)
            b_idx = torch.nonzero(~is_one, as_tuple=False).squeeze(1)
            quota = max(1, int(round(float(cfg["hop1_frac"]) * cap)))
            k1 = int(min(a_idx.numel(), cap if b_idx.numel() == 0 else quota))
            parts = []
            if k1 > 0:
                _, o1 = torch.topk(-d2sel[a_idx], k1)
                parts.append(a_idx[o1])
            k2 = int(min(b_idx.numel(), cap - k1))
            if k2 > 0:
                _, o2 = torch.topk(hsel[b_idx] - d2sel[b_idx], k2)
                parts.append(b_idx[o2])
            if not parts:
                return 0.0
            keep = parts[0] if len(parts) == 1 else torch.cat(parts)
            bsel = bsel[keep]; isel = isel[keep]; jsel = jsel[keep]

        e = bsel.numel()
        ksel = k_of_j[bsel, jsel]
        hop1 = (hops[bsel, isel, jsel] == 1) & (ksel > isel)
        seen = idxn.view(1, n) <= jsel.view(e, 1)
        one_i = F.one_hot(isel, n).bool()
        one_j = F.one_hot(jsel, n).bool()
        cand_mask = content[bsel] & seen
        pos_obs = (dup_o[bsel, isel] | one_i) & cand_mask
        key_cand = content[bsel]
        pos_src = (dup_c[bsel, isel] | one_i) & key_cand
        pos_dst = (dup_c[bsel, jsel] | one_j) & key_cand
        cand_sq = osq[bsel]

    pred_cmd = None
    if stash is not None and stash["tok"] is tok:
        reads_t = stash["reads"]
        pred_cmd = stash["pred"][:, 0::2, :][:, :n, :]
    else:
        reads_t, _ = mod.run(tok, batch["types"], batch["key_pad"])
    if reads_t is None:
        return 0.0
    reads_t = reads_t[:, :n, :].float()

    cand = o[bsel]
    temp = max(1e-3, float(cfg["cand_temp"]))

    def obs_nce(q):
        dot = torch.bmm(cand, q.unsqueeze(2)).squeeze(2)
        d2 = (q.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / dm
        logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
        logp = torch.log_softmax(logits, dim=1)
        return -(torch.logsumexp(logp.masked_fill(~pos_obs, _NEG), dim=1)).mean()

    r_sel = reads_t[bsel, jsel]
    total = float(cfg["read_weight"]) * obs_nce(r_sel)
    total = total + float(cfg["mse_weight"]) * (r_sel - o[bsel, isel]).pow(2).mean()
    if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
        total = total + float(cfg["pred_weight"]) * obs_nce(pred_cmd.float()[bsel, jsel])

    ks_all, kd_all, _ = mod.address(c.to(tok.dtype))
    ks_all = ks_all.float()
    kd_all = kd_all.float()

    if float(cfg["sym_weight"]) > 0.0:
        tied = (content & valid).float()
        cos_sd = (ks_all * kd_all).sum(-1)
        total = total + float(cfg["sym_weight"]) * (
            ((1.0 - cos_sd) * tied).sum() / tied.sum().clamp_min(1.0))

    if float(cfg["align_weight"]) > 0.0 and bool(hop1.any().item()):
        rows = torch.nonzero(hop1, as_tuple=False).squeeze(1)
        b1 = bsel[rows]; k1 = ksel[rows]
        kcand = ks_all[b1]
        mcand = key_cand[rows]
        kt = max(1e-3, float(cfg["key_temp"]))

        def key_nce(q, pos):
            logits = (torch.bmm(kcand, q.unsqueeze(2)).squeeze(2) / kt).masked_fill(~mcand, _NEG)
            logp = torch.log_softmax(logits, dim=1)
            return -(torch.logsumexp(logp.masked_fill(~pos, _NEG), dim=1)).mean()

        src_loss = key_nce(ks_all[b1, k1], pos_src[rows])
        dst_loss = key_nce(kd_all[b1, k1], pos_dst[rows])
        total = total + float(cfg["align_weight"]) * (src_loss + dst_loss)

    return total


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

    parts = []
    tp = _transport_terms(st, mod, batch, tok, valid, B, n, stash)
    if torch.is_tensor(tp):
        parts.append(tp.float())
    tr = _transition_terms(st, net, batch, tok, valid, B, n, device)
    if torch.is_tensor(tr):
        parts.append(tr.float())
    if not parts:
        return 0.0

    total = parts[0]
    for extra in parts[1:]:
        total = total + extra
    out = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out).item()):
        return 0.0
    return out.to(tok.dtype)


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    extra = dict(params or {})
    for k in extra:
        if k not in _DEFAULTS:
            return False
    p.update(extra)
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        vals["key_d"] >= 1.0,
        vals["hid"] >= 8.0,
        vals["blank_min"] >= 1.0,
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_hops"] >= 1.0,
        vals["max_examples"] >= 1.0,
        vals["key_temp"] > 0.0,
        vals["cand_temp"] > 0.0,
        vals["align_weight"] >= 0.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["sym_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["trans_weight"] >= 0.0,
        vals["trans_cos"] >= 0.0,
        vals["trans_mse"] >= 0.0,
        0.0 < vals["trans_row_frac"] <= 1.0,
        -1.0 <= vals["trans_path_thresh"] < 1.0,
        vals["trans_change_floor"] >= 0.0,
        vals["trans_max_examples"] >= 1.0,
        0.0 <= vals["mem_arm_w"] <= 10.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["sym_init_noise"] >= 0.0,
        0.0 <= vals["hop1_frac"] <= 1.0,
    ]
    return all(checks)
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r6_backward_erasure_resolved_transport (axis head)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r6_backward_erasure_resolved_transport"
DESCRIPTION = (
    "A backward content resolver over command positions with dual addresses, an explicit erasure "
    "penalty, and an unbounded-depth closed form. Every command's raw embedding passes one shared "
    "GELU trunk and two unit-normalised heads giving a SOURCE address (the location the command "
    "reads from) and a DESTINATION address (the location it writes to). A learned RELAY gate, fed "
    "by the command and its paired observation, decides whether a step merely relays content "
    "(empty-output mutation) or exposes it; the location a step's content ends up at is therefore "
    "the gated blend of its destination and its source address, and that single gate is reused as "
    "the coefficient of the resolution recursion. A read at step i queries its own source address "
    "against the end-location keys of strictly earlier steps, discounted by a learned recency rate "
    "and by an ERASURE term equal to the strongest later relay that read from that same key, so a "
    "location whose content was moved away stops answering. The recursion 'the content at the "
    "location step i reads from is, at the matched step, either that step's observation or the "
    "content that step itself resolved' is one strictly-lower-triangular linear system in the "
    "content variable, solved exactly in a single unit-triangular solve, so resolution depth is "
    "unbounded rather than a fixed hop count and no sequential scan over the sequence is needed. "
    "The query is built from the command embedding alone, so the final command of an odd-length "
    "window, whose observation is the answer and is absent, still issues a full query. The "
    "resolved content enters the arch's own prediction at every command position under a sigmoid "
    "gate through two parallel paths: a zero-initialised (D,D) readout, which stays exactly zero "
    "at build time so a cold-parameter optimizer can open it, and a small non-zero per-dimension "
    "direct scale, so every parameter of the resolver measurably moves the prediction at the "
    "scored read position from the first step. Train-time aux, label-free from the batch alone: "
    "a binary calibration of "
    "the relay gate against steps whose observation falls in an oversized duplicate cluster, and, "
    "on mined reads whose observation duplicates an earlier observation with at least one such "
    "step in between, a listwise ranking of the resolved content and of the head-augmented "
    "prediction against the contentful observations seen so far under the eval's own per-dimension "
    "squared-L2 decision variable."
)

_DEFAULTS = {
    "key_d": 64,
    "addr_hidden": 256,
    "obs_hidden": 128,
    "temp_init": 0.25,
    "recency_init": 0.15,
    "erase_init": 2.0,
    "erase_sharp_init": 8.0,
    "erase_thresh_init": 0.5,
    "move_cap": 0.98,
    "direct_init": 0.02,
    "gate_bias": -2.0,
    "relay_bias": 0.0,
    "occ_bias": 1.5,
    "dup_frac": 0.05,
    "max_dup": 3,
    "max_examples": 192,
    "nce_temp": 0.5,
    "read_weight": 1.0,
    "pred_weight": 0.5,
    "relay_weight": 0.25,
    "mse_weight": 0.05,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _inv_softplus(y):
    y = max(1e-4, float(y))
    return math.log(math.expm1(y)) if y < 20.0 else y


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


class _BackwardResolver(nn.Module):
    def __init__(self, d_model, key_d, addr_hidden, obs_hidden, temp_init, recency_init,
                 erase_init, erase_sharp_init, erase_thresh_init, move_cap, direct_init,
                 gate_bias, relay_bias, occ_bias):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = max(8, int(key_d))
        self.addr_h = max(32, int(addr_hidden))
        self.obs_h = max(16, int(obs_hidden))
        self.move_cap = min(0.999, max(0.05, float(move_cap)))

        self.addr_ln = nn.LayerNorm(self.d_model)
        self.addr_in = nn.Linear(self.d_model, self.addr_h)
        self.addr_mid = nn.Linear(self.addr_h, self.addr_h)
        self.addr_src = nn.Linear(self.addr_h, self.key_d, bias=False)
        self.addr_dst = nn.Linear(self.addr_h, self.key_d, bias=False)

        self.obs_in = nn.Linear(self.d_model, self.obs_h)
        self.relay_score = nn.Linear(self.addr_h + self.obs_h, 1)
        self.occ_score = nn.Linear(self.addr_h + self.obs_h, 1)
        nn.init.constant_(self.relay_score.bias, float(relay_bias))
        nn.init.constant_(self.occ_score.bias, float(occ_bias))

        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temp_init)))))
        self.recency_raw = nn.Parameter(torch.tensor(_inv_softplus(recency_init)))
        self.erase_raw = nn.Parameter(torch.tensor(_inv_softplus(erase_init)))
        self.erase_sharp = nn.Parameter(torch.tensor(float(erase_sharp_init)))
        self.erase_thresh = nn.Parameter(torch.tensor(float(erase_thresh_init)))
        self.null_logit = nn.Parameter(torch.zeros(1))

        self.read_score = nn.Linear(self.addr_h + 3, 1)
        nn.init.constant_(self.read_score.bias, float(gate_bias))
        self.read_out = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)
        self.direct_scale = nn.Parameter(
            torch.full((self.d_model,), float(direct_init)))

    def solve_unit_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        rows = []
        n = system.size(1)
        for i in range(n):
            yi = rhs[:, i, :]
            if rows:
                prev = torch.stack(rows, dim=1)
                yi = yi - torch.bmm(system[:, i:i + 1, :i], prev).squeeze(1)
            rows.append(yi)
        return torch.stack(rows, dim=1)

    def resolve(self, tok, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, _ = tok.shape
        n = (L + 1) // 2
        n_pair = L // 2
        if n < 2 or n_pair < 1:
            return None
        dev = tok.device
        dt = tok.dtype

        cmd = _clean(tok[:, 0::2, :])
        obs_p = _clean(tok[:, 1::2, :])
        if n_pair < n:
            obs = torch.cat([obs_p, obs_p.new_zeros(B, n - n_pair, self.d_model)], dim=1)
        else:
            obs = obs_p[:, :n, :]

        if key_pad is None:
            cmd_ok = torch.ones(B, n, dtype=torch.bool, device=dev)
            obs_ok = torch.ones(B, n, dtype=torch.bool, device=dev)
        else:
            live = ~key_pad.bool()
            cmd_ok = live[:, 0::2][:, :n]
            obs_ok = live[:, 1::2][:, :n]
            if obs_ok.size(1) < n:
                obs_ok = torch.cat(
                    [obs_ok, obs_ok.new_zeros(B, n - obs_ok.size(1))], dim=1)
        node_ok = cmd_ok & obs_ok
        node_f = node_ok.to(dt)

        a = F.gelu(self.addr_mid(F.gelu(self.addr_in(self.addr_ln(cmd)))))
        s = _unit(self.addr_src(a))
        dst = _unit(self.addr_dst(a))
        of = F.gelu(self.obs_in(obs))
        pair = torch.cat([a, of], dim=-1)

        m = self.move_cap * torch.sigmoid(self.relay_score(pair)).squeeze(-1) * node_f
        occ = torch.sigmoid(self.occ_score(pair)).squeeze(-1) * node_f
        key = _unit(m.unsqueeze(-1) * dst + (1.0 - m).unsqueeze(-1) * s)

        sim = torch.bmm(s, key.transpose(1, 2))

        idx = torch.arange(n, device=dev)
        strict = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        strict_f = strict.to(dt)

        sharpness = self.erase_sharp.clamp(0.1, 64.0)
        thresh = self.erase_thresh.clamp(-1.0, 1.0)
        taken = m.unsqueeze(-1) * torch.sigmoid(sharpness * (sim - thresh)) * strict_f
        run_max = torch.cummax(taken, dim=1).values
        away = torch.cat([run_max.new_zeros(B, 1, n), run_max[:, :-1, :]], dim=1)

        temp = self.log_temp.exp().clamp(0.02, 4.0)
        recency = F.softplus(self.recency_raw).clamp(0.0, 4.0)
        erase_w = F.softplus(self.erase_raw).clamp(0.0, 20.0)
        gap = (idx.view(n, 1) - idx.view(1, n) - 1.0).clamp_min(0.0).unsqueeze(0).to(dt)

        logits = (sim / temp
                  + torch.log(occ.clamp_min(1e-6)).unsqueeze(1)
                  - recency * gap
                  - erase_w * away)
        allowed = strict & node_ok.unsqueeze(1) & cmd_ok.unsqueeze(2)
        logits = logits.masked_fill(~allowed, _NEG)
        nullcol = self.null_logit.clamp(-30.0, 30.0).view(1, 1, 1).expand(B, n, 1).to(logits.dtype)
        att = torch.softmax(torch.cat([logits, nullcol], dim=-1), dim=-1)[..., :n] * strict_f

        eye = torch.eye(n, device=dev, dtype=att.dtype).unsqueeze(0)
        system = eye - att * m.unsqueeze(1)
        rhs = torch.bmm(att, obs * ((1.0 - m) * node_f).unsqueeze(-1))
        content = _clean(self.solve_unit_lower(system, rhs)).clamp(-1e4, 1e4)

        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        peak = att.amax(dim=2, keepdim=True)
        mass = att.sum(dim=2, keepdim=True)
        gate = torch.sigmoid(self.read_score(torch.cat([a, rms, peak, mass], dim=-1)))
        routed = self.read_out(content) + self.direct_scale.view(1, 1, -1) * content
        contrib = _clean(gate * routed) * cmd_ok.unsqueeze(-1).to(dt)

        return {"content": content, "contrib": contrib, "att": att, "move": m,
                "obs": obs, "node_ok": node_ok, "n_pair": n_pair}


def wrap(net, D, **params):
    existing = getattr(net, "_backward_resolver_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)
    mod = _BackwardResolver(
        int(D), int(cfg["key_d"]), int(cfg["addr_hidden"]), int(cfg["obs_hidden"]),
        float(cfg["temp_init"]), float(cfg["recency_init"]), float(cfg["erase_init"]),
        float(cfg["erase_sharp_init"]), float(cfg["erase_thresh_init"]),
        float(cfg["move_cap"]), float(cfg["direct_init"]), float(cfg["gate_bias"]),
        float(cfg["relay_bias"]), float(cfg["occ_bias"]),
    )
    net.backward_resolver = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 3:
            return pred, h
        res = mod.resolve(tok_emb, key_pad)
        if res is None:
            return pred, h
        contrib = res["contrib"]
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        k = min(contrib.size(1), cmd_view.size(1))
        cmd_view[:, :k, :] = cmd_view[:, :k, :] + contrib[:, :k, :].to(out.dtype)
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "res": res, "pred": out}
        return out, h

    net.forward = _forward
    net._backward_resolver_state = state
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
    pred_cmd = None
    if stash is not None and stash["tok"] is tok:
        res = stash["res"]
        pred_cmd = stash["pred"][:, 0::2, :]
    else:
        res = mod.resolve(tok, batch["key_pad"])
    if res is None:
        return 0.0

    npair = int(res["n_pair"])
    if npair < 3:
        return 0.0
    obs = res["obs"][:, :npair, :].float()
    content = res["content"][:, :npair, :].float()
    move = res["move"][:, :npair].float()
    active = res["node_ok"][:, :npair]
    B, n, Dm = obs.shape
    dim = float(Dm)
    dev = obs.device

    with torch.no_grad():
        osq = obs.pow(2).sum(-1)
        d2 = (osq.unsqueeze(2) + osq.unsqueeze(1)
              - 2.0 * torch.bmm(obs, obs.transpose(1, 2))).clamp_min(0.0) / dim
        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        off = (active.unsqueeze(2) & active.unsqueeze(1)) & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2.dtype)
        ref = (d2 * off).sum(dim=(1, 2)) / cnt
        dup = off & (d2 <= (float(cfg["dup_frac"]) * ref).view(B, 1, 1))

        clus = dup.sum(dim=2)
        max_dup = int(cfg["max_dup"])
        contentful = active & (clus <= max_dup)
        contentless = active & (clus > max_dup)

        idx = torch.arange(n, device=dev)
        lower = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        none_long = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)
        earlier = dup & lower & contentful.unsqueeze(2) & contentful.unsqueeze(1)
        has_e = earlier.any(dim=2)
        last_e = torch.where(earlier, idx.view(1, 1, n), none_long).amax(dim=2)

        mcum = torch.cumsum(contentless.long(), dim=1)
        prev_pos = (idx.view(1, n) - 1).clamp_min(0).expand(B, n)
        hops = (torch.gather(mcum, 1, prev_pos)
                - torch.gather(mcum, 1, last_e.clamp_min(0))).clamp_min(0)
        mined = has_e & contentful & (hops > 0)

    total = None
    relay_w = float(cfg["relay_weight"])
    if relay_w > 0.0:
        wf = active.to(content.dtype)
        denom = wf.sum().clamp_min(1.0)
        prob = (move / mod.move_cap).clamp(1e-4, 1.0 - 1e-4)
        lab = contentless.to(content.dtype)
        bce = -(lab * torch.log(prob) + (1.0 - lab) * torch.log(1.0 - prob))
        total = relay_w * ((bce * wf).sum() / denom)

    with torch.no_grad():
        nz = torch.nonzero(mined, as_tuple=False)
    if nz.numel() > 0:
        with torch.no_grad():
            b_i = nz[:, 0]
            t_i = nz[:, 1]
            cap = int(cfg["max_examples"])
            if b_i.numel() > cap:
                _, order = torch.topk(hops[b_i, t_i].float(), cap)
                b_i = b_i[order]
                t_i = t_i[order]
            seen = idx.view(1, n) <= t_i.view(-1, 1)
            cand_mask = contentful[b_i] & seen
            pos_mask = (dup[b_i, t_i] | F.one_hot(t_i, n).bool()) & cand_mask
            cand_sq = osq[b_i]

        cand = obs[b_i]
        temp = max(1e-3, float(cfg["nce_temp"]))

        def _nce(z):
            dot = torch.bmm(cand, z.unsqueeze(2)).squeeze(2)
            dz = (z.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / dim
            lg = (-dz / temp).masked_fill(~cand_mask, _NEG)
            lp = torch.log_softmax(lg, dim=1)
            return -(torch.logsumexp(lp.masked_fill(~pos_mask, _NEG), dim=1)).mean()

        routed = content[b_i, t_i]
        truth = obs[b_i, t_i]
        term = float(cfg["read_weight"]) * _nce(routed)
        term = term + float(cfg["mse_weight"]) * (routed - truth).pow(2).mean()
        if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
            pc = pred_cmd[:, :n, :].float()
            term = term + float(cfg["pred_weight"]) * _nce(pc[b_i, t_i])
        total = term if total is None else total + term

    if total is None:
        return 0.0
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
        vals["key_d"] >= 8.0,
        vals["addr_hidden"] >= 32.0,
        vals["obs_hidden"] >= 16.0,
        0.0 < vals["temp_init"] <= 10.0,
        vals["recency_init"] >= 0.0,
        vals["erase_init"] >= 0.0,
        0.0 < vals["erase_sharp_init"] <= 64.0,
        -1.0 <= vals["erase_thresh_init"] <= 1.0,
        0.0 < vals["move_cap"] < 1.0,
        0.0 < abs(vals["direct_init"]) <= 1.0,
        -20.0 <= vals["gate_bias"] <= 20.0,
        -20.0 <= vals["relay_bias"] <= 20.0,
        -20.0 <= vals["occ_bias"] <= 20.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["relay_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)

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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0037 (r5-18-relsam-spectral-transport, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
