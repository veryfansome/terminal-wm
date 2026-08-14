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
